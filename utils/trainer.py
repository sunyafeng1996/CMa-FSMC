import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np


def get_single_mask(featuremap, HW=(224,224), th=0.2):
    B, C, H_orig, W_orig = featuremap.shape
    num_elements = H_orig * W_orig
    threshold_index = int(th * (num_elements - 1))
    sorted_abc, _ = torch.sort(featuremap.view(B, C, -1), dim=-1, descending=True)
    Tbc = sorted_abc[:, :, threshold_index]
    Abc_interpolated = F.interpolate(
        featuremap, size=(HW[0], HW[1]), mode="bilinear", align_corners=False
    )
    Tbc = Tbc.view(B, C, 1, 1)
    mask_cond = Tbc == 0
    mask1 = Abc_interpolated != 0
    mask2 = Abc_interpolated >= Tbc 
    masks = torch.where(mask_cond, mask1, mask2)
    return masks

def concept_awared_cutmix(samples, trainer, model, last_layer, mode, y = None, th = 0.2):
    # featuremap
    if last_layer:
        if 'FCN' in str(type(model)):
            with torch.no_grad():
                _ = model(samples)
                fm = trainer.layer_outputs['model']
        else:
            _, fm = model(samples,out_feats=True)
            fm = fm[0]
    else:
        fm = F.relu(model.bn1(model.conv1(samples)))
    # aggregation
    fm = fm.sum(dim=1).unsqueeze(1)
    # mask
    masks = get_single_mask(fm, HW=(samples.shape[2],samples.shape[3]), th=th)
    masks = masks.float()
    batch_size = samples.size(0)
    index = torch.randperm(batch_size).to(samples.device)  # Randomly shuffle the batch
    if mode == 'cover':
        mask_temp = (masks - 1).abs()
        mixed_samples = samples * mask_temp[index] + samples[index] * masks[index]
    elif mode == 'fuse':
        mixed_samples = samples + samples[index] * masks[index]
    if y != None:
        return mixed_samples, y, y[index], torch.tensor([0.5]).to(samples.device)
    else:
        return mixed_samples

def gridmix_two(image1, image2, S_max=5, kappa=0.5):
    C, H, W = image1.shape
    S = torch.randint(1, S_max + 1, (1,)).item()
    lambda_grid = torch.distributions.Beta(kappa, kappa).sample((S, S))
    h_offset = torch.empty(1).uniform_(-H/(2*S), H/(2*S)).item()
    w_offset = torch.empty(1).uniform_(-W/(2*S), W/(2*S)).item()
    y_coords = torch.linspace(0, H, S + 1) + h_offset
    x_coords = torch.linspace(0, W, S + 1) + w_offset
    mask = torch.zeros(H, W)
    for i in range(S):
        for j in range(S):
            y_start = max(0, int(y_coords[i].item()))
            y_end = min(H, int(y_coords[i + 1].item()))
            x_start = max(0, int(x_coords[j].item()))
            x_end = min(W, int(x_coords[j + 1].item()))
            if y_start >= y_end or x_start >= x_end:
                continue
            mask[y_start:y_end, x_start:x_end] = lambda_grid[i, j]
    
    mask = mask.unsqueeze(0).to(image1.device)
    mixed_image = mask * image1 + image2
    return mixed_image

def gridmix(x, S_max=5, kappa=0.5):
    B = x.shape[0]
    index = torch.randperm(B).to(x.device)
    mixed_image = torch.zeros_like(x)
    for i in range(B):
        mixed_image[i] = gridmix_two(x[i],x[index[i]],S_max,kappa)
    return mixed_image

def mixup(x, y = None, alpha=0.4):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)  # Random lambda from Beta distribution
    else:
        lam = 1.0
    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)  # Randomly shuffle the batch
    mixed_x = lam * x + (1 - lam) * x[index]  # Mix the inputs
    if y != None:
        return mixed_x, y, y[index], lam
    else:
        return mixed_x

def cutmix(data, y = None, alpha=1.0):
    if alpha <= 0:
        return data, 1.0
    batch_size, _, h, w = data.size()
    indices = torch.randperm(batch_size, device=data.device)
    shuffled_data = data[indices]
    lam = torch.distributions.beta.Beta(alpha, alpha).sample().to(data.device)
    cut_ratio = torch.sqrt(1. - lam)
    cut_w = (w * cut_ratio).type(torch.int)
    cut_h = (h * cut_ratio).type(torch.int)
    cx = torch.randint(0, w, (1,), device=data.device)
    cy = torch.randint(0, h, (1,), device=data.device)
    x1 = torch.clamp(cx - cut_w // 2, 0, w)
    y1 = torch.clamp(cy - cut_h // 2, 0, h)
    x2 = torch.clamp(cx + cut_w // 2, 0, w)
    y2 = torch.clamp(cy + cut_h // 2, 0, h)
    mixed_data = data.clone()
    mixed_data[:, :, x1:x2, y1:y2] = shuffled_data[:, :, x1:x2, y1:y2]
    if y != None:
        return mixed_data, y, y[indices], lam
    else:
        return mixed_data
    
class MiRTrainer():
    def __init__(self, model, pm, sampled_loader, lr, weight_decay, momentum, decrease_lr, epochs, gamma, \
                 eval_frequency, device, logger, save_dir, evaluator, data_enhancer = "none"):
        self.model = model.to(device).eval()
        self.pm = pm.to(device)
        if 'MobileNetV2' in str(type(model)):
            for param in self.pm.classifier.parameters():
                param.requires_grad = False
        elif 'ResNet' in str(type(model)):
            for param in self.pm.fc.parameters():
                param.requires_grad = False
        elif 'FCN' in str(type(model)):
            for param in self.pm.final_block.parameters():
                param.requires_grad = False
            for param in self.pm.aux_block.parameters():
                param.requires_grad = False
        self.sampled_loader = sampled_loader
        self.lr = lr
        self.weight_decay = weight_decay
        self.momentum = momentum
        self.decrease_lr = decrease_lr
        self.epochs = epochs
        self.gamma = gamma
        self.device = device
        self.eval_frequency = eval_frequency
        self.logger = logger
        self.save_dir = save_dir
        self.evaluator = evaluator
        self.optimizer = optim.SGD(pm.parameters(), lr=self.lr,weight_decay=self.weight_decay,momentum=self.momentum)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, self.decrease_lr * self.epochs, gamma=self.gamma)
        self.criterion = torch.nn.MSELoss(reduction='mean')
        self.loss_recoder = []
        self.acc_recoder = {}
        self.data_enhancer = data_enhancer

    def one_epoch(self):
        self.pm.train()
        losses = 0
        for samples, _ in self.sampled_loader:
            samples = samples.to(self.device)
            if self.data_enhancer == "CMa":
                mixed_samples = concept_awared_cutmix(samples, self, self.model, last_layer = True, mode = 'cover')
            elif self.data_enhancer == "none":
                mixed_samples = samples
            elif self.data_enhancer == "mixup":
                mixed_samples = mixup(samples)
            elif self.data_enhancer == "gridmix":
                mixed_samples = gridmix(samples)
            elif self.data_enhancer == "cutmix":
                mixed_samples = cutmix(samples)
            if 'FCN' in str(type(self.pm)):
                s_feats = self.pm(mixed_samples)
                with torch.no_grad():
                    t_feats = self.model(mixed_samples)
            else:
                _, s_feats = self.pm(mixed_samples,out_feats=True)
                with torch.no_grad():
                    _, t_feats = self.model(mixed_samples,out_feats=True)
            s_feat, t_feat = s_feats[0], t_feats[0]
            loss = self.criterion(s_feat, t_feat)
            losses += loss.item()
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        self.loss_recoder.append(losses/len(self.sampled_loader.dataset))
    
    def train(self):
        self.logger.info('==> Start training')
        torch.backends.cudnn.benchmark = True
        for epoch in range(self.epochs):
            self.pm.train()
            self.one_epoch()
            self.scheduler.step()
            self.logger.info('    [Epoch {}]  LR = {}  Train Loss = {:.4f}'.format(epoch, self.optimizer.param_groups[0]['lr'], self.loss_recoder[epoch]))
            if epoch != 0 and (epoch % self.eval_frequency == 0 or epoch == self.epochs-1):
                self.pm.eval()
                self.logger.info('==> Under evaluation')
                eval_results = self.evaluator(self.pm, device=self.device)
                if 'FCN' in str(type(self.model)):
                    acc, miou = eval_results[0], eval_results[1]
                    self.acc_recoder[epoch] = (acc, miou)
                    self.logger.info('    [model] Pixel Acc = {:.2%}  mIOU = {:.2%}'.format(acc, miou))
                else:
                    (acc1, acc5), val_loss = eval_results['Acc'], eval_results['Loss']
                    self.acc_recoder[epoch] = (acc1, acc5)
                    self.logger.info('    [Epoch {}]  LR = {}  Top1 = {:.4f}  Top5 = {:.4f}  Val Loss = {:.4f}'.format(
                        epoch, self.optimizer.param_groups[0]['lr'], acc1, acc5, val_loss))
                torch.save(self.pm.state_dict(), self.save_dir+'model.pth')
                torch.save(self.loss_recoder, self.save_dir+'loss_recoder.pth')
                torch.save(self.acc_recoder, self.save_dir+'acc_recoder.pth')

class KDTrainer():
    def __init__(self, model, pm, sampled_loader, lr, weight_decay, momentum, decrease_lr, epochs, gamma, \
                 eval_frequency, device, logger, save_dir, evaluator, data_enhancer = "none", temperature = 4, alpha = 0.1):
        self.model = model.to(device).eval()
        self.pm = pm.to(device)
        if 'MobileNetV2' in str(type(model)):
            for param in self.pm.classifier.parameters():
                param.requires_grad = False
        elif 'ResNet' in str(type(model)):
            for param in self.pm.fc.parameters():
                param.requires_grad = False
        elif 'FCN' in str(type(model)):
            for param in self.pm.final_block.parameters():
                param.requires_grad = False
            for param in self.pm.aux_block.parameters():
                param.requires_grad = False
        self.sampled_loader = sampled_loader
        self.lr = lr
        self.weight_decay = weight_decay
        self.momentum = momentum
        self.decrease_lr = decrease_lr
        self.epochs = epochs
        self.gamma = gamma
        self.device = device
        self.eval_frequency = eval_frequency
        self.logger = logger
        self.save_dir = save_dir
        self.evaluator = evaluator
        self.optimizer = optim.SGD(pm.parameters(), lr=self.lr,weight_decay=self.weight_decay,momentum=self.momentum)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, self.decrease_lr * self.epochs, gamma=self.gamma)
        self.criterion = torch.nn.CrossEntropyLoss()
        self.loss_recoder = []
        self.acc_recoder = {}
        self.data_enhancer = data_enhancer
        self.alpha = alpha
        self.temperature = temperature

    def one_epoch(self):
        self.pm.train()
        ce_loss = torch.nn.CrossEntropyLoss()
        kl_loss = torch.nn.KLDivLoss(reduction='batchmean')
        losses = 0
        for samples, labels in self.sampled_loader:
            samples, labels = samples.to(self.device), labels.to(self.device)
            if self.data_enhancer == "CMa":
                mixed_samples, labels, mixed_labels, lam = concept_awared_cutmix(samples, self, self.model, last_layer = True, mode = 'cover', y = labels)
            elif self.data_enhancer == "none":
                mixed_samples, labels, mixed_labels, lam = samples, labels, labels, 1
            elif self.data_enhancer == "mixup":
                mixed_samples, labels, mixed_labels, lam = mixup(samples, labels)
            elif self.data_enhancer == "cutmix":
                mixed_samples, labels, mixed_labels, lam = cutmix(samples, labels)
            s_out = self.pm(mixed_samples,out_feats=False)
            student_probs = torch.nn.functional.log_softmax(s_out / self.temperature, dim=1)
            with torch.no_grad():
                t_out = self.model(mixed_samples,out_feats=False)
                teacher_probs = torch.nn.functional.softmax(t_out / self.temperature, dim=1)
            loss_ce = lam * ce_loss(s_out, labels) + (1 - lam) * ce_loss(s_out, mixed_labels)
            loss_kd = kl_loss(student_probs, teacher_probs.detach()) * (self.temperature**2)
            loss = self.alpha * loss_ce + (1 - self.alpha) * loss_kd
            losses += loss.item()
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        self.loss_recoder.append(losses/len(self.sampled_loader.dataset))
    
    def train(self):
        self.logger.info('==> Start training')
        torch.backends.cudnn.benchmark = True
        for epoch in range(self.epochs):
            self.pm.train()
            self.one_epoch()
            self.scheduler.step()
            self.logger.info('    [Epoch {}]  LR = {}  Train Loss = {:.4f}'.format(epoch, self.optimizer.param_groups[0]['lr'], self.loss_recoder[epoch]))
            if epoch != 0 and (epoch % self.eval_frequency == 0 or epoch == self.epochs-1):
                self.pm.eval()
                self.logger.info('==> Under evaluation')
                eval_results = self.evaluator(self.pm, device=self.device)
                if 'FCN' in str(type(self.model)):
                    acc, miou = eval_results[0], eval_results[1]
                    self.acc_recoder[epoch] = (acc, miou)
                    self.logger.info('    [model] Pixel Acc = {:.2%}  mIOU = {:.2%}'.format(acc, miou))
                else:
                    (acc1, acc5), val_loss = eval_results['Acc'], eval_results['Loss']
                    self.acc_recoder[epoch] = (acc1, acc5)
                    self.logger.info('    [Epoch {}]  LR = {}  Top1 = {:.4f}  Top5 = {:.4f}  Val Loss = {:.4f}'.format(
                        epoch, self.optimizer.param_groups[0]['lr'], acc1, acc5, val_loss))
                torch.save(self.pm.state_dict(), self.save_dir+'model.pth')
                torch.save(self.loss_recoder, self.save_dir+'loss_recoder.pth')
                torch.save(self.acc_recoder, self.save_dir+'acc_recoder.pth')

class BPTrainer():
    def __init__(self, model, pm, sampled_loader, lr, weight_decay, momentum, decrease_lr, epochs, gamma, \
                 eval_frequency, device, logger, save_dir, evaluator, data_enhancer = "none"):
        self.pm = pm.to(device)
        self.model = model.to(device).eval()
        if 'MobileNetV2' in str(type(model)):
            for param in self.pm.classifier.parameters():
                param.requires_grad = False
        elif 'ResNet' in str(type(model)):
            for param in self.pm.fc.parameters():
                param.requires_grad = False
        elif 'FCN' in str(type(model)):
            for param in self.pm.final_block.parameters():
                param.requires_grad = False
            for param in self.pm.aux_block.parameters():
                param.requires_grad = False
        self.sampled_loader = sampled_loader
        self.lr = lr
        self.weight_decay = weight_decay
        self.momentum = momentum
        self.decrease_lr = decrease_lr
        self.epochs = epochs
        self.gamma = gamma
        self.device = device
        self.eval_frequency = eval_frequency
        self.logger = logger
        self.save_dir = save_dir
        self.evaluator = evaluator
        self.optimizer = optim.SGD(pm.parameters(), lr=self.lr,weight_decay=self.weight_decay,momentum=self.momentum)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, self.decrease_lr * self.epochs, gamma=self.gamma)
        self.criterion = torch.nn.CrossEntropyLoss()
        self.loss_recoder = []
        self.acc_recoder = {}
        self.data_enhancer = data_enhancer

    def one_epoch(self):
        self.pm.train()
        ce_loss = torch.nn.CrossEntropyLoss()
        losses = 0
        for samples, labels in self.sampled_loader:
            samples, labels = samples.to(self.device), labels.to(self.device)
            if self.data_enhancer == "CMa":
                mixed_samples, labels, mixed_labels, lam = concept_awared_cutmix(samples, self, self.model, last_layer = True, mode = 'cover', y = labels)
            elif self.data_enhancer == "none":
                mixed_samples, labels, mixed_labels, lam = samples, labels, labels, 1
            elif self.data_enhancer == "mixup":
                mixed_samples, labels, mixed_labels, lam = mixup(samples, labels)
            elif self.data_enhancer == "cutmix":
                mixed_samples, labels, mixed_labels, lam = cutmix(samples, labels)
            s_out = self.pm(mixed_samples,out_feats=False)
            loss = lam * ce_loss(s_out, labels) + (1 - lam) * ce_loss(s_out, mixed_labels)
            losses += loss.item()
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        self.loss_recoder.append(losses/len(self.sampled_loader.dataset))
    
    def train(self):
        self.logger.info('==> Start training')
        torch.backends.cudnn.benchmark = True
        for epoch in range(self.epochs):
            self.pm.train()
            self.one_epoch()
            self.scheduler.step()
            self.logger.info('    [Epoch {}]  LR = {}  Train Loss = {:.4f}'.format(epoch, self.optimizer.param_groups[0]['lr'], self.loss_recoder[epoch]))
            if epoch != 0 and (epoch % self.eval_frequency == 0 or epoch == self.epochs-1):
                self.pm.eval()
                self.logger.info('==> Under evaluation')
                eval_results = self.evaluator(self.pm, device=self.device)
                if 'FCN' in str(type(self.model)):
                    acc, miou = eval_results[0], eval_results[1]
                    self.acc_recoder[epoch] = (acc, miou)
                    self.logger.info('    [model] Pixel Acc = {:.2%}  mIOU = {:.2%}'.format(acc, miou))
                else:
                    (acc1, acc5), val_loss = eval_results['Acc'], eval_results['Loss']
                    self.acc_recoder[epoch] = (acc1, acc5)
                    self.logger.info('    [Epoch {}]  LR = {}  Top1 = {:.4f}  Top5 = {:.4f}  Val Loss = {:.4f}'.format(
                        epoch, self.optimizer.param_groups[0]['lr'], acc1, acc5, val_loss))
                torch.save(self.pm.state_dict(), self.save_dir+'model.pth')
                torch.save(self.loss_recoder, self.save_dir+'loss_recoder.pth')
                torch.save(self.acc_recoder, self.save_dir+'acc_recoder.pth')

class CMaTrainer():
    def __init__(self, model, pm, sampled_loader, lr, weight_decay, momentum, decrease_lr, epochs, gamma, \
                 eval_frequency, device, logger, save_dir, evaluator, th):
        self.model = model.to(device).eval()
        self.pm = pm.to(device)
        if 'MobileNetV2' in str(type(model)):
            for param in self.pm.classifier.parameters():
                param.requires_grad = False
        elif'ResNet' in str(type(model)):
            for param in self.pm.fc.parameters():
                param.requires_grad = False
        elif 'FCN' in str(type(model)):
            for param in self.pm.final_block.parameters():
                param.requires_grad = False
            for param in self.pm.aux_block.parameters():
                param.requires_grad = False
            ## hook
            self.layer_outputs = {
            'model': None,
            'pm': None
            }
            layer_name = 'backbone.4.unit3.body'
            module1 = self.model.get_submodule(layer_name)
            module2 = self.pm.get_submodule(layer_name)
            def hook_function(module, input, output, key):
                self.layer_outputs[key] = output
            self.hook1 = module1.register_forward_hook(lambda m, i, o: hook_function(m, i, o, 'model'))
            self.hook2 = module2.register_forward_hook(lambda m, i, o: hook_function(m, i, o, 'pm'))
        self.sampled_loader = sampled_loader
        self.lr = lr
        self.weight_decay = weight_decay
        self.momentum = momentum
        self.decrease_lr = decrease_lr
        self.epochs = epochs
        self.gamma = gamma
        self.device = device
        self.eval_frequency = eval_frequency
        self.logger = logger
        self.save_dir = save_dir
        self.evaluator = evaluator
        self.th = th
        self.optimizer = optim.SGD(pm.parameters(), lr=self.lr,weight_decay=self.weight_decay,momentum=self.momentum)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, self.decrease_lr * self.epochs, gamma=self.gamma)
        self.criterion = torch.nn.MSELoss(reduction='mean')
        self.loss_recoder = []
        self.acc_recoder = {}

    def one_epoch(self):
        self.pm.train()
        losses = 0
        for samples, _ in self.sampled_loader:
            samples = samples.to(self.device)
            mixed_samples = concept_awared_cutmix(samples, self, self.model, last_layer = True, mode = 'cover', th = self.th)
            if 'FCN' in str(type(self.pm)):
                s_feats = self.pm(mixed_samples)
                with torch.no_grad():
                    t_feats = self.model(mixed_samples)
            else:
                _, s_feats = self.pm(mixed_samples,out_feats=True)
                with torch.no_grad():
                    _, t_feats = self.model(mixed_samples,out_feats=True)
            s_feat, t_feat = s_feats[0], t_feats[0]
            loss = self.criterion(s_feat, t_feat)
            losses += loss.item()
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        self.loss_recoder.append(losses/len(self.sampled_loader.dataset))
    
    def train(self):
        self.logger.info('==> Start training')
        for epoch in range(self.epochs):
            self.pm.train()
            self.one_epoch()
            self.scheduler.step()
            self.logger.info('    [Epoch {}]  LR = {}  Train Loss = {:.4f}'.format(epoch, self.optimizer.param_groups[0]['lr'], self.loss_recoder[epoch]))
            if epoch != 0 and (epoch % self.eval_frequency == 0 or epoch == self.epochs-1):
                self.pm.eval()
                self.logger.info('==> Under evaluation')
                eval_results = self.evaluator(self.pm, device=self.device)
                if 'FCN' in str(type(self.model)):
                    acc, miou = eval_results[0], eval_results[1]
                    self.acc_recoder[epoch] = (acc, miou)
                    self.logger.info('    [model] Pixel Acc = {:.2%}  mIOU = {:.2%}'.format(acc, miou))
                else:
                    (acc1, acc5), val_loss = eval_results['Acc'], eval_results['Loss']
                    self.acc_recoder[epoch] = (acc1, acc5)
                    self.logger.info('    [Epoch {}]  LR = {}  Top1 = {:.4f}  Top5 = {:.4f}  Val Loss = {:.4f}'.format(
                        epoch, self.optimizer.param_groups[0]['lr'], acc1, acc5, val_loss))
                torch.save(self.pm.state_dict(), self.save_dir+'model.pth')
                torch.save(self.loss_recoder, self.save_dir+'loss_recoder.pth')
                torch.save(self.acc_recoder, self.save_dir+'acc_recoder.pth')
