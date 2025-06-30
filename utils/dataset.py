import os
import torch
import numpy as np
from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from torchvision import datasets
import torchvision
from torchvision.transforms import functional as F
from typing import Any, Callable, Dict, List, Optional, Tuple

def pil_loader(path):
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')

class FewShotImageFolder(torch.utils.data.Dataset):
    # set default seed=None, check the randomness
    def __init__(self, root, transform=None, N=1000, K=-1, few_samples=-1, seed=None):
        super(FewShotImageFolder, self).__init__()
        self.root = os.path.abspath(os.path.expanduser(root))
        self._transform = transform
        # load and parse from a txt file
        self.N = N
        self.K = K
        self.few_samples = few_samples
        self.seed = seed
        self.samples = self._parse_and_sample()
    
    def samples_to_file(self, save_path):
        with open(save_path, "w") as f:
            for (path, label) in self.samples:
                f.writelines("{}, {}\n".format(path.replace(self.root, "."), label))
        print("Writing train samples into {}".format(os.path.abspath(save_path)))

    def __parse(self):
        file_path = os.path.join(self.root, "train.txt")
        full_data = {}
        with open(file_path, "r") as f:
            raw_data = f.readlines()
        for rd in raw_data:
            img_path, target = rd.replace("\n", "").split()
            assert target.isalnum()
            if target not in full_data.keys():
                full_data[target] = []
            full_data[target].append(img_path)
        return full_data
    
    def _parse_and_sample(self):
        N, K, seed = self.N, self.K, self.seed
        assert 1<=N<=1000, r"N with maximum num 1000"
        assert K<=500, r"If you want to use the whole dataset, set K=-1"
        # txt default path: self.root + "/train.txt"
        full_data = self.__parse()
        all = 0
        for v in full_data.values():
            all += len(v)
        print("Full dataset has {} classes and {} images.".format(len(full_data), all))
        print("Using seed={} to sample images.".format(seed))
        sampled_data = []

        np.random.seed(seed)
        # sample classes
        if self.few_samples > 0:
            for i in range(self.few_samples):
                while True:
                    sampled_cls = np.random.choice(list(full_data.keys()), 1, replace=False)
                    cls = sampled_cls[0]
                    sampled_img = np.random.choice(full_data[cls], 1, replace=False)[0]
                    curr_sample = (os.path.join(self.root, "train", sampled_img), cls)
                    if curr_sample not in sampled_data:
                        sampled_data.append(curr_sample)
                        break
            print("Final samples: {}".format(len(sampled_data)))
        else:
            sampled_cls = np.random.choice(list(full_data.keys()), N, replace=False)
            sampled_cls.sort()
            for cls in sampled_cls:
                if K == -1:
                    # use all data
                    sampled_imgs = full_data[cls]
                else:
                    # sample images of every class
                    sampled_imgs = np.random.choice(full_data[cls], K, replace=False)
                sampled_data += [(os.path.join(self.root, "train", i), cls) for i in sorted(sampled_imgs)]
        
        self.idx_to_class = {}
        self.class_to_idx = {}
        for k, v in full_data.items():
            idx = k
            cls = v[0].split("/")[0]
            self.class_to_idx[cls] = idx
            self.idx_to_class[idx] = cls
        self.classes = list(self.idx_to_class.values())
        self._full_data = full_data
        return sampled_data
        
    def __getitem__(self, index):
        path, label = self.samples[index]
        img = pil_loader(path)
        if self._transform is not None:
            img = self._transform(img)
        return img, int(label)

    def __len__(self):
        return len(self.samples)

    def __repr__(self) -> str:
        return super().__repr__()

def imagenet(train, batch_size, workers, sub_idx=None, imagenet_path = '/your/datasets/imagenet'):
    if train:
        transform = torchvision.transforms.Compose([
            torchvision.transforms.RandomResizedCrop(224),
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        loader = torch.utils.data.DataLoader(
            datasets.ImageFolder(os.path.join(imagenet_path, 'train'), transform), 
            batch_size=batch_size, shuffle=True, num_workers=workers, pin_memory=False)
    else:
        transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize(256),
            torchvision.transforms.CenterCrop(224),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        loader = torch.utils.data.DataLoader(
            datasets.ImageFolder(os.path.join(imagenet_path, 'val'), transform), 
            batch_size=batch_size, shuffle=False, num_workers=workers, pin_memory=False)
    return loader

def imagenet_fewshot(img_num=1000, batch_size=64, workers=0, seed=2021, train=True, imagenet_path = '/your/datasets/imagenet', rand_sample = False):
     
    if img_num < 1000 :
        few_samples = img_num
        N = 1000
        K = -1
    else:
        few_samples = -1
        N = 1000
        K = img_num // N
    if rand_sample:
        K = -1

    if train:
        transform = torchvision.transforms.Compose([
                torchvision.transforms.RandomResizedCrop(224),
                torchvision.transforms.RandomHorizontalFlip(),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        shuffle = True
    else:
        transform = torchvision.transforms.Compose([
                torchvision.transforms.Resize(256),
                torchvision.transforms.CenterCrop(224),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        shuffle = False
        
    dataset = FewShotImageFolder(
        imagenet_path,
        transform,
        N=N, K=K, few_samples=few_samples, seed=seed)

    # if not os.path.exists(save_dir+'dataset'):
    #     os.makedirs(save_dir+'dataset')
    # for idx in range(len(dataset.samples)):
    #     original_path, label = dataset.samples[idx]
    #     new_path = save_dir +'dataset/' + original_path.split('/')[-1]
    #     shutil.copy(original_path, new_path)
    #     dataset.samples[idx] = (new_path, label)

    drop_last=False
    # if train and len(dataset) >= batch_size:
    #     drop_last = True

    loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, 
        num_workers=workers, pin_memory=False, drop_last=drop_last
    )
    return loader

'''' voc2012 '''
def get_modules(use_v2):
    # We need a protected import to avoid the V2 warning in case just V1 is used
    if use_v2:
        import torchvision.transforms.v2
        import torchvision.tv_tensors
        import v2_extras

        return torchvision.transforms.v2, torchvision.tv_tensors, v2_extras
    else:
        import utils.transforms as transforms

        return transforms, None, None
    
class SegmentationPresetTrain:
    def __init__(
        self,
        *,
        base_size,
        crop_size,
        hflip_prob=0.5,
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
        backend="pil",
        use_v2=False,
    ):
        T, tv_tensors, v2_extras = get_modules(use_v2)

        transforms = []
        backend = backend.lower()
        if backend == "tv_tensor":
            transforms.append(T.ToImage())
        elif backend == "tensor":
            transforms.append(T.PILToTensor())
        elif backend != "pil":
            raise ValueError(f"backend can be 'tv_tensor', 'tensor' or 'pil', but got {backend}")

        transforms += [T.RandomResize(min_size=int(0.5 * base_size), max_size=int(2.0 * base_size))]

        if hflip_prob > 0:
            transforms += [T.RandomHorizontalFlip(hflip_prob)]

        if use_v2:
            # We need a custom pad transform here, since the padding we want to perform here is fundamentally
            # different from the padding in `RandomCrop` if `pad_if_needed=True`.
            transforms += [v2_extras.PadIfSmaller(crop_size, fill={tv_tensors.Mask: 255, "others": 0})]

        transforms += [T.RandomCrop(crop_size)]

        if backend == "pil":
            transforms += [T.PILToTensor()]

        if use_v2:
            img_type = tv_tensors.Image if backend == "tv_tensor" else torch.Tensor
            transforms += [
                T.ToDtype(dtype={img_type: torch.float32, tv_tensors.Mask: torch.int64, "others": None}, scale=True)
            ]
        else:
            # No need to explicitly convert masks as they're magically int64 already
            transforms += [T.ToDtype(torch.float, scale=True)]

        transforms += [T.Normalize(mean=mean, std=std)]
        if use_v2:
            transforms += [T.ToPureTensor()]

        self.transforms = T.Compose(transforms)

    def __call__(self, img, target):
        return self.transforms(img, target)

class SegmentationPresetEval:
    def __init__(
        self, *, base_size, crop_size, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), backend="pil", use_v2=False
    ):
        T, _, _ = get_modules(use_v2)

        transforms = []
        backend = backend.lower()
        if backend == "tensor":
            transforms += [T.PILToTensor()]
        elif backend == "tv_tensor":
            transforms += [T.ToImage()]
        elif backend != "pil":
            raise ValueError(f"backend can be 'tv_tensor', 'tensor' or 'pil', but got {backend}")

        if use_v2:
            transforms += [T.Resize(size=(base_size, base_size))]
        else:
            # transforms += [T.RandomResize(min_size=int(0.5 * base_size), max_size=int(2.0 * base_size))]
            transforms += [T.Resize(crop_size,crop_size)]

        if backend == "pil":
            # Note: we could just convert to pure tensors even in v2?
            transforms += [T.ToImage() if use_v2 else T.PILToTensor()]

        transforms += [
            T.ToDtype(torch.float, scale=True),
            T.Normalize(mean=mean, std=std),
        ]
        if use_v2:
            transforms += [T.ToPureTensor()]

        self.transforms = T.Compose(transforms)

    def __call__(self, img, target):
        return self.transforms(img, target)

def cat_list(images, fill_value=0):
    max_size = tuple(max(s) for s in zip(*[img.shape for img in images]))
    batch_shape = (len(images),) + max_size
    batched_imgs = images[0].new(*batch_shape).fill_(fill_value)
    for img, pad_img in zip(images, batched_imgs):
        pad_img[..., : img.shape[-2], : img.shape[-1]].copy_(img)
    return batched_imgs
def collate_fn(batch):
    images, targets = list(zip(*batch))
    batched_imgs = cat_list(images, fill_value=0)
    batched_targets = cat_list(targets, fill_value=255)
    return batched_imgs, batched_targets

class FlexibleVOCSegmentation(torchvision.datasets.voc._VOCBase):
    _SPLITS_DIR = "Segmentation"
    _TARGET_DIR = "SegmentationClass"
    _TARGET_FILE_EXT = ".png"

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is the image segmentation.
        """
        img = Image.open(self.images[index]).convert("RGB")
        target = Image.open(self.targets[index])

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

def voc2012(root, image_set, batch_size, num_workers):
    if image_set == 'train':
        trans = SegmentationPresetTrain(base_size=520, crop_size=480, backend="PIL", use_v2=False)
    else:
        trans = SegmentationPresetEval(base_size=520, crop_size=480, backend="PIL", use_v2=False)

    ds = FlexibleVOCSegmentation(root=root,image_set=image_set,download=False,transforms=trans)

    if image_set == 'train':
        sampler = torch.utils.data.RandomSampler(ds)
    else:
        sampler = torch.utils.data.SequentialSampler(ds)

    loader = torch.utils.data.DataLoader(ds,batch_size=batch_size,sampler=sampler,num_workers=num_workers,collate_fn=collate_fn,drop_last=False)
    return loader

def voc2012_fewshot(num_samples, seed, save_dir, root, batch_size, num_workers):
    trans = SegmentationPresetTrain(base_size=520, crop_size=480, backend="PIL", use_v2=False)
    ds = FlexibleVOCSegmentation(root=root,image_set='train',download=False,transforms=trans)
    images, targets = ds.images, ds.targets
    gen = torch.Generator()
    gen.manual_seed(seed)
    random_indices = torch.randperm(len(images), generator=gen)[:num_samples]
    new_images = [images[i.item()] for i in random_indices]
    new_targets = [targets[i.item()] for i in random_indices]
    ds.images = new_images
    ds.targets = new_targets
    sampler = torch.utils.data.RandomSampler(ds)
    loader = torch.utils.data.DataLoader(ds,batch_size=batch_size,sampler=sampler,num_workers=num_workers,collate_fn=collate_fn,drop_last=False)
    with open(save_dir + "samples.txt", "w") as f:
        for image in new_images:
            f.writelines("{}\n".format(image))
    return loader
