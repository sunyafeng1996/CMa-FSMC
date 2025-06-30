import numpy as np

from utils.compute_flops_paras import get_flops_paras
from utils.prune_utils import l1_pruning_with_prs_for_fcn_resnet101, l1_pruning_with_prs_for_resnet34, l1_pruning_with_prs_for_mobilenet_v2

def objective_function_pruning(x, additional_parameter):
    model, flops_ori, paras_ori, doc, device, target_df, target_dp = additional_parameter['model'], additional_parameter['flops_ori'], \
        additional_parameter['paras_ori'], additional_parameter['doc'], additional_parameter['device'], additional_parameter['target_drf'], additional_parameter['target_drp']
    prs = {}
    idx = 0
    for k, v in  doc.items():
        prs[k] = x[idx]
        idx += 1
    if 'MobileNetV2' in str(type(model)):
        pm = l1_pruning_with_prs_for_mobilenet_v2(model, prs)
        HW = 224
    elif 'ResNet' in str(type(model)):
        pm = l1_pruning_with_prs_for_resnet34(model, prs)
        HW = 224
    elif 'FCN' in str(type(model)):
        pm = l1_pruning_with_prs_for_fcn_resnet101(model, prs, True)
        HW = 480
    flops, paras = get_flops_paras(pm, device, HW)
    df = (flops_ori - flops) / flops_ori
    dp = (paras_ori - paras) / paras_ori
    rate_doc = np.array(list(doc.values()))
    rate_doc = rate_doc / rate_doc.sum()
    rate_x = x / x.sum()
    obj = np.absolute(rate_doc - rate_x).sum()
    if target_df == 0:
        vio = max(0, abs(target_dp -dp)-0.0004)
    elif target_dp ==0:
        vio = max(0, abs(target_df - df)-0.0004)
    else:
        vio = max(0, abs(target_df - df)-0.0004) + max(0, abs(target_dp -dp)-0.0004)
    return obj, vio

class DE():
    def __init__(self, obj_func, additional_parameter, bounds, pop_size, max_iter, F, CR, print_interval, logger):
        self.obj_func = obj_func
        self.additional_parameter = additional_parameter
        self.bounds = bounds
        self.pop_size = pop_size
        self.max_iter = max_iter
        self.F = F
        self.CR = CR
        self.print_interval = print_interval
        self.logger = logger
        self.best_solutions = []
        self.best_differents = []
    
    def init_population(self):
        self.dim = len(self.bounds)
        self.low = np.array([b[0] for b in self.bounds])
        self.high = np.array([b[1] for b in self.bounds])
        self.population = np.random.uniform(self.low, self.high, (self.pop_size, self.dim))
        self.obj_values = np.zeros(self.pop_size)
        self.violations = np.zeros(self.pop_size)
        for i in range(self.pop_size):
            self.obj_values[i], self.violations[i] = self.obj_func(self.population[i],self.additional_parameter)
        fitness = np.where(self.violations == 0, self.obj_values, self.violations + 1e20)
        best_idx = np.argmin(fitness)
        self.best_obj = self.obj_values[best_idx]
        self.best_sol = self.population[best_idx]
        self.best_violation = self.violations[best_idx]
        
    def iter_search(self):
        self.init_population()
        for iter in range(self.max_iter):
            for i in range(self.pop_size):
                candidates = [idx for idx in range(self.pop_size) if idx != i]
                a, b, c = self.population[np.random.choice(candidates, 3, replace=False)]
                mutant = a + self.F * (b - c)
                cross_points = np.random.rand(self.dim) < self.CR
                if not np.any(cross_points):
                    cross_points[np.random.randint(self.dim)] = True
                trial = np.where(cross_points, mutant, self.population[i])
                trial = np.clip(trial, self.low, self.high)
                trial_obj, trial_viol = self.obj_func(trial,self.additional_parameter)
                if (trial_viol < self.violations[i]) or \
                (trial_viol == self.violations[i] and trial_viol == 0 and trial_obj < self.obj_values[i]) or \
                (trial_viol == self.violations[i] and trial_viol > 0 and trial_obj < self.obj_values[i]):
                    self.population[i] = trial
                    self.obj_values[i] = trial_obj
                    self.violations[i] = trial_viol
                    if trial_viol < self.best_violation or \
                    (trial_viol == self.best_violation and trial_obj < self.best_obj):
                        self.best_obj = trial_obj
                        self.best_sol = trial
                        self.best_violation = trial_viol
                        if self.best_violation == 0:
                            self.best_solutions.append(self.best_sol)
                            self.best_differents.append(self.best_obj)
            if self.print_interval != 0 and iter % self.print_interval == 0:
                self.logger.info('    iter = {}  best_obj = {:.6f}  best_violation = {:.6f}'.format(iter,self.best_obj, self.best_violation))
