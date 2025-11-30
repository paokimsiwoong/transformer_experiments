import torch
import numpy as np
import random

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)

def set_seed_new(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True, warn_only=False)
    np.random.seed(seed)
    random.seed(seed)

# 파라메터에 NaN이 있는지 확인하는 함수
def check_nan_in_parameters(model):
    for name, param in model.named_parameters():
        if torch.isnan(param).any():
            print(f"NaN found in parameter: {name}")
            return True
        
        if torch.isinf(param).any():
            print(f"inf found in parameter: {name}")
            return True
    return False