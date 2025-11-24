import torch


# 파라메터에 hook로 등록하면 grad에 NaN이 들어올 때 raise하는 함수
def nan_hook(grad):
    if torch.isnan(grad).any():
        print("NaN detected in gradient!")
        # raise ValueError("NaN in gradient!")

# 파라메터에 그래디언트 클리핑 적용하는 hook
def clip_grad_embed_hook(grad):
    max_norm = 10.0  # 원하는 클리핑 임계값

    # if torch.isnan(grad).any():
    #     print("NaN detected in gradient!")
    #     raise ValueError("NaN in gradient!")
    
    norm = grad.norm()
    if norm > max_norm:
        grad = grad * (max_norm / norm)
    return grad

def clip_grad_hook(grad):
    max_norm = 1.0  # 원하는 클리핑 임계값

    # if torch.isnan(grad).any():
    #     print("NaN detected in gradient!")
    #     raise ValueError("NaN in gradient!")
    
    norm = grad.norm()
    if norm > max_norm:
        grad = grad * (max_norm / norm)
    return grad