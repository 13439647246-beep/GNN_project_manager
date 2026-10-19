# 设备选择：自动判断 cuda/mps/cpu
# ============================================================
# src/utils/device.py
# 设备选择 + 递归搬移 batch 到目标设备
# ============================================================
import torch


def get_device(arg):
    """根据 arg['train']['device'] 决定实际设备，'auto' 时自动探测"""
    d = arg["train"]["device"]
    if d == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(d)


def move_to_device(obj, device):
    """
    递归把 batch 里的张量搬到 device。
    支持 dict / list / tuple / 拥有 .to() 方法的对象（如 PyG 的 Data）。
    """
    if torch.is_tensor(obj):
        return obj.to(device)
    if isinstance(obj, dict):
        return {k: move_to_device(v, device) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return type(obj)(move_to_device(v, device) for v in obj)
    if hasattr(obj, "to") and callable(obj.to):
        return obj.to(device)
    return obj