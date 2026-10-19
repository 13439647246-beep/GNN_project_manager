# ============================================================
# src/utils/registry.py
# 全局注册表：模型 / 损失 / 数据集，按名字注册与构建。
#
# 目的：
#   - engine / run_test 只通过"名字"构建对象，绝不 import 具体实现
#   - 新增模型不改公共代码，只在自己的包里注册
#
# 约定：
#   每个注册项是一个构建函数，签名为 fn(arg) -> 对象
#   arg 是全局的 arg dict（见 src/config/default_arg.py）
# ============================================================

# 三张独立的注册表：名字字符串 -> 构建函数
MODEL_REGISTRY = {}
LOSS_REGISTRY  = {}
DATA_REGISTRY  = {}


# ------------------------------------------------------------
# 装饰器工厂：避免为三个注册表写三份重复代码
# ------------------------------------------------------------
def _make_register(registry, kind):
    def register(name):
        def deco(fn):
            if name in registry:
                # 提前暴露重名，避免后面静默覆盖
                raise ValueError(f"{kind} '{name}' 已被注册，名字冲突")
            registry[name] = fn
            return fn
        return deco
    return register


# 三个公开装饰器
register_model = _make_register(MODEL_REGISTRY, "model")
register_loss  = _make_register(LOSS_REGISTRY,  "loss")
register_data  = _make_register(DATA_REGISTRY,  "data")


# ------------------------------------------------------------
# 构建函数：由 arg 里的 "name" 字段查表并调用构建函数
# ------------------------------------------------------------
def _build(registry, kind, name, arg):
    if name not in registry:
        avail = ", ".join(sorted(registry.keys())) or "(空)"
        raise KeyError(f"未找到 {kind} '{name}'，已注册的有：{avail}")
    return registry[name](arg)


def build_model(arg):
    """从 arg['model']['name'] 构建模型实例"""
    return _build(MODEL_REGISTRY, "model", arg["model"]["name"], arg)


def build_loss(arg):
    """从 arg['loss']['name'] 构建损失实例"""
    return _build(LOSS_REGISTRY, "loss", arg["loss"]["name"], arg)


def build_data(arg):
    """从 arg['data']['name'] 构建数据集"""
    return _build(DATA_REGISTRY, "data", arg["data"]["name"], arg)


# ------------------------------------------------------------
# 查询接口：便于打印帮助 / 调试 / 断言
# ------------------------------------------------------------
def list_models():   return sorted(MODEL_REGISTRY.keys())
def list_losses():   return sorted(LOSS_REGISTRY.keys())
def list_datasets(): return sorted(DATA_REGISTRY.keys())