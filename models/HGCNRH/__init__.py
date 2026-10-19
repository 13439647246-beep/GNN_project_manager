# 导出 build_model / build_loss，并注册到 registry
from models.HGCNRH import model   # noqa: F401  触发 @register_model("gcn")
from models.HGCNRH import loss    # noqa: F401  触发 @register_loss("gcn_loss")

