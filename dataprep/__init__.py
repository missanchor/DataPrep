# dataprep/__init__.py

# 从 correction 子包中导入主要的类
from .correction.zeroec import ZeroEC
from .correction.base import BaseDataCorrector

# 如果您有其他模块（例如 detection, imputation），也可以在这里导入其主要类
# from .detection import BaseErrorDetector, ZeroED
# from .imputation import MISS

__version__ = "0.1.0" # 定义包的版本号

# 定义 __all__ 列表来明确哪些内容在 from dataprep import * 时会被导入
__all__ = [
    "ZeroEC",
    "BaseDataCorrector",
    "__version__"
]