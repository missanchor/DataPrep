# dataprep/base.py
from abc import ABC, abstractmethod
import joblib  # 或者使⽤ pickle/cloudpickle


class BaseEstimator(ABC):


    @abstractmethod
    def train(self, *args, **kwargs):
        """
        训练模型。具体参数由各任务的抽象基类定义。
        对于⽆需训练的模型，可以为空实现，直接 return self。
        """
        raise NotImplementedError


    @abstractmethod
    def predict(self, *args, **kwargs):
        """
        使⽤已训练的模型进⾏核⼼转换操作。具体参数由各任务的抽象基类定义。
        """
        raise NotImplementedError

    def train_and_predict(self, *args, **kwargs):
        """先训练，后转换。"""
        # 注意：这⾥的参数传递需要更精细的设计，或者假设train和predict的输⼊参数有重叠
        # ⼀个简单的实现如下，但具体实现可能需要调整
        self.train(*args, **kwargs)
        return self.predict(*args, **kwargs)


    def save_model(self, path: str):
        """将训练好的模型实例保存到⽂件。"""
        # 使⽤ joblib 可以更好地处理包含numpy数组的对象
        joblib.dump(self, path)
        print(f"Model saved to {path}")


    @staticmethod
    def load_model(path: str):
        """从⽂件加载模型实例。"""
        model = joblib.load(path)
        print(f"Model loaded from {path}")
        return model
