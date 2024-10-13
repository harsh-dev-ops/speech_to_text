from abc import ABC, abstractmethod
import torch

from conf.methods import Singleton


class SpeechToText(ABC):

    @abstractmethod
    def _load_model(self):
        pass

    def _load_processor(self):
        pass

    def _get_device(self):
        return "cuda:0" if torch.cuda.is_available() else "cpu"

    def _get_dtype(self):
        return torch.float16 if torch.cuda.is_available() else torch.float32
