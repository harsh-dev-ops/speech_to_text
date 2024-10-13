from speechbrain.inference.separation import SepformerSeparation
import numpy as np
import torch

from conf.methods import Singleton


class SpeechBrainNoiseReducer():
    def __init__(
            self,
            model_id: str = "speechbrain/sepformer-wham16k-enhancement",
            savedir='pretrained_models/sepformer-wham16k-enhancement'
    ):
        self.savedir = savedir
        self.model_id = model_id
        self.model = self._load_model()

    def _load_model(self):
        return SepformerSeparation.from_hparams(
            source=self.model_id, savedir=self.savedir)

    def reduice_noise(
            self,
            ndarray: np.ndarray,
            sr: int
    ):
        tensors = torch.from_numpy(ndarray)
        return self.model.separate_batch(tensors)
