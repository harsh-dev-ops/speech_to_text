import numpy
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor
from datasets import Audio, load_dataset

from .base import HfSpeechToText, SpeechToText
from conf.methods import Singleton


class Whisper(HfSpeechToText, SpeechToText):
    def __init__(
        self,
        model_id: str = "openai/whisper-large-v3"
    ):
        self.model_id = model_id
        self.model = self._load_model()
        self.device = self._get_device()
        self.dtype = self._get_dtype()
        self.processor = self._load_processor()

    def _load_model(self):

        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            self.model_id, torch_dtype=self._get_dtype(),
            low_cpu_mem_usage=True)
        model.to(self._get_device())
        return model

    def _load_processor(self):
        return AutoProcessor.from_pretrained(self.model_id)

    def transcribe(
        self,
        ndarray: numpy.ndarray,
        sr: int
    ) -> str:

        input_features = self.processor(
            ndarray, sampling_rate=sr, return_tensors="pt"
        ).input_features

        predicted_ids = self.model.generate(input_features)

        transcription = self.processor.batch_decode(
            predicted_ids, skip_special_tokens=True)

        pred_text = transcription[0]

        return pred_text
