from transformers import SpeechT5Processor, SpeechT5ForSpeechToText
from datasets import load_dataset
import torch
import numpy

from .base import SpeechToText, HfSpeechToText
from conf.methods import Singleton


class SpeechT5(SpeechToText, HfSpeechToText):
    def __init__(
        self,
        model_id: str = "microsoft/speecht5_asr",
    ):
        self.model_id = model_id
        self.model = self._load_model()
        self.processor = self._load_processor()

    def _load_model(self):
        return SpeechT5ForSpeechToText.from_pretrained(self.model_id)

    def _load_processor(self):
        return SpeechT5Processor.from_pretrained(self.model_id)

    def transcribe(
        self,
        ndarray: numpy.ndarray,
        sr: int
    ):
        inputs = self.processor(
            audio=ndarray, sampling_rate=sr, return_tensors="pt"
        )

        predicted_ids = self.model.generate(**inputs, max_length=100)

        transcription = self.processor.batch_decode(
            predicted_ids, skip_special_tokens=True)

        return transcription[0]
