import numpy
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
from datasets import load_dataset
import torch

from .base import HfSpeechToText, SpeechToText
from conf.methods import Singleton


class Wav2Vec2(HfSpeechToText, SpeechToText):
    def __init__(
        self,
        model_id: str = "facebook/wav2vec2-large-960h"
    ):
        self.model_id = model_id
        self.model = self._load_model()
        self.processor = self._load_processor()

    def _load_model(self):
        return Wav2Vec2ForCTC.from_pretrained(self.model_id)

    def _load_processor(self):
        return Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-large-960h")

    def transcribe(
        self,
        ndarray: numpy.ndarray,
        sr: int
    ) -> str:

        input_values = self.processor(
            ndarray, sr=sr,
            return_tensors="pt", padding="longest").input_values

        logits = self.model(input_values).logits

        predicted_ids = torch.argmax(logits, dim=-1)
        transcription = self.processor.batch_decode(predicted_ids)

        return transcription[0]
