import numpy
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor
from datasets import Audio, load_dataset

from base import SpeechToText


class Whisper(SpeechToText):
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
        inputs = self.processor(ndarray,
                                sr=sr, return_tensors="pt",
                                truncation=False, padding="longest",
                                return_attention_mask=True)

        inputs = inputs.to(self.device, dtype=self.dtype)

        gen_kwargs = {
            "max_new_tokens": 448,
            "num_beams": 1,
            "condition_on_prev_tokens": False,
            # zlib compression ratio threshold (in token space)
            "compression_ratio_threshold": 1.35,
            "temperature": (0.0, 0.2, 0.4, 0.6, 0.8, 1.0),
            "logprob_threshold": -1.0,
            "no_speech_threshold": 0.6,
            "return_timestamps": True,
        }

        pred_ids = self.model.generate(**inputs, **gen_kwargs)
        pred_text = self.processor.batch_decode(
            pred_ids, skip_special_tokens=True, decode_with_timestamps=False)

        return pred_text
