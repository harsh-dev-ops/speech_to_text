from speechbrain.inference import EncoderDecoderASR
import numpy as np
import torch

from base import SpeechToText


class CustomEncoderDecoderASR(EncoderDecoderASR):
    def load_audio(
        self,
        ndarray: np.ndarray,
        sr: int,
        path: str | None = None,
        savedir="."
    ):
        if not path:
            signal = torch.from_numpy(ndarray).float()
            return self.audio_normalizer(signal, sr)
        return super().load_audio(path, savedir)

    def transcribe_ndarray(
            self,
            ndarray: np.ndarray,
            sr: int,
            **kwargs
    ):
        waveform = self.load_audio(
            ndarray, sr, **kwargs)

        batch = waveform.unsqueeze(0)
        rel_length = torch.tensor([1.0])
        predicted_words, predicted_tokens = self.transcribe_batch(
            batch, rel_length
        )
        return predicted_words[0]


class SpeechBrain(SpeechToText):
    def __init__(
        self,
        model_id: str = "speechbrain/asr-conformer-transformerlm-librispeech",
        save_dir: str = "pretrained_models/asr-transformer-transformerlm-librispeech"
    ):
        self.model_id = model_id
        self.save_dir = save_dir
        self.model = self._load_model()

    def _load_model(self):
        return CustomEncoderDecoderASR.from_hparams(source=self.model_id,
                                                    savedir=self.save_dir)

    def transcribe(
        self,
        ndarray: np.ndarray,
        sr: int
    ):
        return self.model.transcribe_ndarray(ndarray, sr).lower()
