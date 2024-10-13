from speechbrain.inference import EncoderDecoderASR
import numpy as np
import torch


class CustomEncoderDecoderASR(EncoderDecoderASR):
    def load_audio(
        self,
        audio_array: np.ndarray,
        sr: int,
        path: str | None = None,
        savedir="."
    ):
        if not path:
            signal = torch.from_numpy(audio_array).float()
            return self.audio_normalizer(signal, sr)
        return super().load_audio(path, savedir)

    def transcribe_ndarray(
            self,
            audio_array: np.ndarray,
            sr: int,
            **kwargs
    ):
        waveform = self.load_audio(
            audio_array, sr, **kwargs)

        batch = waveform.unsqueeze(0)
        rel_length = torch.tensor([1.0])
        predicted_words, predicted_tokens = self.transcribe_batch(
            batch, rel_length
        )
        return predicted_words[0]


asr_model = CustomEncoderDecoderASR.from_hparams(source="speechbrain/asr-conformer-transformerlm-librispeech",
                                                 savedir="pretrained_models/asr-transformer-transformerlm-librispeech")
