import base64
from click import File
import numpy as np
import librosa
import io


class FileToBase64:
    def __init__(self):
        pass

    def base64_string(
        self,
        audio_file: str
    ) -> str:

        with open(audio_file, 'rb') as f:
            audio_data = f.read()
            base64_audio = base64.b64encode(audio_data)
        audio_data = base64_audio.decode('utf-8')
        return audio_data


class AudioToArray:
    def __init__(self):
        pass

    def to_audio_stream(self, base64_string: str):
        audio_bytes = base64.b64decode(base64_string)
        return io.BytesIO(audio_bytes)

    def base64_string_to_ndarray(
        self,
        base64_string: str
    ) -> tuple[np.ndarray, int]:

        audio_stream = self.to_audio_stream(base64_string)
        audio_data, sample_rate = librosa.load(audio_stream)
        return np.array(audio_data), sample_rate

    def audio_file_to_ndarray(
        self,
        path: str
    ) -> tuple[np.ndarray, int]:

        audio_data, sample_rate = librosa.load(path)
        return np.array(audio_data), sample_rate


class AudioConverter(FileToBase64, AudioToArray):
    def __init__(
        self,
        base64String: str | None = None,
        path: str | None = None
    ):
        self.base64String = base64String
        self.path = path
        self.data, self.sr = self._load_audio()

    def _load_audio(
        self
    ):
        if self.base64String:
            return self.base64_string_to_ndarray(
                self.base64String)
        elif self.path:
            return self.audio_file_to_ndarray(self.path)


"""
audio_converter = AudioConverter()

print(audio_converter.convert(audio_file='samples/sample-0.mp3'))
"""
