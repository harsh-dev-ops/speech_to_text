import io
import scipy
import speech_recognition
import numpy as np


class SpeechRecognition:

    def __init__(
        self,
        model_id: str = "google",
    ):
        self.model_id = model_id
        self.recognizer = speech_recognition.Recognizer()

    def _recognize(
        self,
        audio_data: speech_recognition.AudioData
    ):
        if self.model_id == "google":
            return self.recognizer.recognize_google(audio_data)
        elif self.model_id == "whisper":
            return self.recognizer.recognize_whisper(audio_data)

    def transcribe(
        self,
        ndarray: np.ndarray,
        sr: int
    ) -> str:

        byte_io = io.BytesIO(bytes())
        scipy.io.wavfile.write(byte_io, sr, ndarray)
        audio_bytes = byte_io.read()
        audio_data = speech_recognition.AudioData(audio_bytes, sr, 2)

        text = None
        try:
            text = self._recognize(audio_data)
        except speech_recognition.UnknownValueError:
            print("Speech recognition could not understand the audio.")
        except speech_recognition.RequestError as e:
            print(f"Could not request results from service; {e}")

        return text.lower() if text else ""
