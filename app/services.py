from voice.utils.audio import AudioConverter
from voice.noise_reduction import SpeechBrainNoiseReducer, BaseNoiseReducer
from voice.speech_to_text import SpeechBrain, SpeechRecognition, Wav2Vec2, Whisper
from voice.modules.devices import torch_gc
from schema import VoiceRequest, VoiceRequestSchema


class SpeechToTextService:

    def __init__(
        self,
        transcriber,
        noise_reducer: str | None = None,
        base64String: str | None = None,
        path: str | None = None
    ):
        self.transcriber = transcriber
        self.noise_reducer = noise_reducer
        audio = AudioConverter(base64String, path)
        self.audio, self.sr = audio.data, audio.sr

    def transcribe(
        self,
    ) -> str:
        if self.noise_reducer:
            self.audio = self.noise_reducer.reduce_noise(
                self.audio, self.sr
            )
        return self.transcriber.transcribe(self.audio, self.sr)


class SpeechToTextServiceCreator:

    @staticmethod
    def _get_noise_reducer(noiseReducer: int):
        if noiseReducer == 1:
            return BaseNoiseReducer()
        elif noiseReducer == 2:
            return SpeechBrainNoiseReducer()
        else:
            raise Exception("Invalid noise reducer.")

    @staticmethod
    def _get_transcriber(speechToText: int):
        if speechToText == 1:
            return SpeechRecognition()
        elif speechToText == 2:
            return Whisper()
        elif speechToText == 3:
            return Wav2Vec2()
        elif speechToText == 4:
            return SpeechBrain()
        else:
            raise Exception("Invalid speech to text.")

    @staticmethod
    def speech_to_text(
        data: VoiceRequest,
        base64String: str | None = None,
        path: str | None = None
    ):
        """
        Create a SpeechToTextService instance given a VoiceRequest object.

        This factory function takes a VoiceRequest object and creates a SpeechToTextService
        instance with the appropriate transcriber and noise reducer based on the values
        of the VoiceRequest object.

        Args:
            data (VoiceRequest): The VoiceRequest object containing the configuration
                for the SpeechToTextService instance.
            base64String (str | None): The base64 encoded string for the audio data.
            path (str | None): The path to the audio file.

        Returns:
            SpeechToTextService: The created SpeechToTextService instance.
        """
        noise_reducer = None
        if data.enableNoiseReduction:
            noise_reducer = SpeechToTextServiceCreator._get_noise_reducer(
                data.noiseReducer)

        transcriber = SpeechToTextServiceCreator._get_transcriber(
            data.speechToText)

        return SpeechToTextService(
            transcriber=transcriber,
            noise_reducer=noise_reducer,
            base64String=base64String,
            path=path
        )
