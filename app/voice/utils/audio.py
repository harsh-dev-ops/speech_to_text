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
        
    def base64_string_to_array(
        self, 
        base64_string: str
        ) -> tuple[np.ndarray, int]:
        
        audio_bytes = base64.b64decode(base64_string)
        audio_stream = io.BytesIO(audio_bytes)
        audio_data, sample_rate = librosa.load(audio_stream, sr=None)
        return np.array(audio_data), sample_rate

class AudioConverter(FileToBase64, AudioToArray):
    
    def convert(
        self, 
        audio_file: str | None = None, 
        base64_string: str | None = None
        ) -> tuple[np.ndarray, int]:
        
        if not base64_string and audio_file:
            base64_string = self.base64_string(audio_file)
            
        return self.base64_string_to_array(base64_string)


"""
audio_converter = AudioConverter()

print(audio_converter.convert(audio_file='samples/sample-0.mp3'))
"""
