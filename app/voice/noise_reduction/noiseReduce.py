import numpy
from scipy.io import wavfile
import noisereduce as nr

from .base import NoiseReducer


class BaseNoiseReducer(NoiseReducer):

    def reduce_noise(
        self,
        ndarray: numpy.ndarray,
        sr: int
    ) -> numpy.ndarray:

        return nr.reduce_noise(y=ndarray, sr=sr)
