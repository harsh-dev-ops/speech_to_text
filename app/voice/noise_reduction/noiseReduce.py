import numpy
from scipy.io import wavfile
import noisereduce as nr


def basic_noise_reduce(
    ndarray: numpy.ndarray,
    sr: int
) -> numpy.ndarray:

    return nr.reduce_noise(y=ndarray, sr=sr)
