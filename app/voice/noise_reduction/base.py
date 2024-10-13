from abc import ABC, abstractmethod
import numpy


class NoiseReducer(ABC):
    @abstractmethod
    def reduce_noise(
        self,
        ndarray: numpy.ndarray,
        sr: int
    ) -> numpy.ndarray:

        pass
