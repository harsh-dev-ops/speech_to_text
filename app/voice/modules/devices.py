import sys
import contextlib
from functools import lru_cache
from packaging import version

import torch

if sys.platform == "darwin":
    from . import mac_specific


def check_for_mps() -> bool:
    if version.parse(torch.__version__) <= version.parse("2.0.1"):
        if not getattr(torch, 'has_mps', False):
            return False
        try:
            torch.zeros(1).to(torch.device("mps"))
            return True
        except Exception:
            return False
    else:
        return torch.backends.mps.is_available() and torch.backends.mps.is_built()


def has_mps() -> bool:
    if sys.platform != "darwin":
        return False
    else:
        return check_for_mps()


def get_optimal_device_name():
    if torch.cuda.is_available():
        return "cuda"

    if has_mps():
        return "mps"

    return "cpu"


def get_optimal_device():
    return torch.device(get_optimal_device_name())


def torch_gc():

    if torch.cuda.is_available():
        with torch.cuda.device(get_optimal_device_name()):
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

    if has_mps():
        mac_specific.torch_mps_gc()
