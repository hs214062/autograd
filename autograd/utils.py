"""딥러닝을 위한 다양한 편의 함수를 정의한 모듈"""
from .core import *
import pickle

def rand(*shape: int, dtype=None, requires_grad=False):
    """가중치 초기화를 위한 numpy.random.rand. 기본 dtype은 Config를 따른다."""
    if dtype is None: dtype = Config.dtype
    return Tensor(np.random.rand(*shape).astype(dtype), requires_grad=requires_grad)

def randn(*shape: int, dtype=None, requires_grad=False):
    """가중치 초기화를 위한 numpy.random.randn. 기본 dtype은 Config를 따른다."""
    if dtype is None: dtype = Config.dtype
    return Tensor(np.random.randn(*shape).astype(dtype), requires_grad=requires_grad)

def ones(shape: int, dtype=None, requires_grad=False):
    """가중치 초기화를 위한 numpy.ones. 기본 dtype은 Config를 따른다."""
    if dtype is None: dtype = Config.dtype
    return Tensor(np.ones(shape, dtype=dtype), requires_grad=requires_grad)

def zeros(shape: int, dtype=None, requires_grad=False):
    """가중치 초기화를 위한 numpy.zeros. 기본 dtype은 Config를 따른다."""
    if dtype is None: dtype = Config.dtype
    return Tensor(np.zeros(shape, dtype=dtype), requires_grad=requires_grad)

def linspace(start: float, stop: float, num: int, dtype=None, requires_grad=False) -> Tensor:
    """numpy.linspace와 같다. 기본 dtype은 Config를 따른다."""
    if dtype is None: dtype = Config.dtype
    return Tensor(np.linspace(start, stop, num, dtype=dtype), requires_grad=requires_grad)

def save_to_pickle(*objects, path: str):
    """입력된 객체를 pickle로 저장한다."""
    if len(objects) == 1:
        objects = objects[0]
    
    with open(path, "wb") as f:
        pickle.dump(objects, f)

def load_from_pickle(path: str):
    """pickle에서 객체를 불러온다."""
    with open(path, "rb") as f:
        objects = pickle.load(f)

    return objects