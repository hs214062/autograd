from ..core import Tensor, Function, unbroadcast, Config
from ..utils import randn, ones, zeros
from typing import Self, Literal
from ..numpy import *
from ..numpy_function import mean

class Layer:
    _params: dict[str, Tensor | Self]

    def __init__(self):
        super().__setattr__("_params", {})

    def _flatten_params(self, parent_key="", params={}):
        """레이어에 저장된 파라미터를 재귀적으로 읽어 딕셔너리로 만들어 준다.
        키는 내부적으로 저장된 이름이다.
        
        인수
        --
        parent_key: str = ""
            이 레이어를 포함하는 부모 레이어의 이름이다.
            굳이 건드릴 필요는 없다.
        
        params: dict[str, Tensor] = {}
            파라미터가 추가될 딕셔너리이다.
        """
        for key, param in self._params.items():
            
            full_key = parent_key+"_"+key if parent_key != "" else key
            if isinstance(param, Layer):
                param._flatten_params(full_key, params)
                continue
            
            params[full_key] = param

    @property
    def params(self) -> dict[Tensor]:
        """레이어에 저장된 파라미터를 재귀적으로 읽어 딕셔너리로 만들어 준다.
        키는 내부적으로 저장된 이름이다."""
        params = {}
        self._flatten_params(params=params)

        return params

    def __setattr__(self, name, value):
        if isinstance(value, Tensor) or isinstance(value, Layer):
            self._params[name] = value

        super().__setattr__(name, value)

    def save_params(self, path: str):
        """파라미터를 npz파일로 저장한다."""
        params = {key: param.value for key, param in self.params.items()}

        np.savez(path, **params)

    def load_params(self, path: str):
        """파라미터를 npz파일에서 불러온다."""
        params = self.params

        with np.load(path) as data:
            for key in data.npz_file.files:
                params[key].value = data[key]

    def __call__(self, *args, **kwargs):
        raise NotImplementedError()

class DenseFunc(Function):
    def forward(self, x, w, b):
        return x @ w + b
    
    def backward(self, dy):
        x, w, b = self.inputs

        dx = lambda: unbroadcast(x, dy @ w.swapaxes(-1, -2))
        dw = lambda: unbroadcast(w, x.swapaxes(-1, -2) @ dy)
        db = lambda: unbroadcast(b, dy)

        return dx, dw, db

class Dense(Layer):
    """완전 연결 레이어. Linear라고도 불린다."""
    w: Tensor
    b: Tensor | None

    def __init__(self, w_shape: tuple[int], b_shape: tuple[int]=None, w_init: Literal["xavier", "he"] | None=None):
        super().__init__()
        self.w = randn(*w_shape, requires_grad=True)
        self.b = None

        if w_init == "xavier":
            self.w.value *= np.sqrt(Config.dtype(1 / w_shape[-2]))

        elif w_init == "he":
            self.w.value *= np.sqrt(Config.dtype(2 / w_shape[-2]))

        if b_shape is not None:
            self.b = zeros(*b_shape, requires_grad=True)

    def __call__(self, x):
        if self.b is None:
            return x @ self.w

        return DenseFunc().apply(x, self.w, self.b)
    
class Normalize(Function):
    def __init__(self, axis=-1, epsilon=1e-03):
        self.axis = axis
        self.epsilon = epsilon

    def forward(self, x):
        axis = self.axis
        self.mx = x - np.mean(x, axis=axis, keepdims=True)
        self.istd = 1 / np.sqrt(np.mean(np.square(self.mx), axis=axis, keepdims=True) + self.epsilon)

        return self.mx * self.istd

    def backward(self, dy):
        axis, mx, istd = self.axis, self.mx, self.istd
        size = dy.shape[axis]

        def dx():
            dmx = istd * (dy - (dy * mx).sum(axis=axis, keepdims=True) * (istd ** 2) * mx / size)
            return dmx - mean(dmx, axis=axis, keepdims=True)

        return dx,

def normalize(x, axis=-1, epsilon=1e-03):
    return Normalize(axis, epsilon).apply(x)

class LayerNorm(Layer):
    def __init__(self, *shape, axis=-1, epsilon=1e-03):
        super().__init__()
        self.w = ones(shape, requires_grad=True)
        self.b = zeros(shape, requires_grad=True)
        self.axis = axis
        self.epsilon = epsilon

    def __call__(self, x):
        return normalize(x, self.axis, self.epsilon) * self.w + self.b