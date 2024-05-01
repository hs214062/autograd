from ..core import *

class Optimizer:
    def update(self):
        raise NotImplementedError()

class SGD(Optimizer):
    params: dict[str, Tensor]
    lr: float

    def __init__(self, params: dict[str, Tensor], lr: float=0.06):
        self.params = params
        self.lr = lr

    def update(self):
        for param in self.params.values():
            param.value -= param.grad * self.lr

class Adam(Optimizer):
    params: dict[str, Tensor]
    m: dict[str, NDArray]
    v: dict[str, NDArray]
    lr: float
    beta1: float
    beta2: float
    iter: int
    eps: float

    def __init__(self, params: dict[str, Tensor], lr=0.001, beta1=0.9, beta2=0.999, eps=1e-09):
        self.params = params
        self.m = {key: np.zeros(param.shape, dtype=param.dtype) for key, param in params.items()}
        self.v = {key: np.zeros(param.shape, dtype=param.dtype) for key, param in params.items()}
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.iter = 0
        self.eps = eps

    def update(self):
        self.iter += 1
        lr_t  = self.lr * np.sqrt(1.0 - self.beta2**self.iter) / (1.0 - self.beta1**self.iter)

        for key in self.params.keys():
            p, m, v = self.params[key], self.m[key], self.v[key]
            m += (1 - self.beta1) * (p.grad - m)
            v += (1 - self.beta2) * (p.grad**2 - v)

            p.value -= (lr_t * m) / (np.sqrt(v) + self.eps)