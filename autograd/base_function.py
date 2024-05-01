"""
이 파일에서는 Tensor를 NDArray와 똑같이 사용할 수 있도록 하는 기본 연산자를 구현한다.
"""

from .core import *
from numpy import argsort

class Transpose(Function):
    def __init__(self, axes=None):
        self.axes = axes
    
    def forward(self, x):
        return np.transpose(x, axes=self.axes)

    def backward(self, dy):
        axes = self.axes
        if axes is not None:
            axes = argsort(axes)

        return lambda: dy.transpose(axes),

def transpose(x, axes=None):
    return Transpose(axes).apply(x)

Tensor.transpose = transpose

class Swapaxes(Function):
    def __init__(self, axis1: int, axis2: int):
        self.axis1 = axis1
        self.axis2 = axis2
    
    def forward(self, x):
        return x.swapaxes(self.axis1, self.axis2)
    
    def backward(self, dy):
        return lambda: dy.swapaxes(self.axis1, self.axis2),

def swapaxes(x, axis1, axis2):
    return Swapaxes(axis1, axis2).apply(x)

Tensor.swapaxes = lambda x, axis1, axis2: Swapaxes(axis1, axis2).apply(x)

class GetItem(Function):
    def __init__(self, index):
        self.index = index

    def forward(self, x):
        return x[self.index]

    def backward(self, dy):
        x, = self.inputs

        def dx():
            out = np.zeros(x.shape, dtype=x.dtype) # zeros_like(x.value)와 같다.
            np.add.at(out, self.index, dy.value if isinstance(dy, Tensor) else dy)
            return out

        return dx,

Tensor.__getitem__ = lambda x, index: GetItem(index).apply(x)

class Add(Function):
    def forward(self, a, b):
        return a + b

    def backward(self, dy):
        a, b = self.inputs
        return lambda: unbroadcast(a, dy), lambda: unbroadcast(b, dy)

def add(a, b):
    return Add().apply(a, b)

Tensor.__add__ = lambda a, b: Add().apply(a, b)
Tensor.__radd__ = lambda a, b: Add().apply(b, a)

class Sub(Function):
    def forward(self, a, b):
        return a - b

    def backward(self, dy):
        a, b = self.inputs
        return lambda: unbroadcast(a, dy), lambda: unbroadcast(b, -dy)

def subtract(a, b):
    return Sub().apply(a, b)

Tensor.__sub__ = lambda a, b: Sub().apply(a, b)
Tensor.__rsub__ = lambda a, b: Sub().apply(b, a)

class Mul(Function):
    def forward(self, a, b):
        return a * b

    def backward(self, dy):
        a, b = self.inputs
        return lambda: unbroadcast(a, dy * b), lambda: unbroadcast(b, dy * a)

def multiply(a, b):
    return Mul().apply(a, b)

Tensor.__mul__ = lambda a, b: Mul().apply(a, b)
Tensor.__rmul__ = lambda a, b: Mul().apply(b, a)

class TrueDiv(Function):
    def forward(self, a, b):
        return a / b
        
    def backward(self, dy):
        a, b = self.inputs
        return lambda: unbroadcast(a, dy / b), lambda: unbroadcast(b, -dy * a / b**2)

def divide(a, b):
    return TrueDiv().apply(a, b)

Tensor.__truediv__ = lambda a, b: TrueDiv().apply(a, b)
Tensor.__rtruediv__ = lambda a, b: TrueDiv().apply(b, a)

class Neg(Function):
    def forward(self, x):
        return -x
    
    def backward(self, dy):
        return lambda: -dy,

def negative(x):
    return Neg().apply(x)

Tensor.__neg__ = lambda x: Neg().apply(x)

class Pow(Function):
    def forward(self, a, b):
        return a ** b
    
    def backward(self, dy):
        a, b = self.inputs
        return lambda: unbroadcast(a, dy * b * a ** (b-1)), lambda: unbroadcast(b, dy * (a ** b) * a.log())

def power(a, b):
    return Pow().apply(a, b)

Tensor.__pow__ = lambda a, b: Pow().apply(a, b)
Tensor.__rpow__ = lambda a, b: Pow().apply(b, a)