from .core import *

class Matmul(Function):
    def forward(self, a, b):
        return a @ b
    
    def backward(self, dy):
        a, b = self.inputs
        return lambda: unbroadcast(a, dy @ b.swapaxes(-1, -2)), lambda: unbroadcast(b, a.swapaxes(-1, -2) @ dy)

def matmul(a, b):
    return Matmul().apply(a, b)

Tensor.__matmul__ = lambda a, b: Matmul().apply(a, b)
Tensor.__rmatmul__ = lambda a, b: Matmul().apply(b, a)

class Sqrt(Function):
    def forward(self, x):
        return np.sqrt(x)
    
    def backward(self, dy):
        return lambda: dy / (2 * self.y()),

def sqrt(x):
    return Sqrt().apply(x)

class Square(Function):
    def forward(self, x):
        return np.square(x)
    
    def backward(self, dy):
        x, = self.inputs
        return lambda: dy * 2 * x,

def square(x):
    return Square().apply(x)

class Exp(Function):
    def forward(self, x):
        return np.exp(x)
    
    def backward(self, dy):
        return lambda: dy * self.y(),

def exp(x):
    return Exp().apply(x)

class Log(Function):
    def forward(self, x):
        return np.log(x)
    
    def backward(self, dy):
        x, = self.inputs
        return lambda: dy / x,

def log(x):
    return Log().apply(x)

Tensor.log = lambda x: Log().apply(x)

class Mean(Function):
    def __init__(self, axis=None, keepdims=False):
        self.axis = axis
        self.keepdims = keepdims

    def forward(self, x):
        return np.mean(x, axis=self.axis, keepdims=self.keepdims)
    
    def backward(self, dy):
        x, = self.inputs
        num = x.size // dy.size
        return lambda: dy / num,

def mean(x, axis=None, keepdims=False):
    return Mean(axis, keepdims).apply(x)

class Tanh(Function):
    def forward(self, x):
        return np.tanh(x)

    def backward(self, dy):
        return lambda: dy * (1 - self.y()**2),

def tanh(x):
    return Tanh().apply(x)