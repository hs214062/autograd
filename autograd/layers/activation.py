from ..core import *

def dropout(x, rate=0.2):
    """입력값의 일부를 0으로 만들어주는 함수. rate가 None인 경우 연산을 적용하지 않는다."""
    if not rate:
        return x
    
    return x * ((np.random.rand(*x.shape) > rate).astype(Config.dtype) / (1 - rate))

def relu(x):
    """0이하의 값을 0으로 만드는 함수."""
    return x * (x > 0)

class Sigmoid(Function):
    def forward(self, x):
        return 1 / (1 + np.exp(-x))
    
    def backward(self, dy):
        return lambda: dy * self.y() * (1 - self.y()),

def sigmoid(x):
    """입력을 0과 1사이로 만들어주는 함수. 로지스틱 함수라고도 한다."""
    return Sigmoid().apply(x)

class SiLU(Function):
    def forward(self, x):
        self.sigmoid_y = 1 / (1 + np.exp(-(1.702 * x)))
        return x * self.sigmoid_y
    
    def backward(self, dy):
        x, = self.inputs
        return lambda: (dy * self.sigmoid_y) * (1 + 1.702 * x * (1 - self.sigmoid_y)),

def silu(x):
    return SiLU().apply(x)

class Softmax(Function):
    def __init__(self, axis=-1):
        self.axis = axis
    
    def forward(self, x):
        e = np.exp(x - np.max(x, axis=self.axis, keepdims=True))
        return e / np.sum(e, axis=self.axis, keepdims=True)

    def backward(self, dy):
        dx = self.y() * dy

        return lambda: dx - (self.y() * dx.sum(axis=self.axis, keepdims=True)),

def softmax(x, axis=-1):
    return Softmax(axis).apply(x)

class CrossEntropy(Function):
    def __init__(self, t):
        self.t = t
    
    def forward(self, y):
        return -np.float64(np.sum(self.t * np.log(1e-06 + y)))

def cross_entropy(y, t):
    return CrossEntropy(t).apply(y)

class CrossEntropyLoss(Function):
    def __init__(self, t, axis=-1):
        self.t = t
        self.axis = axis

    def forward(self, y):
        e = np.exp(y - np.max(y, axis=self.axis, keepdims=True))
        out = e / np.sum(e, axis=self.axis, keepdims=True)
        self.out = out

        reshaped = out.reshape(-1, out.shape[-1])
        return -np.float64(np.sum(np.log(1e-06 + reshaped[np.arange(reshaped.shape[0]), self.t.ravel()]))) / reshaped.shape[0]

    def backward(self, dy):
        def dx():
            reshaped = self.out.reshape(-1, self.out.shape[-1])
            reshaped[np.arange(reshaped.shape[0]), self.t.ravel()] -= 1
            return (dy * self.out / reshaped.shape[0]).astype(Config.dtype)

        return dx,

def crossentropy_loss(y, t, axis=-1):
    return CrossEntropyLoss(t, axis).apply(y)