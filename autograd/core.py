from .numpy import *
from typing import Self, Callable
import weakref

# 인스턴스 변수(동일한 클래스 내에서 공유되는 변수)
class Config:
    backprop: bool = True
    dtype: DTypeLike = np.float32

class Tensor:
    # 배열 요소
    value: NDArray | None
    grad: Self | None

    # 그래프 요소
    requires_grad: bool
    parents: tuple[Self, ...] | None
    grad_fn: Callable[[Self], tuple[Callable[[], Self], ...]] | None

    # CuPy의 경우 우선 순위 값이 100이기 때문에 그보다 높은 101로 한다.
    __array_priority__ = 101

    def __init__(self, value: ArrayLike, requires_grad=False):
        # 내부값은 shape, size, ndim, dtype을 사용할 수 있도록 NDArray로 변환된다.
        if not isinstance(value, np.ndarray):
            value = np.asarray(value)

        self.value = value
        self.grad = None
        self.requires_grad = requires_grad
        self.parents = None
        self.grad_fn = None

    # NDArray처럼 사용해도 위화감이 없도록 다양한 메서드를 구현했다.
    @property
    def shape(self): return self.value.shape

    @property
    def size(self): return self.value.size

    @property
    def ndim(self): return self.value.ndim

    @property
    def dtype(self): return self.value.dtype

    # print문
    def __repr__(self):
        replaced = self.value.__str__().replace('\n ', '\n        ')
        return f"Tensor({replaced})"
        
    def __format__(self, __format_spec: str):
        return self.value.__format__(__format_spec)

    # 딕셔너리에서 key로 사용될 수 있도록 한다.
    def __hash__(self):
        return id(self)
    
    # 비교 연산은 역전파가 가능한 연산이 아니기 때문에 이렇게 구현했다.
    def __eq__(self, other):
        return self.value == other

    def __lt__(self, other):
        return self.value < other

    def __le__(self, other):
        return self.value <= other

    def __gt__(self, other):
        return self.value > other

    def __ge__(self, other):
        return self.value >= other

    # 그래프 메서드
    def attach(self, parents, grad_fn):
        self.parents = parents
        self.grad_fn = grad_fn

    def detach(self):
        self.parents = None
        self.grad_fn = None
    
    # 역전파 코드의 직관성을 위해서 추가했다.
    @property
    def is_leaf(self):
        return self.parents is None

    def clear_grad(self):
        self.grad = None

    def clear_value(self):
        self.value = None

def get_vals(inputs: tuple):
    return tuple(input.value if isinstance(input, Tensor) else input for input in inputs)

class Function:
    """역전파에서 연산의 단위로 사용되는 클래스. \\
    새로운 함수를 만드려면 Function을 상속받아 forward와 backward를 구현하면 된다. \\
    Exp 함수는 다음과 같이 구현 가능하다.
    ```
    class Exp(Function):
        def forward(self, x):
            return np.exp(x)
        
        def backward(self, dy):
            return lambda: dy * self.y(),
    ```
    """
    def forward(self, *inputs: ArrayLike) -> ArrayLike:
        raise NotImplementedError()

    def backward(self, dy: ArrayLike | Tensor) -> tuple[Callable[[], Tensor], ...]:
        raise NotImplementedError()
    
    def apply(self, *inputs):
        y = self.forward(*get_vals(inputs))

        # 역전파가 꺼져있을 경우 Function의 출력은 입력값들의 단순 계산이 된다.
        if not Config.backprop:
            return y
        
        y = Tensor(y)
        y.attach(parents=inputs, grad_fn=lambda dy: self.backward(dy))
        
        self.inputs = inputs
        self.y = weakref.ref(y)

        return y

# 브로드캐스팅을 위한 Sum, Reshape
class Sum(Function):
    def __init__(self, axis: bool=None, keepdims: bool=False):
        self.axis = axis
        self.keepdims = keepdims
    
    def forward(self, x):
        y = np.sum(x, axis=self.axis, keepdims=True)

        if not self.keepdims:
            y = np.squeeze(y, axis=self.axis)

        return y

    def backward(self, dy):
        return lambda: dy.reshape(self.y().shape),

Tensor.sum = lambda x, axis=None, keepdims=False: Sum(axis, keepdims).apply(x)

class Reshape(Function):
    def __init__(self, newshape):
        self.newshape = newshape

    def forward(self, x):
        return np.reshape(x, newshape=self.newshape)

    def backward(self, dy):
        x_shape = self.inputs[0].shape

        return lambda: dy.reshape(x_shape),

Tensor.reshape = lambda x, newshape: Reshape(newshape).apply(x)

def broadcasted_axis(shape: tuple[int, ...], ndim: int) -> tuple[int, ...]:
    if len(shape) > ndim:
        return tuple()

    return tuple(idx for idx, axis in enumerate((ndim - len(shape))*(1, ) + shape) if axis == 1)

def unbroadcast(target: Tensor, grad: Tensor):
    """브로드캐스팅된 배열을 다시 되돌린다."""
    assert not target.size > grad.size, "target is not broadcasted" # Sum 연산

    if target.shape == grad.shape: # 원소 별 연산
        return grad

    axis = broadcasted_axis(target.shape, grad.ndim)
    grad = grad.sum(axis=axis, keepdims=True)

    if grad.size == target.size:
        return grad.reshape(target.shape)

    return grad