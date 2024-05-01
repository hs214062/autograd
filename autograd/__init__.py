"""
자동으로 역전파를 해주는 패키지.

여기서 제공하는 Tensor, Function을 통해 역전파를 구현하지 않고 머신 러닝을 할 수 있다.

## 역전파 모드
Config.backprop이 True일 경우 모든 연산이 기록된다.

## 추론 모드
Config.backprop이 False일 경우 연산을 기록하지 않으며 Function의 출력은 입력의 단순 계산이 된다.
"""

from .core import *
from .backprop import *
from .base_function import *
from .numpy_function import *
from .utils import *
from .layers import *