from .core import *

def contains_only_is(iterable, other):
    """__contains__는 같은 원소로 판단하는 조건이 x == element or x is element이다.
    그러나 이 함수는 조건이 오직 x is element이다.
    """
    return any(other is item for item in iterable)

def toposort_sub(node, visited: list[Tensor], result: list[Tensor]):
    if not isinstance(node, Tensor) or node.is_leaf:
        return

    visited.append(node)

    for parent in node.parents:
        if contains_only_is(visited, parent):
            continue

        toposort_sub(parent, visited, result)

    result.append(node)

def toposort(node):
    result = []
    toposort_sub(node, [], result)
    return result[::-1]

def backward(last, grad: NDArray | Tensor | None=None, retain_graph=False, create_graph=False):
    if grad is None:
        grad = Tensor(np.ones_like(last.value))

    grads = {last: grad}

    backprop = Config.backprop
    Config.backprop = create_graph

    # 역전파
    for node in toposort(last):
        fns = node.grad_fn(grads[node])

        for parent, fn in zip(node.parents, fns):
            if not isinstance(parent, Tensor) or not parent.requires_grad and parent.is_leaf:
                continue

            if grads.get(parent) is None:
                grads[parent] = fn()
                continue

            grads[parent] = grads[parent] + fn()

    Config.backprop = backprop

    # 기울기 반영
    # 이 과정에서 불필요한 배열을 제거한다.
    for node, grad in grads.items():
        # 메모리 해제를 코드에 포함하면 벤치마크에서 속도가 대폭 향상된다.
        is_intermediate = not (create_graph or node.is_leaf) # not create_graph and not node.is_leaf
        if is_intermediate: # 중간값은 기울기를 반영하지 않음
            node.detach() # 메모리 해제
            node.clear_grad()
            continue

        if retain_graph and node.grad is not None:
            node.grad = node.grad + grad
            continue

        node.grad = grad

Tensor.backward = backward