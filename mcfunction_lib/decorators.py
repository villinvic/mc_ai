from functools import wraps
from typing import Callable


class ObservationWrapper:
    def __init__(self, func, *, raw_func, mc_function: Callable, sampler: Callable, private: bool):
        self._func = func
        self.raw = raw_func
        self.sampler = sampler
        self.mc_function = mc_function
        self._private = private

        self.name = self._func.__name__

        # Make the instance itself callable
        wraps(func)(self)

    def __call__(self, *args, **kwargs):
        return self._func(*args, **kwargs)


def observation(func=None, *, mc_function: Callable, raw: Callable, sampler: Callable, private=False):
    if func is None:
        return lambda f: ObservationWrapper(f, raw_func=raw, mc_function=mc_function, sampler=sampler, private=private)
    return ObservationWrapper(func, raw_func=raw, mc_function=mc_function, sampler=sampler, private=private)


class ActionWrapper:
    def __init__(self, func, *, mc_function: Callable, cost: float):
        self._func = func
        self.mc_function = mc_function
        self.cost = cost

        self.name = self._func.__name__
        self.index = None

        # Make the instance itself callable
        wraps(func)(self)

    def __call__(self, *args, **kwargs):
        return self._func(*args, **kwargs)


def action(func=None, *, mc_function: Callable, cost = 0):
    if func is None:
        return lambda f: ActionWrapper(f, mc_function=mc_function, cost=cost)
    return ActionWrapper(func, mc_function=mc_function, cost=cost)


class ToggleWrapper:
    def __init__(self, func, *, mc_function: Callable):
        self._func = func
        self.mc_function = mc_function

        self.name = self._func.__name__
        self.index = None
        self.cost = 0.

        # Make the instance itself callable
        wraps(func)(self)

    def __call__(self, *args, **kwargs):
        return self._func(*args, **kwargs)


def toggle(func=None, *,  mc_function: Callable):
    if func is None:
        return lambda f: ToggleWrapper(f, mc_function=mc_function)
    return ToggleWrapper(func, mc_function=mc_function)
