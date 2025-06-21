import os
from functools import wraps
from typing import Callable


def mcfunction(func: Callable):
    @wraps(func)
    def wrapper(*args, **kwargs):
        mc_path = kwargs.pop("mc_path", None) + "/" + func.__name__ + kwargs.pop("suffix", "") + ".mcfunction"
        result = func(*args, **kwargs)

        if mc_path:
            os.makedirs(os.path.dirname(mc_path), exist_ok=True)
            with open(mc_path, 'w') as f:
                f.write(result)

        return result

    return wrapper