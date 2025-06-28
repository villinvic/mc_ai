import importlib
from enum import Enum
from typing import Tuple, List, NamedTuple, Callable, Dict, Any

import numpy as np

from mcfunction_lib.compile import mcfunction

class Constants:
    # global
    GRAVITY_UP = 0.0828
    GRAVITY_DOWN = 0.075
    GRAVITY_GROUND = 0.0784

    # ground motion
    FRICTION = 0.45
    WALK_ACCELERATION = 0.0971
    SPRINT_ACCELERATION = WALK_ACCELERATION * 1.3
    JUMP_FORCE = 0.415 + GRAVITY_UP
    JUMP_BOOST = 0.2 # motion gained by jumping while sprinting

    # air motions
    FRICTION_AIR = 0.09 # for dpos out of hurt time, and motion in hurt time
    WALK_ACCELERATION_AIR = 0.0197
    SPRINT_ACCELERATION_AIR = WALK_ACCELERATION_AIR * 1.3

    # knockback motions
    # if on ground, knocked up plus horizontally, with dpos
    # multiplier if sprinting (but we can't apply it in game, we will just use base knockbacks)
    # if in air, no knock up
    # full knockback does not depend on distance, just the norm (dx,dz) vector.
    KNOCK_UP = 0.434
    KNOCK_BACK = 0.39
    DAMAGE_FRAME = 10

    # Lava / water physics ?
    FRICTION_WATER = 1


class MobConfig(NamedTuple):
    name: str
    cls: str
    config: dict[str, Any]


def load_mob(mob_config: MobConfig) -> "Mob":
    module_path, class_name = mob_config.cls.rsplit(".", 1)

    module = importlib.import_module(module_path)
    cls = getattr(module, class_name)
    return cls(name=mob_config.name, **mob_config.config)