import inspect
from functools import wraps
from typing import Dict, Callable, Set, List

import numpy as np

from mcfunction_lib.mcfunction import mcfunction
from gymnasium.spaces import Box, Discrete


class ObservationWrapper:
    def __init__(self, func, *, raw_func, mc_function: Callable, private: bool, lower: float, upper: float):
        self._func = func
        self.raw = raw_func
        self.mc_function = mc_function
        self._private = private
        self.lower = lower
        self.upper = upper

        self.name = self._func.__name__

        # Make the instance itself callable
        wraps(func)(self)

    def __call__(self, *args, **kwargs):
        return self._func(*args, **kwargs)


def observation(func=None, *, mc_function: Callable, raw: Callable, private=False, lower=-3, upper=3):
    if func is None:
        return lambda f: ObservationWrapper(f, raw_func=raw, mc_function=mc_function, private=private, lower=-3., upper=3.)
    return ObservationWrapper(func, raw_func=raw, mc_function=mc_function, private=private,lower=-3., upper=3.)


class Observable:
    def __init__(self):
        self.observables: Dict[str, ObservationWrapper] = {}
        self.public_observables: Dict[str, ObservationWrapper] = {}

        for name, member in inspect.getmembers(self, predicate=lambda f: isinstance(f, ObservationWrapper)):
            if hasattr(member, "_private"):
                if not member._private:
                    self.public_observables[self.name + "." + member.name] = member
                self.observables[self.name + "." + member.name] = member


    def observe(self, observe_private: bool = True, **kwargs):
        obs_ops = self.observables if observe_private else self.public_observables

        observations = []
        for op in obs_ops.values():
            x = op(**kwargs)
            if isinstance(x, list):
                observations.extend(x)
            else:
                observations.append(x)

        return observations

    @mcfunction
    def fetch(self, **kwargs):
        fetch = """
        execute unless entity @s[nbt={DeathTime:0s}] run tag @s add dying
        execute if entity @s[tag=dying] run return fail
        """
        for obs_path, obs in self.observables.items():
            fetch += f"""
            # setup observation {self.name}
            {obs.mc_function(mob=self, **kwargs)}
            data modify storage ai {obs_path} set from storage ai obs_tmp
            """

    def specs(self, observe_private: bool = True):
        lows = []
        highs = []
        for obs in self.observables.values():
            lows.append(obs.lower)
            highs.append(obs.upper)

        return lows, highs

    def observation_space(self, mobs):

        lows = []
        highs = []

        for mob in mobs:
            nlows, nhighs = mob.specs(mob.name == self.name)
            lows.extend(nlows)
            highs.extend(nhighs)

        return Box(np.array(lows, dtype=np.float32), np.array(highs, dtype=np.float32))


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

        # Make the instance itself callable
        wraps(func)(self)

    def __call__(self, *args, **kwargs):
        return self._func(*args, **kwargs)


def toggle(func=None, *,  mc_function: Callable):
    if func is None:
        return lambda f: ToggleWrapper(f, mc_function=mc_function)
    return ToggleWrapper(func, mc_function=mc_function)


class Controllable:
    def __init__(self):
        self.actions: List[ActionWrapper] = []
        self.toggles: Set[int] = set()
        self.toggled = None

        for name, member in inspect.getmembers(self, predicate=lambda f: isinstance(f, ToggleWrapper)):
            member.index = len(self.actions)
            self.actions.append(member)
            self.toggles.add(member.index)

        for name, member in inspect.getmembers(self, predicate=lambda f: isinstance(f, ActionWrapper)):
            self.actions.append(member)

        self.num_toggles = len(self.toggles)

    def act(self, action_index: int, **kwargs):
        if action_index in self.toggles:
            self.toggled = action_index
        else:
            self.actions[action_index](**kwargs)

    def apply_toggle(self, **kwargs):
        if self.toggled is not None:
            self.actions[self.toggled](**kwargs)

    @mcfunction
    def step(self, module_name, **kwargs):
        return f"""
        # return if dying
        execute if entity @s[tag=dying] run return run ai:ai_modules/{module_name}/{self.name}/on_death

        # infer action
        function ai:ai_modules/{module_name}/{self.name}/inference/node_0

        # preprocess
        execute store result score @s vx run data get entity @s Motion[0] 100000
        execute store result score @s vy run data get entity @s Motion[1] 100000
        execute store result score @s vz run data get entity @s Motion[2] 100000

        execute store result score @s x run data get entity @s Pos[0] 100000
        execute store result score @s y run data get entity @s Pos[1] 100000
        execute store result score @s z run data get entity @s Pos[2] 100000

        scoreboard players operation #n x = @s x
        scoreboard players operation #n z = @s z

        # the OnGround tag is not perfectly reliable
        tag @s remove on_ground
        execute if entity @s[nbt={{OnGround:1b}}] if score @s vy matches -7841 run tag @s add on_ground

        # take action
        # todo: see if this is ok, or we need to split functions into multiple, with each 8 possible inputs
        execute store result storage ai action.id int 1 run scoreboard players get @s action
        function ai:ai_modules/{module_name}/{self.name}/take_action with storage ai action 

        # toggles
        execute store result storage ai toggle.id int 1 run scoreboard players get @s toggled
        function ai:ai_modules/{module_name}/{self.name}/apply_toggle with storage ai toggle 

        # compute changes in velocity
        scoreboard players operation #n x -= @s x
        scoreboard players operation #n z -= @s z
        scoreboard players operation @s vx += #n x
        scoreboard players operation @s vz += #n z

        # apply changes to entity
        execute store result entity @s Motion[0] double 0.00001 run scoreboard players get @s vx
        execute store result entity @s Motion[2] double 0.00001 run scoreboard players get @s vz
        execute store result entity @s Motion[1] double 0.00001 run scoreboard players get @s vy

        # post_process
        data remove entity @s last_hurt_by_mob
        """

    @mcfunction
    def take_action(self, module_name, **kwargs):
        return f"""
    $function ai:ai_modules/{module_name}/{self.name}/action_$(id)
    """

    @mcfunction
    def apply_toggle(self, module_name, **kwargs):
        return f"""
    $function ai:ai_modules/{module_name}/{self.name}/toggle_$(id)
    """

    @mcfunction
    def action(self, module_name, **kwargs):
        actions = []
        for index, act in enumerate(self.actions):
            if index in self.toggles:
                actions.append(f"execute if score @s action matches {index} run scoreboard players set @s toggled {index}")
            else:
                actions.append(act.mc_function(mob=self, module_name=module_name, **kwargs))
        return actions

    @mcfunction
    def toggle(self, module_name, **kwargs):
        return [self.actions[tog].mc_function(mob=self, module_name=module_name, **kwargs) for tog in self.toggles]

    def action_space(self):
        return Discrete(len(self.actions))


