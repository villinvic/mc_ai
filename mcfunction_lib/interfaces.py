import inspect
from typing import List, Set, Dict

import numpy as np
from gymnasium.spaces import Discrete, Box

from mcfunction_lib.decorators import ActionWrapper, ToggleWrapper, ObservationWrapper, observation
from mcfunction_lib.compile import mcfunction

def onehot(n: int, i: np.ndarray):
    oh = np.zeros((len(i), n), dtype=np.float32)
    oh[np.arange(len(i)), i] = 1.
    return oh

class Controllable:
    def __init__(self):
        self.actions: List[ActionWrapper] = []
        self.toggles: Set[int] = set()
        self.toggled = 0
        self.action_from_name: Dict[str, int] = {}

        for name, member in inspect.getmembers(self, predicate=lambda f: isinstance(f, ToggleWrapper)):
            member.index = len(self.actions)
            self.actions.append(member)
            self.toggles.add(member.index)
            self.action_from_name[member.name] = member.index

        for name, member in inspect.getmembers(self, predicate=lambda f: isinstance(f, ActionWrapper)):
            member.index = len(self.actions)
            self.actions.append(member)
            self.action_from_name[member.name] = member.index

        self.num_toggles = len(self.toggles)

    def act(self, action_index: int, **kwargs):
        if action_index is None:
            return
        if action_index in self.toggles:
            self.toggled = action_index
        else:
            self.actions[action_index](self, **kwargs)

    def apply_toggle(self, **kwargs):
        if self.toggled is not None:
            self.actions[self.toggled](self, **kwargs)

    @mcfunction
    def step(self, module_name, **kwargs):
        return f"""
        # return if dying
        execute if entity @s[tag=dying] run return run function ai:ai_modules/{module_name}/{self.name}/on_death

        # infer action
        function ai:ai_modules/{module_name}/{self.name}/inference/node_0

        # preprocess
        execute store result score @s vx run data get entity @s Motion[0] 100000
        execute store result score @s vz run data get entity @s Motion[2] 100000

        scoreboard players operation #n x = @s x
        scoreboard players operation #n z = @s z

        # take action
        # todo: see if this is ok, or we need to split functions into multiple, with each 8 possible inputs
        execute store result storage ai action.id int 1 run scoreboard players get @s action
        function ai:ai_modules/{module_name}/{self.name}/take_action with storage ai action 

        # toggles
        execute store result storage ai toggle.id int 1 run scoreboard players get @s toggled
        function ai:ai_modules/{module_name}/{self.name}/step_toggle with storage ai toggle 

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
    def step_toggle(self, module_name, **kwargs):
        return f"""
    $function ai:ai_modules/{module_name}/{self.name}/toggle_$(id)
    """

    @observation(
        raw=lambda self, **kwargs: -1 if self.toggled is None else self.toggled,
        mc_function=lambda obs_path, **_: f"execute store result storage ai {obs_path} int 1 run scoreboard players get @s toggled",
        sampler=lambda self, size, **kwargs: obs_raw_tuple(np.random.randint(-1, self.num_toggles, size), lambda obs: onehot(self.num_toggles, obs)),
        private=True,
    )
    def toggle_obs(self, **kwargs):
        one_hot_toggle = [0] * self.num_toggles
        if self.toggled is not None:
            one_hot_toggle[self.toggled] = 1
        return one_hot_toggle

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
            x = op(self, **kwargs)
            if isinstance(x, list):
                observations.extend(x)
            else:
                observations.append(x)

        return observations

    def observe_raw(self, observe_private: bool = True, **kwargs):
        obs_ops = self.observables if observe_private else self.public_observables

        observations = {}
        for name, op in obs_ops.items():
            x = op.raw(self, **kwargs)
            observations[name] =x

        return observations

    def sample_obs(self, observe_private: bool = True, **kwargs):
        obs_ops = self.observables if observe_private else self.public_observables
        raw_observations = {}
        observations = []
        for name, op in obs_ops.items():
            obs, raw = op.sampler(self, **kwargs)
            if isinstance(obs, list):
                observations.extend(obs)
            else:
                observations.append(obs)
            raw_observations[name] = raw

        return observations, raw_observations

    @mcfunction
    def fetch(self, **kwargs):
        fetch = """
        tag @s remove dying
        execute unless entity @s[nbt={DeathTime:0s}] run tag @s add dying
        execute if entity @s[tag=dying] run return fail
        
        # the OnGround tag is not perfectly reliable
        execute store result score @s vy run data get entity @s Motion[1] 100000
        tag @s remove on_ground
        execute if entity @s[nbt={OnGround:1b}] if score @s vy matches -7841 run tag @s add on_ground
        """
        for obs_path, obs in self.observables.items():
            fetch += f"""
            {obs.mc_function(mob=self, obs_path=obs_path, **kwargs)}
            """
        return fetch


def obs_raw_tuple(raw_obs, postprocess_func = lambda obs: obs):
    return postprocess_func(raw_obs), raw_obs

def rescale(scale=1., offset=0.):
    def rescaler(obs):
        return obs * scale + offset
    return rescaler
