import os
from collections import defaultdict
from pprint import pprint
from typing import Optional, Dict, List

import imageio
from matplotlib import pyplot as plt
from polaris.environments import PolarisEnv
from gymnasium.spaces import Discrete, Box
import numpy as np

from mc_ai.arena import Arena
from mc_ai.mc_types import MobConfig, load_mob
from mc_ai.mob import Mob


class MiniMinecraft(PolarisEnv):
    env_id = "MiniMinecraft"

    def __init__(
            self,
            attackers: List[MobConfig],
            defenders: List[MobConfig],
            layout: str,
            env_index: int = -1,
            tick_skip: int = 1,
            render: bool = True,
            record_raw_observations=False,
            random_raw_observations=False,
            raw_position_offset=(0, 0, 0),
            **kwargs
    ):
        # uses the player's dynamics to learn a bot
        PolarisEnv.__init__(self, env_id="mini_minecraft")
        self.arena = Arena(layout)
        self.tick_skip = tick_skip

        if not isinstance(attackers[0], MobConfig):
            attackers = [MobConfig(*args) for args in attackers]
            defenders = [MobConfig(*args) for args in defenders]

        self.attackers = {mob.name: mob for mob in attackers}
        self.defenders = {mob.name: mob for mob in defenders}

        self.allies = {
            mob_name: [ally for ally in group if ally != mob_name]
            for group in (self.attackers, self.defenders)
            for mob_name in group
        }

        self.opponents = {
            mob_name: [opponent for opponent in
                       (self.defenders if group is self.attackers else self.attackers)]
            for group in (self.attackers, self.defenders)
            for mob_name in group
        }

        self.mobs: Dict[str, Mob]  = {}

        self._agent_ids = set(self.attackers.keys()) | set(self.defenders.keys())
        if len(self._agent_ids) < len(self.attackers) + len(self.defenders):
            raise ValueError("Conflicting mob names.")

        self.should_render = render
        self.num_players = len(self._agent_ids)
        self.episode_length = 4096
        self.step_count = 0
        self.num_episodes = 0
        self.render_freq = 100
        self.env_index = env_index
        self.record_raw_observations = record_raw_observations
        self.random_raw_observations = random_raw_observations
        self.raw_position_offset = raw_position_offset

        self.observation_paths = None

        self.reset()



    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ):

        if self.num_episodes % self.render_freq == 0 and self.env_index == 0:
            self._render = self.should_render
        else:
            self._render = False

        self.episode_length = 512 if self._render else 4096
        self.num_episodes += 1
        self.step_count = 0

        self.metrics = {}

        self.mobs = {}
        for attacker in self.attackers.values():
            self.mobs[attacker.name] = load_mob(attacker)
        for defender in self.defenders.values():
            self.mobs[defender.name] = load_mob(defender)

        for i, (aid, mob) in enumerate(self.mobs.items()):
            pos = self.arena.sample_start_pos(mob)
            mob.move(*pos)

        if self.record_raw_observations:
            if self.observation_paths is None:
                self.observation_paths = {
                    aid: self.get_raw_obs(aid)
                    for aid in self.attackers
                }

            self.metrics["observation_paths"] = self.observation_paths
            self.metrics["raw_observations"] = {
                aid: []
                for aid in self.attackers
            }
            self.metrics["actions"] = {
                aid: []
                for aid in self.attackers
            }

        self.observation_space = {aid: Box(-3., 3., self.get_obs(aid).shape) for aid, mob in self.mobs.items()}
        self.action_space = {aid: mob.action_space() for aid, mob in self.mobs.items()}

        self.render()
        return {aid: self.get_obs(aid) for aid in self.get_agent_ids()}, {aid: {} for aid in self.get_agent_ids()}

    def get_raw_obs(self, aid):
        nearest_opponent = self.get_nearest_opponent(aid)
        observations = self.mobs[aid].observe_raw(arena=self.arena, nearest_opponent=nearest_opponent, position_offset=self.raw_position_offset)
        # only get info about nearest opponent
        for ally in self.allies[aid]:
            observations.update(self.mobs[ally].observe_raw(arena=self.arena, position_offset=self.raw_position_offset, observe_private=False))

        observations.update(nearest_opponent.observe_raw(arena=self.arena, position_offset=self.raw_position_offset, observe_private=False))

        return observations

    def sample_random_obs(self, aid, size=(1,)):
        nearest_opponent = self.get_nearest_opponent(aid)
        observations, raw_observations = self.mobs[aid].sample_obs(arena=self.arena, position_offset=self.raw_position_offset, size=size)
        # only get info about nearest opponent
        for ally in self.allies[aid]:
            obs, raw = self.mobs[ally].sample_obs(arena=self.arena, position_offset=self.raw_position_offset, size=size, observe_private=False)
            observations.extend(obs)
            raw_observations.update(raw)

        obs, raw = nearest_opponent.sample_obs(arena=self.arena, position_offset=self.raw_position_offset, size=size, observe_private=False)
        observations.extend(obs)
        raw_observations.update(raw)

        return observations, raw_observations

    def get_nearest_opponent(self, aid):

        nearest = None
        d = np.inf

        for opponent in self.opponents[aid]:
            mob = self.mobs[opponent]
            nd = self.mobs[aid].distance(mob)
            if nd < d:
                nearest = mob
                d = nd
        return nearest

    def get_obs(self, aid):
        nearest_opponent = self.get_nearest_opponent(aid)
        observations = self.mobs[aid].observe(arena=self.arena, nearest_opponent=nearest_opponent)

        # only get info about nearest opponent
        for ally in self.allies[aid]:
            observations.extend(self.mobs[ally].observe(arena=self.arena, observe_private=False))

        observations.extend(nearest_opponent.observe(arena=self.arena, observe_private=False))

        return np.clip(observations, -3., 3., dtype=np.float32)

    def make_record(self):
        #with imageio.get_writer(f"mini_mc_movie.gif", mode="I", fps=2) as writer:
        with imageio.get_writer(f"mini_mc_movie_realtime.gif", mode="I", fps=round(20/self.tick_skip)) as fwriter:
            for frame in self.arena.renderer.frames:
                #writer.append_data(frame)
                fwriter.append_data(frame)
        self.arena.renderer.frames = []

    def render(self):
        if not self._render:
            return
        self.arena.render(self.mobs)

    def step(
            self,
            actions: Dict[str, int]
    ):
        if self.random_raw_observations and self.random_raw_observations:
            for aid in self.attackers:
                self.metrics["raw_observations"][aid].append([obs for obs in self.get_raw_obs(aid).values()])
                self.metrics["actions"][aid].append(actions[aid])
            self.step_count += 1
            done = self.step_count == self.episode_length
            dones =  {aid: done for aid in self.mobs}
            dones["__all__"] = done
            return ({aid: self.observation_space[aid].sample() for aid in self.mobs},
                    {aid: 0. for aid in self.mobs},
                    dones, dones,
                    {aid: {} for aid in self.mobs})

        # 1: apply friction
        # 2: apply gravity
        # 3: actions
        # 4: knockbacks
        # 5: tick

        healths = {aid: p.health for aid, p in self.mobs.items()}
        for _ in range(self.tick_skip):
                for mob in self.mobs.values():
                    mob.apply_friction()
                    mob.apply_gravity()
                if _ == 0:
                    for aid, mob in self.mobs.items():
                        allies = [self.mobs[ally] for ally in self.allies[aid]]
                        opponents = [self.mobs[opponent] for opponent in self.opponents[aid]]
                        mob.act(actions[aid], allies=allies, opponents=opponents, arena=self.arena)
                for mob in self.mobs.values():
                    mob.tick(self.arena)
                self.step_count += 1

        damages = {
            aid: (healths[aid] - p.health) for aid, p in self.mobs.items()
        }

        trunc = self.step_count == self.episode_length
        truncs = {
            "__all__": trunc
        }

        attacker_deaths = {aid: self.mobs[aid].health == 0. for aid in self.attackers}
        defender_deaths = {aid: self.mobs[aid].health == 0. for aid in self.defenders}

        done = all(attacker_deaths.values()) or all(defender_deaths.values())

        if done:
            self.metrics["attacker_win"] = 1-int(all(attacker_deaths.values()))
        dones = {
            aid: done for aid in self.get_agent_ids()
        }
        dones["__all__"] = done

        if (done or trunc) and self._render:
            self.make_record()

        rews = {}
        all_deaths = attacker_deaths | defender_deaths
        for aid, mob in self.mobs.items():
            r = - mob.actions[actions[aid]].cost - 100 * int(all_deaths[aid]) - damages[aid]
            for ally in self.allies[aid]:
                r += - 100 * int(all_deaths[ally]) - damages[ally]
            for opponent in self.opponents[aid]:
                r += 100 * int(all_deaths[opponent]) + damages[opponent]
            rews[aid] = r

        obs = {
            aid: self.get_obs(aid) for aid in self.get_agent_ids()
        }
        if self.record_raw_observations and not self.random_raw_observations:
            for aid in self.attackers:
                self.metrics["raw_observations"][aid].append([obs for obs in self.get_raw_obs(aid).values()])
                self.metrics["actions"][aid].append(actions[aid])


        info = {
            aid: {} for aid in self.get_agent_ids()
        }

        self.render()

        return obs, rews, dones, truncs, info

    def get_episode_metrics(self):

        return self.metrics.copy()


if __name__ == '__main__':

    np.random.seed(1)
    repeat = 1
    skip = True

    attackers = [
        MobConfig(
            name="zombie1",
            cls="mc_ai.mob.PlayerLike",
            config=dict(
                position_offset=(1, 0, 1)
            )
        )
    ]

    defenders = [
        MobConfig(
            name="player",
            cls="mc_ai.mob.PlayerLike",
            config=dict(
                position_offset=(-1, 0, -1)
            )
        )
    ]

    env = MiniMinecraft(
        attackers=attackers,
        defenders=defenders,
        layout="arena_test",
        tick_skip=1 if not skip else repeat,
        env_index=0
    )

    r = 1 if skip else repeat
    n = 200
    acts = {
        "zombie1": [1] + [4] * 19,
        "player": [9] * 20,
    }
    n = len(acts["zombie1"])

    print(acts)

    vz = []
    pz = []
    env.mobs["zombie1"].x = 5.5
    env.mobs["zombie1"].y = 1.01
    env.mobs["zombie1"].z = 5.5

    env.mobs["player"].x = 7.5
    env.mobs["player"].y = 1.01
    env.mobs["player"].z = 3.5

    for t in range(n):
        actions = {
            aid: acts[aid][t]
            for aid in env.get_agent_ids()
        }
        for s in range(r):
            env.step(actions)
        #print(f"[rot][{env.players[0].rot * 180 / np.pi}f,0.0f]")
        #print(f"[xyz][{env.players[0].x - env.arena.center[0]}d,{env.players[0].y - 60 - 1}d,{env.players[0].z - env.arena.center[2]}d]")


    # plt.plot(range(len(vz)), vz, label="vz")
    # plt.plot(range(len(pz)), pz, label="pz")

    plt.show()
    env.make_record()







