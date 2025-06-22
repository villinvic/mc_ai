import os
from collections import defaultdict
from pprint import pprint
from typing import Optional, Dict, List

import imageio
from matplotlib import pyplot as plt
from polaris.environments import PolarisEnv
from gymnasium.spaces import Discrete, Box
import numpy as np

from minecraft_ai.arena import Arena
from minecraft_ai.mc_types import MobConfig, load_mob
from minecraft_ai.mob import Mob


class MiniMinecraft(PolarisEnv):
    env_id = "MiniMinecraft"

    def __init__(
            self,
            attackers: List[MobConfig],
            defenders: List[MobConfig],
            layout: str,
            env_index: int = -1,
            tick_skip: int = 1,
            render: bool = False,
            **kwargs
    ):
        # uses the player's dynamics to learn a bot
        PolarisEnv.__init__(self, env_id="mini_minecraft")
        self.arena = Arena(layout)
        self.tick_skip = tick_skip

        self.attackers = {mob.name: mob for mob in attackers}
        self.defenders = {mob.name: mob for mob in defenders}

        self.allies = {
            mob_name: [ally for ally in group.values() if ally.name != mob_name]
            for group in (self.attackers, self.defenders)
            for mob_name in group
        }

        self.opponents = {
            mob_name: [opponent for opponent in
                       (self.defenders if group is self.attackers else self.attackers).values()]
            for group in (self.attackers, self.defenders)
            for mob_name in group
        }

        self.mobs: Dict[str, Mob]  = {}

        self._agent_ids = set(self.attackers.keys()) | set(self.defenders.keys())
        if len(self._agent_ids) < len(self.attackers) + len(self.defenders):
            raise ValueError("Conflicting mob names.")

        self._render = render
        self.num_players = len(self._agent_ids)
        self.episode_length = 2048
        self.step_count = 0
        self.num_episodes = 0
        self.render_freq = 30
        self.env_index = env_index

        self.reset()
        self.observation_space = {aid: mob.observation_space(self.mobs) for aid, mob in self.mobs.items()}
        self.action_space = {aid: mob.action_space() for aid, mob in self.mobs.items()}

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ):

        if self.num_episodes % self.render_freq == 0 and self.env_index == 0:
            self._render = True
        else:
            self._render = False

        self.episode_length = 512 if self._render else 2048
        self.num_episodes += 1
        self.step_count = 0

        self.metrics = defaultdict(float)

        self.mobs = {}
        for attacker in self.attackers.values():
            self.mobs[attacker.name] = load_mob(attacker)
        for defender in self.defenders.values():
            self.mobs[defender.name] = load_mob(defender)

        for i, (aid, player) in enumerate(self.mobs.items()):
            x, y, z = self.arena.center
            player.init_at(x, y+0.1, z)

        self.render()
        return {aid: self.get_obs(aid) for aid in self.get_agent_ids()}, {aid: {} for aid in self.get_agent_ids()}

    def get_obs(self, aid):

        observations = self.mobs[aid].observe()

        for mob in self.mobs.values():
            if mob.name == aid:
                continue
            observations.extend(mob.observe(observe_private=False))

        return np.clip(observations, self.observation_space[aid].low, self.observation_space[aid].high, dtype=np.float32)

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
                        mob.act(actions[aid], allies=self.allies[aid], opponents=self.opponents[aid], arena=self.arena)
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
                r += - 100 * int(all_deaths[ally.name]) - damages[ally.name]
            for opponent in self.opponents[aid]:
                r += 100 * int(all_deaths[opponent.name]) + damages[opponent.name]
            rews[aid] = r

        obs = {
            aid: self.get_obs(aid) for aid in self.get_agent_ids()
        }
        info = {
            aid: {} for aid in self.get_agent_ids()
        }

        self.render()

        return obs, rews, dones, truncs, info

    def get_episode_metrics(self):

        return self.metrics.copy()


if __name__ == '__main__':

    np.random.seed(40)
    repeat = 1
    skip = True

    attackers = [
        MobConfig(
            name="zombie1",
            cls="minecraft_ai.mob.PlayerLike",
            config=dict(
                position_offset=(1, 0, 1)
            )
        )
    ]

    defenders = [
        MobConfig(
            name="player",
            cls="minecraft_ai.mob.PlayerLike",
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
    n = 500
    acts = {
        0: [1]*9 + [9]*10,
        1: [9] * 19,
    }
    n = len(acts[0])

    print(acts)

    vz = []
    pz = []

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







