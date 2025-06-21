import os
from collections import defaultdict
from pprint import pprint
from typing import Optional, Dict

import imageio
from matplotlib import pyplot as plt
from polaris.environments import PolarisEnv
from gymnasium.spaces import Discrete, Box
import numpy as np

from minecraft_ai.arena import Arena
from minecraft_ai.mc_types import Player


class MiniMinecraft(PolarisEnv):
    env_id = "MiniMinecraft"

    def __init__(
            self,
            env_index: int = -1,
            layout: str = None,
            tick_skip: int = 1,
            render: bool = False,
            **kwargs
    ):
        # uses the player's dynamics to learn a bot
        PolarisEnv.__init__(self, env_id="mini_minecraft")
        self.arena = Arena(layout)
        self.tick_skip = tick_skip
        self._agent_ids = {0,1}
        self._render = render
        self.num_players = len(self._agent_ids)
        self.episode_length = 2048
        self.step_count = 0
        self.num_episodes = 0
        self.render_freq = 30
        self.env_index = env_index

        self.reset()
        self.observation_space = {aid: Box(-10., 10., self.get_obs(aid).shape)
                                  for aid in self.get_agent_ids()}
        self.action_space = {aid: Discrete(len(self.players[aid].actions)) for aid in self.get_agent_ids()}



    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ):
        #aid0 = list(self._agent_ids)[0]
        # if options is not None and "render" in options[aid0]:
        #     self._render = options[aid0]["render"]

        if self.num_episodes % self.render_freq == 0 and self.env_index == 0:

            self._render = True
        else:
            self._render = False

        self.episode_length = 512 if self._render else 2048
        self.num_episodes += 1
        self.step_count = 0

        self.metrics = defaultdict(float)

        self.players = {
            aid: Player(aid=aid)
            for aid in self.get_agent_ids()
        }

        pos = [
            (-1., 1.),
            (1., -1.)
        ]

        for i, (aid, player) in enumerate(self.players.items()):
            x, y, z = self.arena.center
            x0, z0 = pos[aid]
            player.move_to(x + x0, y+0.1, z + z0)

        self.opponents = {
            0: [self.players[1]],
            1: [self.players[0]]
        }

        self.render()

        return {aid: self.get_obs(aid) for aid in self.get_agent_ids()}, {aid: {} for aid in self.get_agent_ids()}


    def get_obs(self, aid):
        # TODO: handle more than 2 players
        other = (1 + aid) % 2
        arena_dims = (self.arena.dx, self.arena.dy, self.arena.dz)
        state = self.players[aid].observe(
            arena_dims=arena_dims,
            motion_scale=2.
        ) + self.players[other].observe(
            arena_dims=arena_dims,
            motion_scale=2.,
            exclude=("toggle",),
        )

        return np.clip(state, -3., 3., dtype=np.float32)

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
        self.arena.render(self.players)


    def step(
            self,
            actions: Dict[str, int]
    ):
        # 1: apply friction
        # 2: apply gravity
        # 3: actions
        # 4: knockbacks
        # 5: tick

        healths = {aid: p.health for aid, p in self.players.items()}
        for _ in range(self.tick_skip):
                for aid, player in self.players.items():
                    player.apply_friction()
                    player.apply_gravity()
                if _ == 0:
                    for aid, player in self.players.items():
                        player.act(Action(actions[aid]), self.opponents[aid], self.arena)
                for aid, player in self.players.items():
                    player.tick(self.arena)
                self.step_count += 1

        damages = {
            aid: (healths[aid] - p.health) for aid, p in self.players.items()
        }

        trunc = self.step_count == self.episode_length
        truncs = {
            "__all__": trunc
        }

        # TODO handle teams
        deaths = {aid: p.health == 0 for aid, p in self.players.items()}
        done = any(list(deaths.values()))

        if done:
            self.metrics["win"] = 1-int(deaths[0])
        dones = {
            aid: done for aid in self.get_agent_ids()
        }
        dones["__all__"] = done

        if (done or trunc) and self._render:
            self.make_record()


        rotation_costs = {
            Action.ROTATE_LEFT: 0.01,
            Action.ROTATE_RIGHT: 0.01,
            Action.ROTATE_HARD_LEFT: 0.04,
            Action.ROTATE_HARD_RIGHT : 0.04,
            Action.ATTACK: 0.01
        }

        costs = {
            aid: rotation_costs.get(Action(action), 0) for aid, action in actions.items()
        }

        rews = {
            aid: sum(damages[opp.aid]+ 100 * int(deaths[opp.aid]) for opp in self.opponents[aid]) - damages[aid] - costs[aid] - 100 * deaths[aid] for aid in self.get_agent_ids()
        }
        obs = {
            aid: self.get_obs(aid) for aid in self.get_agent_ids()
        }
        info = {
            aid: {} for aid in self.get_agent_ids()
        }

        p = self.players[1]

        self.metrics["distance"] += self.players[0].distance((p.x, p.y, p.z))

        self.render()

        return obs, rews, dones, truncs, info

    def get_episode_metrics(self):
        self.metrics["distance"] /= self.step_count

        return self.metrics.copy()


if __name__ == '__main__':

    np.random.seed(40)

    repeat = 1
    skip = True
    env = MiniMinecraft(
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







