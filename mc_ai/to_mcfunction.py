import os
import pickle
from collections import OrderedDict
import tree as dm_tree

import numpy as np
import six
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV

from sklearn.datasets import make_classification

import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn import tree
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier

from mc_ai.env import MiniMinecraft
from mc_ai.mc_types import MobConfig
from mcfunction_lib.build import build_ai_module

if __name__ == '__main__':
    module_name = "test"
    entity_name = "test"

    attackers = [
        MobConfig(
            name="zombie1",
            cls="mc_ai.mob.PlayerLike",
            config=dict(
                position_offset=(0, 0, 2)
            )
        )
    ]

    defenders = [
        MobConfig(
            name="player",
            cls="mc_ai.mob.PlayerLike",
            config=dict(
                position_offset=(0, 0, 0)
            )
        )
    ]

    env = MiniMinecraft(
        attackers=attackers,
        defenders=defenders,
        layout="arena_test",
        env_index=-1
    )
    env.num_episodes = 0

    position_offset = (-env.arena.dx / 2 , -59,-env.arena.dz / 2)


    num_episodes = 100

    states = []
    actions = []

    def simple_policy(action_space, raw_obs):
        if raw_obs["player.vel_y"] > 0.1:
            return "jump"

        return "rotate_left"

    def random_policy(action_space, raw_obs):
        return action_space.sample()

    policies = {
        "player": random_policy,
        "zombie1": simple_policy
    }

    for _ in range(num_episodes):
        print(_)
        done = False
        env.reset()
        while not done:

            obs = {aid: env.get_raw_obs(aid, position_offset)
                    for aid in env.get_agent_ids()
                      }
            acts = {
                aid: policies[aid](env.action_space[aid], obs[aid])
                for aid in env.get_agent_ids()
            }
            states.append([v for v in obs["zombie1"].values()])
            pred = acts["zombie1"]
            if isinstance(pred, str):
                pred = env.mobs["zombie1"].action_from_name[pred]
                acts["zombie1"] = pred

            actions.append(pred)
            out = env.step(acts)

            done = out[2]["__all__"] or out[3]["__all__"]

    # states, actions = make_classification(n_samples=1000, n_features=16,
    #                        n_informative=8, n_redundant=0,
    #                        n_classes=len(actions),
    #                        random_state=0, shuffle=False)

    print("Here!")
    fn = [obs_name for obs_name in obs["zombie1"]]
    cn = [i for i in range(env.mobs["zombie1"].action_space().n)]

    tr = fit_tree(states, actions, cn)

    d = tree_to_dict(tr, cn, fn)

    path = f"data/models/{module_name}"
    os.makedirs(path, exist_ok=True)
    with open(f"{path}/zombie1.pkl", "wb") as f:
        pickle.dump(d, f)

    build_inference_module(module_name, "zombie1")

    build_ai_module(module_name, list(env.mobs.values()))


    # fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(4, 4), dpi=800)
    # tree.plot_tree(tr,
    #                feature_names=fn,
    #                class_names=cn,
    #                filled=True)
    # fig.savefig('rf_individualtree.png')
