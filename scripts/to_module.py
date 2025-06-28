import os
import pickle
from collections import OrderedDict
from typing import Dict

import numpy as np
import six
import sklearn
from polaris.experience import EpisodeCallbacks

from sacred import Experiment, Ingredient
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier

from mc_ai.env import MiniMinecraft
from ml_collections import ConfigDict

from mc_ai.mc_types import MobConfig
from mcfunction_lib.build import build_ai_module
from mcfunction_lib.inference import PolicyDataset

exp_name = 'player_ditto'
exp_path = "experiments/" + exp_name
ex = Experiment(exp_name)

attackers = [
    MobConfig(
        name="zombie1",
        cls="mc_ai.mob.PlayerLike",
        config=dict(
            position_offset=(0, 0, 1)
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

@ex.config
def cfg():

    env = MiniMinecraft.env_id
    env_config = dict(
        attackers=attackers,
        defenders=defenders,
        layout="arena_test",
        tick_skip=1,
        render=False,
        record_raw_observations=True,
        # TODO: this is simpler to manipulate the dataset, rather than the observations themselves
        raw_position_offset=(0-5 , -60, 0-5) # position of the arena's center ingame minus arena dimension
    )

    module_name = "zombie1"
    dataset_path = f"data/policy_datasets/{exp_name}"
    inference_tree_depth = 19


@ex.automain
def main(_config):

    config = ConfigDict(_config)
    env = MiniMinecraft(**config["env_config"])

    build_ai_module(
            module_name=config.module_name,
            mobs=list(env.mobs.values()),
            position_offset=(0, 0, 0),
            dataset_path=config.dataset_path,
            inference_tree_depth=config.inference_tree_depth
    )






