import importlib
import os
from typing import NamedTuple, List, Any, Tuple, Dict
import numpy as np
import pickle

from ml_collections import ConfigDict
from polaris.checkpointing.checkpointable import Checkpointable
from polaris.environments import PolarisEnv
from polaris.experience import SampleBatch, MatchMaking
from polaris.experience.episode import EpisodeMetrics
from polaris.experience.worker_set import SyncWorkerSet
from polaris.policies import PolicyParams, RandomPolicy
from polaris.policies.PPO import PPO
from polaris.policies.policy import ParamsMap, Policy
from polaris.utils import MetricBank
from tqdm import tqdm
from mcfunction_lib.inference import PolicyDataset

class DatasetBuilder(Checkpointable):
    def __init__(
            self,
            config: ConfigDict,
    ):

        self.config = config

        # Init environment
        self.env = PolarisEnv.make(self.config.env, env_index=-1, **self.config.env_config)

        policy_params = [
            PolicyParams(**ConfigDict(pi_params)) for pi_params in self.config.policy_params
        ]

        self.PolicyCls = getattr(importlib.import_module(self.config.policy_path), self.config.policy_class)

        self.policy_map: Dict[str, Policy] = {}
        self.params_map = ParamsMap()

        self.running_jobs = []

        super().__init__(
            checkpoint_config = config.checkpoint_config,
            components={
                "params_map": self.params_map,
            }
        )

        self.restore()
        self.config = config

        for policy_name, params in self.params_map.items():
            self.policy_map[policy_name] = self.PolicyCls(
                name=policy_name,
                action_space=self.env.action_space[params.options["aid"]],
                observation_space=self.env.observation_space[params.options["aid"]],
                config=self.config,
                policy_config=params.config,
                options=params.options,
                is_online=True,
            )
            self.policy_map[policy_name].setup(params)

        self.dataset = {
            aid: PolicyDataset(size=self.config.dataset_size, aid=aid, num_actions=self.env.action_space[aid].n)
            for aid in self.config.attacking_policies
        }

    def is_done(
            self,
    ) -> bool:
        return all(dataset.is_full() for dataset in self.dataset.values())

    def collect_samples(self):

        for aid, dataset, in self.dataset.items():
            obs, raw = self.env.sample_random_obs(aid, size=(self.config.batch_size,1))
            flat_raw = np.stack([r[:, 0] for r in raw.values()], axis=1)
            flat_obs = np.concatenate(obs, axis=1, dtype=np.float32)
            actions, _ = self.policy_map[self.config.attacking_policies[aid]].compute_single_action(
                obs=flat_obs,
                prev_action=None,
                prev_reward=None,
                state=None
            )
            print(flat_obs.shape)
            print(np.unique(actions))
            dataset.put(actions, flat_raw, observation_paths=raw.keys())


    def save(self):
        os.makedirs(self.config.dataset_path, exist_ok=True)
        for aid, dataset in self.dataset.items():
            dataset.save(self.config.dataset_path + f"/{aid}")

    def run(self):
        try:
            total = sum(dataset.size for dataset in self.dataset.values())
            with tqdm(total=total, desc="Collecting Samples", unit="samples") as pbar:
                previous_total = 0

                while not self.is_done():
                    self.collect_samples()
                    # Compute new total collected samples
                    current_total = sum(dataset.current_size() for dataset in self.dataset.values())
                    pbar.update(current_total - previous_total)
                    previous_total = current_total
            self.save()
        except KeyboardInterrupt:
            print("Caught C^.")