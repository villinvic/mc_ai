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


class SampleDefenders(MatchMaking):

    def __init__(self, agent_ids, attackers, defenders, random_defender_chance):
        super().__init__(agent_ids=agent_ids)

        self.attackers = attackers
        self.defenders = defenders
        self.random_defender_chance = random_defender_chance

    def next(
            self,
            params_map: Dict[str, "PolicyParams"],
            wid,
            num_workers,
            **kwargs,
    ) -> Dict[str, "PolicyParams"]:

        r = {
            aid: params_map[pid]
            for aid, pid in self.attackers.items()
        }

        sampled_defenders = []

        for aid, candidates in self.defenders.items():
            if np.random.random() < self.random_defender_chance:
                r[aid] = PolicyParams(policy_type="random")
            else:
                r[aid] = params_map[np.random.choice(candidates)]

        return r


def load_policy(env, checkpoint_path: str, aid, policy_name: str, config):
    if policy_name == "random":
        return RandomPolicy(env.action_space[aid], config)

    full_path = f"{checkpoint_path}/policy_params/{policy_name}.pkl"

    with open(checkpoint_path, "rb") as f:
        policy_param: PolicyParams = pickle.load(f)

    return PPO(
        name=policy_param.name,
        action_space=env.action_space[aid],  # todo
        observation_space=env.observation_space[aid],
        config=config,
        policy_config=policy_param.config,
        options=policy_param.options,
        is_online=True,
    )



class DatasetBuilder(Checkpointable):
    def __init__(
            self,
            config: ConfigDict,
    ):

        self.config = config
        self.worker_set = SyncWorkerSet(
            config,
            with_spectator=False,
        )

        # Init environment
        self.env = PolarisEnv.make(self.config.env, env_index=-1, **self.config.env_config)

        policy_params = [
            PolicyParams(**ConfigDict(pi_params)) for pi_params in self.config.policy_params
        ]

        self.PolicyCls = getattr(importlib.import_module(self.config.policy_path), self.config.policy_class)

        self.policy_map: Dict[str, Policy] = {}
        self.params_map = ParamsMap()

        self.matchmaking = SampleDefenders(
            agent_ids=self.env.get_agent_ids(),
            attackers=config.attacking_policies,
            defenders=config.defending_policies,
            random_defender_chance=self.config.random_defender_chance
        )

        self.running_jobs = []

        super().__init__(
            checkpoint_config = config.checkpoint_config,
            components={
                "params_map": self.params_map,
            }
        )

        self.restore()
        self.config = config

        self.dataset = {
            aid: PolicyDataset(size=self.config.dataset_size, aid=aid, num_actions=self.env.action_space[aid].n)
            for aid in self.config.attacking_policies
        }

    def is_done(
            self,
    ) -> bool:
        return all(dataset.is_full() for dataset in self.dataset.values())

    def collect_samples(self):
        """
        Executes one iteration of the trainer.
        :return: Training iteration results
        """

        experience_jobs = [self.matchmaking.next(self.params_map, wid, len(self.worker_set.workers)) for wid in self.worker_set.available_workers]

        self.running_jobs += self.worker_set.push_jobs(self.params_map, experience_jobs)


        experience, self.running_jobs = self.worker_set.wait(self.params_map, self.running_jobs, timeout=1e-2)

        for exp_batch in experience:
            if isinstance(exp_batch, EpisodeMetrics):
                for aid in exp_batch.custom_metrics["raw_observations"]:
                    self.dataset[aid].put(
                        actions=exp_batch.custom_metrics["actions"][aid],
                        observations=exp_batch.custom_metrics["raw_observations"][aid],
                        observation_paths=exp_batch.custom_metrics["observation_paths"][aid]
                    )


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