import copy
import importlib
import os
import queue
import threading
import time
from collections import defaultdict
from copy import deepcopy
from typing import Dict

import numpy as np
import polaris.experience.matchmaking
import tree
from ml_collections import ConfigDict

from polaris.checkpointing.checkpointable import Checkpointable
from polaris.experience.episode import EpisodeMetrics, NamedPolicyMetrics
from polaris.experience.worker_set import SyncWorkerSet
from polaris.experience.matchmaking import MatchMaking
from polaris.environments.polaris_env import PolarisEnv
from polaris.policies.policy import Policy, PolicyParams, ParamsMap
from polaris.experience.sampling import ExperienceQueue, SampleBatch
from polaris.utils.metrics import MetricBank, GlobalCounter, GlobalTimer

import psutil

class FictitiousMatchmaking(MatchMaking):

    def __init__(self, agent_ids, trainable_policies):
        super().__init__(agent_ids=agent_ids)
        self.trainable_policies = trainable_policies

        self.render_freq = 100
        self.sampled = 0

    def next(
            self,
            params_map: Dict[str, "PolicyParams"],
            wid,
            num_workers,
            **kwargs,
    ) -> Dict[str, "PolicyParams"]:

        p1 = np.random.choice(self.trainable_policies)
        other = list(params_map.keys())
        if wid == 0 or self.sampled % self.render_freq == 0:
            p2 = p1
        else:
            p2 = np.random.choice(other)

        sampled_policies = [p1, p2]
        #np.random.shuffle(sampled_policies)


        r = {
            aid: params_map[pid] for pid, aid in zip(sampled_policies, self.agent_ids)
        }
        # for param in r.values():
        #     if self.sampled % self.render_freq == 0 and self.sampled > 0:
        #         param.options["render"] = True
        #     else:
        #         param.options["render"] = False

        #print(self.sampled, param.options)

        self.sampled += 1
        return r


class FSP(Checkpointable):
    def __init__(
            self,
            config: ConfigDict,
            restore=False,
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

        self.discarded_policies = []
        self.trainable_policies = []
        self.old_policies = defaultdict(list)

        self.PolicyCls = getattr(importlib.import_module(self.config.policy_path), self.config.policy_class)

        if restore:
            self.policy_map: Dict[str, Policy] = {}
            self.params_map = ParamsMap()

            self.experience_queue: Dict[str, ExperienceQueue] = {}

            self.matchmaking = FictitiousMatchmaking(
                agent_ids=self.env.get_agent_ids(),
                trainable_policies=self.trainable_policies
            )
        else:
            self.policy_map: Dict[str, Policy] = {
                policy_param.name: self.PolicyCls(
                    name=policy_param.name,
                    action_space=self.env.action_space,
                    observation_space=self.env.observation_space,
                    config=self.config,
                    policy_config=policy_param.config,
                    options=policy_param.options,
                    # For any algo that needs to track either we have the online model
                    is_online=True,
                )
                for policy_param in policy_params
            }

            self.params_map = ParamsMap(**{
                name: p.get_params() for name, p in self.policy_map.items()
            })

            self.experience_queue: Dict[str, ExperienceQueue] = {
                policy_name: ExperienceQueue(self.config)
                for policy_name in self.policy_map
            }

            self.matchmaking = FictitiousMatchmaking(
                agent_ids=self.env.get_agent_ids(),
                trainable_policies=self.trainable_policies
            )

            for policy_param in policy_params:
                self.trainable_policies.append(policy_param.name)

        self.running_experience_jobs = []

        self.metricbank = MetricBank(
            report_freq=self.config.report_freq
        )
        self.metrics = self.metricbank.metrics

        super().__init__(
            checkpoint_config = config.checkpoint_config,

            components={
                "matchmaking": self.matchmaking,
                "config": self.config,
                "params_map": self.params_map,
                "metrics": self.metrics,
                "discarded_policies": self.discarded_policies,
                "old_policies": self.old_policies,
                "trainable_policies": self.trainable_policies,
            }
        )

        if restore:
            if isinstance(restore, str):
                self.restore(restore_path=restore)
            else:
                self.restore()

            # override user config
            self.config = config

            # Need to pass the restored references afterward
            self.metricbank.metrics = self.metrics

            env_step_counter = "counters/" + GlobalCounter.ENV_STEPS
            if env_step_counter in self.metrics:
                GlobalCounter[GlobalCounter.ENV_STEPS] = self.metrics["counters/" + GlobalCounter.ENV_STEPS].get()

            for policy_name, params in self.params_map.items():

                self.policy_map[policy_name] = self.PolicyCls(
                    name=policy_name,
                    action_space=self.env.action_space,
                    observation_space=self.env.observation_space,
                    config=self.config,
                    policy_config=params.config,
                    options=params.options,
                    is_online=True,
                )
                self.policy_map[policy_name].setup(params)
                self.experience_queue[policy_name] = ExperienceQueue(self.config)

        self.startup_time = time.time()
        self.agent_frames_since_startup = 0

    def training_step(self):
        """
        Executes one iteration of the trainer.
        :return: Training iteration results
        """

        t = []
        t.append(time.time())
        GlobalTimer[GlobalTimer.PREV_ITERATION] = time.time()
        iteration_dt = GlobalTimer.dt(GlobalTimer.PREV_ITERATION)

        t.append(time.time())

        experience_jobs = [self.matchmaking.next(self.params_map, wid, len(self.worker_set.workers)) for wid in self.worker_set.available_workers]
        t.append(time.time())

        self.running_experience_jobs += self.worker_set.push_jobs(self.params_map, experience_jobs)

        experience_metrics = []
        t.append(time.time())
        frames = 0
        env_steps = 0

        experience, self.running_experience_jobs = self.worker_set.wait(self.params_map, self.running_experience_jobs, timeout=1e-2)
        if len(experience)>0:
            print("collected ", len(experience), "experiences")

        enqueue_time_start = time.time()
        num_batch = 0

        for exp_batch in experience:
            if isinstance(exp_batch, EpisodeMetrics):
                try:
                    experience_metrics.append(exp_batch)
                    env_steps += exp_batch.length
                except Exception as e:
                    print(e, exp_batch)

            else: # Experience batch
                batch_pid = exp_batch.get_owner()
                if batch_pid not in self.trainable_policies:
                    continue
                owner = self.policy_map[batch_pid]

                if (not self.experience_queue[owner.name].is_ready()) and owner.version == exp_batch[SampleBatch.VERSION][0]:
                    num_batch +=1
                    exp_batch = exp_batch.pad_sequences()
                    exp_batch[SampleBatch.SEQ_LENS] = np.array(exp_batch[SampleBatch.SEQ_LENS])
                    #print(f"rcved {owner} {exp_batch[SampleBatch.SEQ_LENS]}, version {exp_batch[SampleBatch.VERSION][0]}")
                    self.experience_queue[owner.name].push([exp_batch])
                    self.policy_map[owner.name].stats["samples_generated"] += exp_batch.size()
                    GlobalCounter.incr("batch_count")
                    frames += exp_batch.size()
                    self.agent_frames_since_startup += frames
                elif owner.version != exp_batch[SampleBatch.VERSION][0]:
                    #pass
                    # toss the batch...
                    print(owner.name, owner.version, exp_batch[SampleBatch.VERSION][0], self.params_map[owner.name].version)
                else:
                    # toss the batch...
                    pass

        if frames > 0:
            GlobalTimer[GlobalTimer.PREV_FRAMES] = time.time()
            prev_frames_dt = GlobalTimer.dt(GlobalTimer.PREV_FRAMES)
        if num_batch > 0:
            enqueue_time_ms = (time.time() - enqueue_time_start) * 1000.
        else:
            enqueue_time_ms = None

        n_experience_metrics = len(experience_metrics)
        GlobalCounter[GlobalCounter.ENV_STEPS] += env_steps

        if n_experience_metrics > 0:
            GlobalCounter[GlobalCounter.NUM_EPISODES] += n_experience_metrics


        t.append(time.time())
        training_metrics = {}
        for policy_name, policy_queue in self.experience_queue.items():
            if policy_queue.is_ready():
                pulled_batch = policy_queue.pull(self.config.train_batch_size)
                if np.any(pulled_batch[SampleBatch.VERSION] != self.policy_map[policy_name].version):
                    print(f"Had older samples in the batch for policy {policy_name} version {self.policy_map[policy_name].version}!"
                          f" {pulled_batch[SampleBatch.VERSION]}")

                train_results = self.policy_map[policy_name].train(
                    pulled_batch,
                )
                training_metrics[f"{policy_name}"] = train_results
                GlobalCounter.incr(GlobalCounter.STEP)

                params = self.policy_map[policy_name].get_params()
                self.params_map[policy_name] = params

                if params.version == 2 or params.version % self.config.update_policy_history_freq == 0:
                    pid = f"{policy_name}_version_{params.version}"
                    new_params = PolicyParams(
                        name=pid,
                        weights=params.weights,
                        config=params.config,
                        options=params.options,
                        stats=params.stats,
                        version=params.version,
                        policy_type=params.policy_type
                    )
                    self.params_map[pid] = new_params
                    self.old_policies[policy_name].append(pid)
                    print("added policy:", pid)
                    if len(self.old_policies[policy_name]) > self.config.policy_history_length:
                        removed = self.old_policies[policy_name].pop(0)
                        del self.params_map[removed]
                        self.discarded_policies.append(removed)
                        print("removed policy:", removed)

        def mean_metric_batch(b):
            return tree.flatten_with_path(tree.map_structure(
                lambda *samples: np.mean(samples),
                *b
            ))

        # Make it policy specific, thus extract metrics of policies.

        if len(training_metrics)> 0:
            for policy_name, policy_training_metrics in training_metrics.items():
                policy_training_metrics = mean_metric_batch([policy_training_metrics])
                self.metricbank.update(policy_training_metrics, prefix=f"training/{policy_name}/",
                                       smoothing=self.config.training_metrics_smoothing)
        if len(experience_metrics) > 0:
            for metrics in experience_metrics:
                self.metricbank.update(tree.flatten_with_path(metrics), prefix=f"experience/",
                                       smoothing=self.config.episode_metrics_smoothing)

        misc_metrics =  [
                    (f'{pi}_queue_length', queue.size())
                    for pi, queue in self.experience_queue.items()
                ]
        misc_metrics.append(("FPS", self.agent_frames_since_startup / (time.time()-self.startup_time)))
        if enqueue_time_ms is not None:
            misc_metrics.append(("experience_enqueue_ms", enqueue_time_ms))

        self.metricbank.update(
            misc_metrics
            , prefix="misc/", smoothing=0.
        )

        self.metricbank.update(
            self.matchmaking.metrics(),
            prefix="matchmaking/", smoothing=0.9
        )

        # We should call those only at the report freq...
        self.metricbank.update(
            tree.flatten_with_path(GlobalCounter.get()), prefix="counters/"
        )

    def run(self):
        try:
            while not self.is_done(self.metricbank):
                self.training_step()
                self.metricbank.report(print_metrics=False)
                self.checkpoint_if_needed()
        except KeyboardInterrupt:
            print("Caught C^.")





