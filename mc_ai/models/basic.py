import types

from polaris.models import BaseModel

import sonnet as snt
import tree
from gymnasium.spaces import Discrete
import tensorflow as tf

from polaris.experience import SampleBatch

tf.compat.v1.enable_eager_execution()

from tensorflow.keras.optimizers import RMSprop
import numpy as np
from polaris.models.utils import CategoricalDistribution

class Basic(BaseModel):
    is_recurrent = False


    def __init__(
            self,
            observation_space,
            action_space: Discrete,
            config,
    ):
        super(Basic, self).__init__(
            name="Basic",
            observation_space=observation_space,
            action_space=action_space,
            config=config,
        )
        self.action_dist = CategoricalDistribution
        self.num_outputs = action_space.n

        # RMSProp, from experience, is much less sample efficient.
        self.optimiser = snt.optimizers.Adam(
            learning_rate=config.lr,
            epsilon=1e-5,
        )

        self.mlp = snt.nets.MLP(config["mlp_dims"], name="mlp", activate_final=True,
                                activation=tf.keras.activations.relu)

        self._policy_head = snt.Linear(self.action_space.n, name="policy_head")
        self._value_head = snt.Linear(1, name="value_head")

    def forward_single_action_with_extras(
            self,
            obs,
            prev_action,
            prev_reward,
            state
    ):
        final_embeddings, next_state = self.single_input(
            obs,
            prev_action,
            prev_reward,
            state
        )

        policy_logits = self._policy_head(final_embeddings)
        extras = {
            SampleBatch.VALUES: tf.squeeze(self._value_head(final_embeddings))
        }

        return policy_logits, next_state, extras

    def forward_single_action(
            self,
            obs,
            prev_action,
            prev_reward,
            state
    ):
        # faster call by skipping value inference.

        final_embeddings, next_state = self.single_input(
            obs,
            prev_action,
            prev_reward,
            state
        )

        return self._policy_head(final_embeddings), next_state

    def __call__(
            self,
            *,
            obs,
            prev_action,
            prev_reward,
            state,
            seq_lens
    ):
        final_embeddings = self.batch_input(
            obs,
            prev_action,
            prev_reward,
            state,
            seq_lens
        )
        policy_logits = self._policy_head(final_embeddings)
        self._values = tf.squeeze(self._value_head(final_embeddings))
        return policy_logits, self._values

    def single_input(
            self,
            obs,
            prev_action,
            prev_reward,
            state
    ):

        return self.mlp(obs), state

    def batch_input(
            self,
            obs,
            prev_action,
            prev_reward,
            state,
            seq_lens
    ):
        return self.mlp(obs)

    def get_initial_state(self):
        return np.zeros((1,1), dtype=np.float32)

    def critic_loss(self, targets):
        # Categorical value function loss
        return tf.math.square(tf.squeeze(targets) - tf.squeeze(self._values))


class BasicDeterministic(Basic):
    is_recurrent = False

    def __init__(
            self,
            observation_space,
            action_space: Discrete,
            config,
    ):
        super(Basic, self).__init__(
            name="Basic",
            observation_space=observation_space,
            action_space=action_space,
            config=config,
        )

        self.action_dist = CategoricalDistribution
        self.num_outputs = action_space.n

        # RMSProp, from experience, is much less sample efficient.
        self.optimiser = snt.optimizers.Adam(
            learning_rate=config.lr,
            epsilon=1e-5,
        )

        self.mlp = snt.nets.MLP(config["mlp_dims"], name="mlp", activate_final=True,
                                activation=tf.keras.activations.relu)

        self._policy_head = snt.Linear(self.action_space.n, name="policy_head")
        self._value_head = snt.Linear(1, name="value_head")

    def forward_single_action_with_extras(
            self,
            obs,
            prev_action,
            prev_reward,
            state
    ):
        final_embeddings, next_state = self.single_input(
            obs,
            prev_action,
            prev_reward,
            state
        )

        policy_logits = self._policy_head(final_embeddings) * 20.

        extras = {
            SampleBatch.VALUES: tf.squeeze(self._value_head(final_embeddings))
        }

        return policy_logits, next_state, extras

    def forward_single_action(
            self,
            obs,
            prev_action,
            prev_reward,
            state
    ):
        # faster call by skipping value inference.

        final_embeddings, next_state = self.single_input(
            obs,
            prev_action,
            prev_reward,
            state
        )

        policy_logits = self._policy_head(final_embeddings) * 20.

        return policy_logits, next_state

    def __call__(
            self,
            *,
            obs,
            prev_action,
            prev_reward,
            state,
            seq_lens
    ):
        final_embeddings = self.batch_input(
            obs,
            prev_action,
            prev_reward,
            state,
            seq_lens
        )
        policy_logits = self._policy_head(final_embeddings) * 20.
        self._values = tf.squeeze(self._value_head(final_embeddings))
        return policy_logits, self._values

    def single_input(
            self,
            obs,
            prev_action,
            prev_reward,
            state
    ):
        return self.mlp(obs), state

    def batch_input(
            self,
            obs,
            prev_action,
            prev_reward,
            state,
            seq_lens
    ):
        return self.mlp(obs)

    def get_initial_state(self):
        return np.zeros((1, 1), dtype=np.float32)

    def critic_loss(self, targets):
        # Categorical value function loss
        return tf.math.square(tf.squeeze(targets) - tf.squeeze(self._values))

