from polaris.experience import EpisodeCallbacks

from sacred import Experiment, Ingredient
from mc_ai.env import MiniMinecraft
from ml_collections import ConfigDict

from mc_ai.mc_types import MobConfig

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
        render=True,
        record_raw_observations=True,
        #random_raw_observations=True,
        # TODO: this is simpler to manipulate the dataset, rather than the observations themselves
        raw_position_offset=(0-5 , -60, 0-5) # position of the arena's center ingame minus arena dimension
    )
    policy_path = 'polaris.policies.PPO'
    model_path = 'mc_ai.models.basic'
    policy_class = 'PPO'
    model_class = 'BasicDeterministic'
    num_workers = 5
    trajectory_length = 256
    max_seq_len = 128
    batch_size = 8192
    compute_advantages_on_workers = True

    default_policy_config = {
        'discount': 0.99,
        'gae_lambda': 0.95,
        'entropy_cost': 0.,
        'lr': 5e-4,

        # PPO
        'grad_clip': 1.,
        'ppo_clip': 0.2, # 0.3
        'initial_kl_coeff': 1.,
        'baseline_coeff': 1.,
        'vf_clip': 2.,
        'kl_target': 1e-2,

        'mlp_dims': [32, 32],
        }

    attacking_policies = {
        "zombie1": "agent"
    }
    defending_policies = {
        "player": ["agent"]
    }

    random_defender_chance=1/3

    policy_params = [{
        "name": "agent",
        "config": default_policy_config,
        "options": {"aid": "zombie1"}
    }]

    dataset_path = f"data/policy_datasets/{exp_name}"
    dataset_size = 500_000


    # dummy stuff
    checkpoint_config = dict(
        checkpoint_frequency=50,
        checkpoint_path=exp_path,
        stopping_condition={"environment_steps": 1e10},
        keep=4,
    )
    episode_callback_class = EpisodeCallbacks


@ex.automain
def main(_config):
    import tensorflow as tf
    tf.compat.v1.enable_eager_execution()
    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, False)
    from on_policy_dataset import DatasetBuilder

    config = ConfigDict(_config)
    MiniMinecraft(**config["env_config"]).register()

    builder = DatasetBuilder(config)
    builder.run()