
import wandb
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
    )
    num_workers = 5
    policy_path = 'polaris.policies.PPO'
    model_path = 'mc_ai.models.basic'
    policy_class = 'PPO'
    model_class = 'Basic' # BasicDeterministic
    trajectory_length = 256
    max_seq_len = 128
    train_batch_size = 8192
    max_queue_size = train_batch_size * 10
    n_epochs=4
    minibatch_size= train_batch_size//8

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

    policy_params = [{
        "name": "agent",
        "config": default_policy_config,
        "options": {"aid": "zombie1"}
    }]

    compute_advantages_on_workers = True
    wandb_logdir = 'logs'
    report_freq = 5
    episode_metrics_smoothing = 0.95
    training_metrics_smoothing = 0.8

    # FSP
    update_policy_history_freq = 50
    policy_history_length = 5
    history_update_end = 800

    checkpoint_config = dict(
        checkpoint_frequency=50,
        checkpoint_path=exp_path,
        stopping_condition={"environment_steps": 1e10},
        keep=4,
    )

    episode_callback_class = EpisodeCallbacks

    restore = False


@ex.automain
def main(_config):
    import tensorflow as tf
    tf.compat.v1.enable_eager_execution()
    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, False)
    from fsp import FSP

    config = ConfigDict(_config)
    MiniMinecraft(**config["env_config"]).register()


    wandb.init(
        config=_config,
        project="mc",
        mode='online',
        group="test",
        name=exp_name,
        notes=None,
        dir=config["wandb_logdir"]
    )

    trainer = FSP(config, restore=config.restore)
    trainer.run()