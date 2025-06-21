
import wandb
from polaris.experience import EpisodeCallbacks

from sacred import Experiment, Ingredient
from minecraft_ai.env import MiniMinecraft
from ml_collections import ConfigDict

exp_name = 'player_ditto'
exp_path = "experiments/" + exp_name
ex = Experiment(exp_name)

@ex.config
def cfg():

    env = MiniMinecraft.env_id
    env_config = dict(
        layout="arena_test",
        tick_skip=1,
        render=False,
    )
    num_workers = 5
    policy_path = 'polaris.policies.PPO'
    model_path = 'minecraft_ai.models.basic'
    policy_class = 'PPO'
    model_class = 'Basic'
    trajectory_length = 256
    max_seq_len = 128
    train_batch_size = 8192*4
    max_queue_size = train_batch_size * 10
    n_epochs=3
    minibatch_size= train_batch_size//16

    default_policy_config = {
        'discount': 0.993,
        'gae_lambda': 0.95,
        'entropy_cost': 5e-3,
        'lr': 5e-4,

        # PPO
        'grad_clip': 1.,
        'ppo_clip': 0.2, # 0.3
        'initial_kl_coeff': 1.,
        'baseline_coeff': 1.,
        'vf_clip': 2.,
        'kl_target': 1e-2,

        'mlp_dims': [128, 128],
        }

    policy_params = [{
        "name": "agent",
        "config": default_policy_config
    }]

    compute_advantages_on_workers = True
    wandb_logdir = 'logs'
    report_freq = 5
    episode_metrics_smoothing = 0.95
    training_metrics_smoothing = 0.8

    # FSP
    update_policy_history_freq = 50
    policy_history_length = 10

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
    from minecraft_ai.fsp import FSP

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