# Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import (
    RslRlOnPolicyRunnerCfg, 
    RslRlPpoActorCriticRecurrentCfg, 
    RslRlPpoAlgorithmCfg
)


@configclass
class Go2RoughPPORunnerCfg_rnn(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 16
    max_iterations = 100_000_000
    save_interval = 500  
    experiment_name = "go2_velocity_async_rough_rnn"
    empirical_normalization = False
    policy = RslRlPpoActorCriticRecurrentCfg(
        init_noise_std=1.0,
        actor_hidden_dims=[128, 128], 
        critic_hidden_dims=[128, 128],  
        activation="elu",
        rnn_type='gru',
        rnn_hidden_dim=128,
        rnn_num_layers=1
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.1,
        entropy_coef=0.01,
        # entropy_coef=0.0025,
        num_learning_epochs=5,
        num_mini_batches=8,
        # num_mini_batches=4,
        learning_rate=1.0e-4,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=0.5,
    )


@configclass
class Go2FlatPPORunnerCfg_rnn(Go2RoughPPORunnerCfg_rnn):
    def __post_init__(self):
        super().__post_init__()
    
        self.max_iterations = 100_000_000
        self.save_interval = 500
        self.experiment_name = "go2_velocity_async_flat_rnn"
