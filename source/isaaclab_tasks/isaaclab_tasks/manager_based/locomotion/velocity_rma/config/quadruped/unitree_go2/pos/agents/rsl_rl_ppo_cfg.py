# Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import (
    RslRlOnPolicyRunnerCfg,  
    RslRlPpoActorCriticCfg,
    RslRlPpoCNNActorCriticCfg,
    RslRlPpoAlgorithmCfg, 
    RslRlDistillationStudentTeacherCfg,
    RslRlDistillationStudentTeacherRecurrentCfg,
    RslRlDistillationCNNStudentTeacherCfg,
    RslRlDistillationAlgorithmCfg
)


@configclass
class Go2RoughPPORunnerCfg_Teacher(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 16
    max_iterations = 100_000_000
    save_interval = 500  
    experiment_name = "go2_velocity_rma_rough"
    empirical_normalization = False
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_hidden_dims=[128, 128, 128], 
        critic_hidden_dims=[128, 128, 128],  
        activation="elu"
    )
    # policy = RslRlPpoCNNActorCriticCfg(
    #     class_name='CNN1d_ActorCritic',
    #     init_noise_std=1.0,
    #     actor_hidden_dims=[128, 128, 128], 
    #     critic_hidden_dims=[128, 128, 128],  
    #     activation="elu",
    #     cnn_kernel_size=3,
    #     cnn_stride=3,
    #     cnn_filters=[32, 16, 8],
    #     paddings=[0, 1, 1]
    # )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.1,
        entropy_coef=0.0025,
        num_learning_epochs=5,
        num_mini_batches=8,
        learning_rate=1.0e-4,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=0.5,
    )


@configclass
class Go2FlatPPORunnerCfg_Teacher(Go2RoughPPORunnerCfg_Teacher):
    def __post_init__(self):
        super().__post_init__()
    
        self.max_iterations = 100_000_000
        self.save_interval = 500
        self.experiment_name = "go2_velocity_rma_flat"


@configclass
class Go2RoughPPORunnerCfg_Policy(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 16
    max_iterations = 100_000_000
    save_interval = 500  
    experiment_name = "go2_velocity_rma_rough"
    empirical_normalization = False
    policy = RslRlDistillationStudentTeacherCfg(
        init_noise_std=0.1,
        student_hidden_dims=[128, 128, 128], 
        teacher_hidden_dims=[128, 128, 128], 
        activation="elu"
    )
    # policy = RslRlDistillationStudentTeacherRecurrentCfg(
    #     init_noise_std=0.1,
    #     student_hidden_dims=[128, 128], 
    #     teacher_hidden_dims=[128, 128, 128], 
    #     activation="elu",
    #     rnn_type="lstm",
    #     rnn_hidden_dim=128,
    #     rnn_num_layers=1,
    #     teacher_recurrent=False
    # )
    # policy = RslRlDistillationCNNStudentTeacherCfg(
    #     class_name="CNN1d_StudentTeacher",
    #     # class_name="CNN1d_o1_StudentTeacher",
    #     init_noise_std=0.1,
    #     student_hidden_dims=[128, 128, 128], 
    #     teacher_hidden_dims=[128, 128, 128], 
    #     activation="elu",
    #     student_cnn_kernel_sizes=[3, 5, 5, 5, 5],
    #     student_cnn_strides=[3, 2, 2, 2, 2],
    #     student_cnn_filters=[128, 64, 32, 16, 8],
    #     student_cnn_paddings=[0, 2, 2, 2, 2],
    #     student_cnn_dilations=[1, 1, 1, 1, 1]
    # )
    # policy = RslRlDistillationCNNStudentTeacherCfg(
    #     class_name="CNN2d_StudentTeacher",
    #     init_noise_std=0.1,
    #     student_hidden_dims=[128, 128, 128], 
    #     teacher_hidden_dims=[128, 128, 128], 
    #     activation="elu",
    #     student_cnn_kernel_sizes=[(5, 3), (5, 5), (5, 5), (5, 1)],
    #     student_cnn_strides=[(2, 3), (2, 2), (2, 2), (2, 1)],
    #     student_cnn_filters=[128, 64, 32, 16],
    #     student_cnn_paddings=[(2, 0), (2, 2), (2, 2), (2, 0)],
    #     student_cnn_dilations=[(1, 1), (1, 1), (1, 1), (1, 1)]
    # )
    algorithm = RslRlDistillationAlgorithmCfg(
        num_learning_epochs=10,
        learning_rate=1e-4,
        gradient_length=1
    )


@configclass
class Go2FlatPPORunnerCfg_Policy(Go2RoughPPORunnerCfg_Policy):
    def __post_init__(self):
        super().__post_init__()
    
        self.max_iterations = 100_000_000
        self.save_interval = 500
        self.experiment_name = "go2_velocity_rma_flat"
