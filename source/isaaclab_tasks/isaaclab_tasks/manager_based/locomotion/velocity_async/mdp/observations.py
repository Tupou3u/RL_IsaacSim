# Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.actuators import DelayedPDActuatorCfg
from isaaclab.sensors import ContactSensor, Imu
import isaaclab.utils.math as math_utils

GRAVITY_VEC = torch.tensor([[0.0, 0.0, -1.0]])

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv, ManagerBasedRLEnv


def joint_pos_rel_without_wheel(
    env: ManagerBasedEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    wheel_asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """The joint positions of the asset w.r.t. the default joint positions.(Without the wheel joints)"""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    joint_pos_rel = asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    joint_pos_rel[:, wheel_asset_cfg.joint_ids] = 0
    return joint_pos_rel


def phase(env: ManagerBasedRLEnv, cycle_time: float) -> torch.Tensor:
    if not hasattr(env, "episode_length_buf") or env.episode_length_buf is None:
        env.episode_length_buf = torch.zeros(env.num_envs, device=env.device, dtype=torch.long)
    phase = env.episode_length_buf[:, None] * env.step_dt / cycle_time
    phase_tensor = torch.cat([torch.sin(2 * torch.pi * phase), torch.cos(2 * torch.pi * phase)], dim=-1)
    return phase_tensor


def joint_eff(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    return asset.data.applied_torque


def masses(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    return asset.root_physx_view.get_masses()[:, asset_cfg.body_ids].clone().to(env.device)


def inertias(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    inertias = asset.root_physx_view.get_inertias()[:, asset_cfg.body_ids].clone()
    return inertias.reshape(inertias.shape[0], inertias.shape[1] * inertias.shape[2]).to(env.device)


def dof_stiffnesses(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    return asset.root_physx_view.get_dof_stiffnesses().clone().to(env.device)


def dof_dampings(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    return asset.root_physx_view.get_dof_dampings().clone().to(env.device)


def material_props(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    props = asset.root_physx_view.get_material_properties()[:, asset_cfg.body_ids].clone()
    return props.reshape(props.shape[0], props.shape[1] * props.shape[2]).to(env.device)


def coms(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    coms = asset.root_physx_view.get_coms()[:, asset_cfg.body_ids].clone()
    return coms.reshape(coms.shape[0], coms.shape[1] * coms.shape[2]).to(env.device)


def delays(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    actuator = asset.actuators["legs"]
    if isinstance(actuator, DelayedPDActuatorCfg):
        return torch.cat(
            (actuator.positions_delay_buffer._time_lags, actuator.velocities_delay_buffer._time_lags, actuator.efforts_delay_buffer._time_lags),
            dim=1
        ).to(env.device)
    return torch.zeros(env.num_envs, 3).to(env.device)

def contact_forces_obs(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    return contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids].reshape(env.num_envs, -1)


def projected_gravity_imu(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("imu")) -> torch.Tensor:
    asset: Imu = env.scene[asset_cfg.name]
    quat = asset.data.quat_w
    return math_utils.quat_rotate_inverse(quat, GRAVITY_VEC.repeat(env.num_envs, 1).to(env.device))