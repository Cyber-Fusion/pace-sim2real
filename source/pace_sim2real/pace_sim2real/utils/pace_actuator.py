# © 2025 ETH Zurich, Robotic Systems Lab
# Author: Filip Bjelonic
# Licensed under the Apache License 2.0

from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

from isaaclab.actuators import DCMotor
from isaaclab.utils.types import ArticulationActions
from isaaclab.utils import DelayBuffer
if TYPE_CHECKING:
    # only for type checking
    from .pace_actuator_cfg import PaceDCMotorCfg


class PaceDCMotor(DCMotor):
    """Pace DC Motor actuator model with encoder bias and per-joint action delay.

    The actuator models a DC motor whose controller receives joint positions in the encoder
    frame by adding a per-joint encoder bias to the true joint positions. In other words,
    the controller operates on biased (encoder) positions rather than the true joint positions.

    The torque command computed by the PD controller is applied after a configurable
    delay per joint (in simulation steps) to represent per-motor latency between command
    calculation and actuation.

    The software implementation is inspired by DelayedPDActuator.
    """

    cfg: PaceDCMotorCfg

    def __init__(self, cfg: PaceDCMotorCfg, *args, **kwargs):
        super().__init__(cfg, *args, **kwargs)
        if isinstance(cfg.encoder_bias, (list, tuple)):
            if len(cfg.encoder_bias) != self.num_joints:
                raise ValueError(
                    f"encoder_bias must have {self.num_joints} elements (one per joint), "
                    f"but got {len(cfg.encoder_bias)}: {cfg.encoder_bias}"
                )
        self.encoder_bias = self._parse_joint_parameter(cfg.encoder_bias, 0.0)

        # one delay buffer per joint so each motor can have independent latency
        self.torques_delay_buffers = [
            DelayBuffer(cfg.max_delay + 1, self._num_envs, device=self._device)
            for _ in range(self.num_joints)
        ]
        init_env_ids = torch.arange(self._num_envs, device=self._device)
        for buf in self.torques_delay_buffers:
            buf.set_time_lag(cfg.max_delay, init_env_ids)

    def reset(self, env_ids: Sequence[int]):
        super().reset(env_ids)
        # reset buffers
        for buf in self.torques_delay_buffers:
            buf.reset(env_ids)

    def update_encoder_bias(self, encoder_bias: torch.Tensor):
        self.encoder_bias = encoder_bias

    def update_time_lags(self, delay: int | torch.Tensor, env_ids: Sequence[int] | None = None):
        """Set per-joint time lags.

        Accepts:
            - int scalar: same lag for every (env, joint)
            - 1D tensor of shape [num_envs]: same lag across joints, varies per env
            - 2D tensor of shape [num_envs, num_joints]: per-(env, joint) lag
        """
        if env_ids is None:
            env_ids = torch.arange(self._num_envs, device=self._device)
        if isinstance(delay, int) or (torch.is_tensor(delay) and delay.ndim < 2):
            for buf in self.torques_delay_buffers:
                buf.set_time_lag(delay, env_ids)
        else:
            for j, buf in enumerate(self.torques_delay_buffers):
                buf.set_time_lag(delay[:, j], env_ids)

    def compute(
        self, control_action: ArticulationActions, joint_pos: torch.Tensor, joint_vel: torch.Tensor
    ) -> ArticulationActions:
        # compute actuator model with encoder bias added to joint positions (joint position in encoder frame, not simulation frame)
        control_action_sim = super().compute(control_action, joint_pos - self.encoder_bias, joint_vel)
        efforts = control_action_sim.joint_efforts  # [num_envs, num_joints]
        # CircularBuffer requires 3D storage [history, batch, feature]; keep a trailing dim of 1 per joint
        delayed = torch.cat(
            [self.torques_delay_buffers[j].compute(efforts[:, j:j+1]) for j in range(self.num_joints)],
            dim=1,
        )
        control_action_sim.joint_efforts = delayed
        return control_action_sim
