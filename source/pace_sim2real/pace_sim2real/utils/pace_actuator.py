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
    """The Pace DC Motor actuator model with encoder bias and action delay.

    We model the joint position that the PD actuator gets by adding an encoder bias term to the true joint position.
    Essentially, the actuator commands in the encoder frame and not the true joint frame.

    Unfortunately, the delayed PD actuator (DelayedPDActuator class) is inheriting from IdealPDActuator and not DCMotor.
    Thus, we need to re-implement the delay buffer logic here again. Sorry for the duplicate code.
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
        self.encoder_bias = torch.tensor(cfg.encoder_bias, device=self._device).unsqueeze(0).repeat(self._num_envs, 1)

        self.positions_delay_buffer = DelayBuffer(cfg.max_delay + 1, self._num_envs, device=self._device)
        self.velocities_delay_buffer = DelayBuffer(cfg.max_delay + 1, self._num_envs, device=self._device)
        self.efforts_delay_buffer = DelayBuffer(cfg.max_delay + 1, self._num_envs, device=self._device)

        self.positions_delay_buffer.set_time_lag(cfg.max_delay, torch.arange(self._num_envs, device=self._device))
        self.velocities_delay_buffer.set_time_lag(cfg.max_delay, torch.arange(self._num_envs, device=self._device))
        self.efforts_delay_buffer.set_time_lag(cfg.max_delay, torch.arange(self._num_envs, device=self._device))

    def reset(self, env_ids: Sequence[int]):
        super().reset(env_ids)
        # reset buffers
        self.positions_delay_buffer.reset(env_ids)
        self.velocities_delay_buffer.reset(env_ids)
        self.efforts_delay_buffer.reset(env_ids)

    def update_encoder_bias(self, encoder_bias: torch.Tensor):
        self.encoder_bias = encoder_bias

    def update_time_lags(self, delay: int | torch.Tensor, env_ids: Sequence[int] | None = None):
        if env_ids is None:
            env_ids = torch.arange(self._num_envs, device=self._device)
        self.positions_delay_buffer.set_time_lag(delay, env_ids)
        self.velocities_delay_buffer.set_time_lag(delay, env_ids)
        self.efforts_delay_buffer.set_time_lag(delay, env_ids)

    def compute(
        self, control_action: ArticulationActions, joint_pos: torch.Tensor, joint_vel: torch.Tensor
    ) -> ArticulationActions:
        control_action.joint_positions = self.positions_delay_buffer.compute(control_action.joint_positions)
        control_action.joint_velocities = self.velocities_delay_buffer.compute(control_action.joint_velocities)
        control_action.joint_efforts = self.efforts_delay_buffer.compute(control_action.joint_efforts)
        # compute actuator model with encoder bias added to joint positions (joint position in encoder frame)
        return super().compute(control_action, joint_pos + self.encoder_bias, joint_vel)
