from isaaclab.utils import configclass

from isaaclab_assets.robots.anymal import ANYMAL_D_CFG
from isaaclab.assets import ArticulationCfg
from pace_sim2real.utils.pace_actuator_cfg import PaceDCMotorCfg
from pace_sim2real.tasks.manager_based.pace_sim2real.pace_sim2real_env_cfg import PaceSim2realEnvCfg, PaceSim2realSceneCfg, PaceCfg

import torch

ANYDRIVE_PACE_ACTUATOR_CFG = PaceDCMotorCfg(
    joint_names_expr=[".*HAA", ".*HFE", ".*KFE"],
    saturation_effort=140.0,
    effort_limit=89.0,
    velocity_limit=8.5,
    stiffness={".*": 85.0},  # P gain in Nm/rad
    damping={".*": 0.6},  # D gain in Nm s/rad
    encoder_bias=[0.0] * 12,  # encoder bias in radians
    max_delay=10,  # max delay in simulation steps
)

REAL_ROBOT_JOINTS = [
    "LF_HAA",
    "LF_HFE",
    "LF_KFE",
    "RF_HAA",
    "RF_HFE",
    "RF_KFE",
    "LH_HAA",
    "LH_HFE",
    "LH_KFE",
    "RH_HAA",
    "RH_HFE",
    "RH_KFE",
]

bounds_params = torch.zeros((49, 2))  # 12 + 12 + 12 + 12 + 1 = 49
bounds_params[:12, 0] = 1e-5
bounds_params[:12, 1] = 1.0  # armature between 0.0 - 1.0 [kgm2]
bounds_params[12:24, 1] = 7.0  # dof_damping between 0.0 - 7.0 [Nm s/rad]
bounds_params[24:36, 1] = 0.5  # friction between 0.0 - 0.5
bounds_params[36:48, 0] = -0.1
bounds_params[36:48, 1] = 0.1  # bias between -0.5 - 0.5 [rad]
bounds_params[48, 1] = 10.0  # delay between 0.0 - 10.0 [sim steps]


@configclass
class ANYmalDPaceSceneCfg(PaceSim2realSceneCfg):
    """Configuration for Anymal-D robot in Pace Sim2Real environment."""
    robot: ArticulationCfg = ANYMAL_D_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot", init_state=ArticulationCfg.InitialStateCfg(pos=(0.0, 0.0, 1.0)),
                                                  actuators={"legs": ANYDRIVE_PACE_ACTUATOR_CFG})


@configclass
class AnymalDPaceEnvCfg(PaceSim2realEnvCfg):

    scene: ANYmalDPaceSceneCfg = ANYmalDPaceSceneCfg()
    sim2real: PaceCfg = PaceCfg(robot_name="anymal_d_sim",
                                joint_order=REAL_ROBOT_JOINTS,
                                bounds_params=bounds_params,
                                data_dir="anymal_d_sim/chirp_data.pt")  # located in pace_sim2real/data/anymal_d_sim/chirp_data.pt

    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # robot sim and control settings
        self.sim.dt = 0.0025  # 400Hz simulation
        self.decimation = 1  # 400Hz control
