from isaaclab.utils import configclass

from isaaclab_assets.robots.ayg import AYG_CFG
from isaaclab.assets import ArticulationCfg
from pace_sim2real.utils import PaceDCMotorCfg
from pace_sim2real import PaceSim2realEnvCfg, PaceSim2realSceneCfg, PaceCfg
import torch


AYGDRIVE_PACE_ACTUATOR_CFG = PaceDCMotorCfg(
    joint_names_expr=[".*HAA", ".*HFE", ".*KFE"],
    saturation_effort=60.0,
    effort_limit=20.0,
    velocity_limit=10.0,
    stiffness={".*": 40.0},  # P gain in Nm/rad
    damping={".*": 1.0},  # D gain in Nm s/rad
    encoder_bias=[0.0] * 12,  # encoder bias in radians
    max_delay=10,  # max delay in simulation steps
)


@configclass
class AygPaceCfg(PaceCfg):
    """Pace configuration for Ayg robot."""
    robot_name: str = "ayg"
    data_dir: str = "ayg/chirp_data.pt"  # located in pace_sim2real/data/ayg/chirp_data.pt
    bounds_params: torch.Tensor = torch.zeros((49, 2))  # 12 + 12 + 12 + 12 + 1 = 49 parameters to optimize
    # IsaacLab determines joint ordering via breadth-first traversal, which may
    # differ from the ordering used in your real robot’s control stack or logged
    # data. To ensure correct alignment between simulated and real trajectories,
    # explicitly define the joint order used on your physical system here.
    joint_order: list[str] = [
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
    
    joint_limits = {
        'lower': torch.tensor([
            -0.50, -0.60, -0.70,
            -0.50, -0.60, -0.70,
            -0.50, -0.90, -1.10,
            -0.50, -0.90, -1.10,
        ]),
        'upper': torch.tensor([
            0.40, 1.20, 1.10,
            0.40, 1.20, 1.10,
            0.80, 0.90, 0.70,
            0.80, 0.90, 0.70,
        ]),
    }

    def __post_init__(self):
        # set bounds for parameters
        self.bounds_params[:12, 0] = 1e-5
        self.bounds_params[:12, 1] = 1.0  # armature between 1e-5 - 1.0 [kgm2]
        self.bounds_params[12:24, 1] = 7.0  # dof_damping between 0.0 - 7.0 [Nm s/rad]
        self.bounds_params[24:36, 1] = 0.5  # friction between 0.0 - 0.5
        self.bounds_params[36:48, 0] = -0.1
        self.bounds_params[36:48, 1] = 0.1  # bias between -0.1 - 0.1 [rad]
        self.bounds_params[48, 1] = 10.0  # delay between 0.0 - 10.0 [sim steps]


@configclass
class AygPaceSceneCfg(PaceSim2realSceneCfg):
    """Configuration for Ayg robot in Pace Sim2Real environment."""
    # Here, you inherit from PaceSim2realSceneCfg and specify your robot USD,
    # initial pose, and associated actuators. Ensure the initial height prevents
    # ground penetration or unwanted contacts. Actuator naming is flexible and
    # purely user-defined.
    robot: ArticulationCfg = AYG_CFG.replace(
        spawn=AYG_CFG.spawn.replace(
            usd_path="./ayg_isaac_lab/source/isaaclab_assets/data/Robots/ayg/ayg.usd",
        ),
        prim_path="{ENV_REGEX_NS}/Robot",
        init_state=ArticulationCfg.InitialStateCfg(pos=(0.0, 0.0, 1.0)),
        actuators={"legs": AYGDRIVE_PACE_ACTUATOR_CFG},
    )


@configclass
class AygPaceEnvCfg(PaceSim2realEnvCfg):

    scene: AygPaceSceneCfg = AygPaceSceneCfg()
    sim2real: PaceCfg = AygPaceCfg()

    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # robot sim and control settings
        self.sim.dt = 1.0/400  # 400Hz simulation
        self.decimation = 1  # 400Hz control
