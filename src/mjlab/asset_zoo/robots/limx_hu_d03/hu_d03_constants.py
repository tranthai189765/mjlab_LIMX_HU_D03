"""LimX HU_D03 constants.

Ported from the LimX `humanoid-description` repo (HU_D03_03 variant): 31 actuated
revolute joints — legs (hip pitch/roll/yaw, knee, ankle pitch/roll x2 = 12),
waist (yaw/roll/pitch = 3), head (yaw/pitch = 2), arms (shoulder pitch/roll/yaw,
elbow, wrist yaw/pitch, hand_yaw x2 = 14).

DATA SOURCES (no values are guessed unless noted):
  * Mass / inertia / kinematics / joint position limits: HU_D03_03 URDF.
  * Joint torque (effort) limits: URDF <limit effort> (continuous/nameplate).
  * Standing pose: LimX `humanoid-rl-deploy-python/controllers/HU_D03_03/
    stand_controller/joint_params.yaml` (stand_pos).
  * PD gains (stiffness=kp, damping=kd): LimX `.../mimic_controller/mimic_param.yaml`
    (control.kp / control.kd) — the DYNAMIC-MOTION gains. NOTE: do NOT use the
    stand_controller gains (kp up to 800) — those are a stiff standing hold and
    are far too stiff for an RL walking task. With mjlab's
    `action_scale = 0.25*effort/stiffness`, the stiff stand gains give the policy
    ~10x less range of motion than G1, and the robot cannot step (it just stands).
    The mimic gains are the correct locomotion regime.
  * `armature` is the only ESTIMATE — LimX has not published rotor inertia.
"""

from pathlib import Path

import mujoco

from mjlab import MJLAB_SRC_PATH
from mjlab.actuator import BuiltinPositionActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.utils.spec_config import CollisionCfg

##
# MJCF and assets.
##

HU_D03_XML: Path = (
  MJLAB_SRC_PATH / "asset_zoo" / "robots" / "limx_hu_d03" / "xmls" / "hu_d03.xml"
)
assert HU_D03_XML.exists()


def get_spec() -> mujoco.MjSpec:
  return mujoco.MjSpec.from_file(str(HU_D03_XML))


##
# Actuator config — LimX mimic (dynamic-motion) gains; effort from URDF.
##

HU_D03_ACTUATOR_HIP_PR = BuiltinPositionActuatorCfg(
  target_names_expr=(".*_hip_pitch_joint", ".*_hip_roll_joint"),
  stiffness=280.0,
  damping=5.0,
  effort_limit=120.0,
  armature=0.01,
)
HU_D03_ACTUATOR_HIP_YAW = BuiltinPositionActuatorCfg(
  target_names_expr=(".*_hip_yaw_joint",),
  stiffness=100.0,
  damping=5.0,
  effort_limit=120.0,
  armature=0.01,
)
HU_D03_ACTUATOR_KNEE = BuiltinPositionActuatorCfg(
  target_names_expr=(".*_knee_joint",),
  stiffness=280.0,
  damping=4.0,
  effort_limit=120.0,
  armature=0.01,
)
HU_D03_ACTUATOR_ANKLE = BuiltinPositionActuatorCfg(
  target_names_expr=(".*_ankle_pitch_joint", ".*_ankle_roll_joint"),
  stiffness=20.0,
  damping=2.0,
  effort_limit=45.0,
  armature=0.01,
)
HU_D03_ACTUATOR_WAIST = BuiltinPositionActuatorCfg(
  target_names_expr=("waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint"),
  stiffness=80.0,
  damping=3.0,
  effort_limit=45.0,
  armature=0.01,
)
HU_D03_ACTUATOR_HEAD = BuiltinPositionActuatorCfg(
  target_names_expr=("head_yaw_joint", "head_pitch_joint"),
  stiffness=5.0,
  damping=0.5,
  effort_limit=18.0,
  armature=0.003,
)
HU_D03_ACTUATOR_ARM = BuiltinPositionActuatorCfg(
  target_names_expr=(
    ".*_shoulder_pitch_joint",
    ".*_shoulder_roll_joint",
    ".*_shoulder_yaw_joint",
    ".*_elbow_joint",
  ),
  stiffness=80.0,
  damping=4.0,
  effort_limit=30.0,
  armature=0.005,
)
HU_D03_ACTUATOR_WRIST = BuiltinPositionActuatorCfg(
  target_names_expr=(
    ".*_wrist_yaw_joint",
    ".*_wrist_pitch_joint",
    ".*_hand_yaw_joint",
  ),
  stiffness=40.0,
  damping=3.0,
  effort_limit=18.0,
  armature=0.003,
)

##
# Keyframe config.
#
# Official LimX HU_D03_03 stand pose (stand_controller stand_pos). Base height
# (0.911 m) measured by forward kinematics so the feet rest on the ground.
##

HOME_KEYFRAME = EntityCfg.InitialStateCfg(
  pos=(0.0, 0.0, 0.911),
  joint_pos={
    # Legs.
    "left_hip_pitch_joint": -0.15,
    "left_hip_roll_joint": 0.0,
    "left_hip_yaw_joint": -0.05,
    "left_knee_joint": 0.30,
    "left_ankle_pitch_joint": -0.16,
    "left_ankle_roll_joint": 0.0,
    "right_hip_pitch_joint": -0.15,
    "right_hip_roll_joint": 0.0,
    "right_hip_yaw_joint": 0.05,
    "right_knee_joint": 0.30,
    "right_ankle_pitch_joint": -0.16,
    "right_ankle_roll_joint": 0.0,
    # Waist + head.
    "waist_yaw_joint": 0.0,
    "waist_roll_joint": 0.0,
    "waist_pitch_joint": 0.0,
    "head_yaw_joint": 0.0,
    "head_pitch_joint": 0.0,
    # Left arm.
    "left_shoulder_pitch_joint": 0.1,
    "left_shoulder_roll_joint": 0.1,
    "left_shoulder_yaw_joint": -0.2,
    "left_elbow_joint": -0.2,
    "left_wrist_yaw_joint": 0.0,
    "left_wrist_pitch_joint": 0.0,
    "left_hand_yaw_joint": 0.0,
    # Right arm.
    "right_shoulder_pitch_joint": 0.1,
    "right_shoulder_roll_joint": -0.1,
    "right_shoulder_yaw_joint": 0.2,
    "right_elbow_joint": -0.2,
    "right_wrist_yaw_joint": 0.0,
    "right_wrist_pitch_joint": 0.0,
    "right_hand_yaw_joint": 0.0,
  },
  joint_vel={".*": 0.0},
)

##
# Collision config (feet-only; other geoms disabled via disable_other_geoms).
##

FEET_ONLY_COLLISION = CollisionCfg(
  geom_names_expr=(r"^(left|right)_foot[1-7]_collision$",),
  contype=0,
  conaffinity=1,
  condim=3,
  priority=1,
  friction=(0.6,),
)

##
# Final config.
##

HU_D03_ARTICULATION = EntityArticulationInfoCfg(
  actuators=(
    HU_D03_ACTUATOR_HIP_PR,
    HU_D03_ACTUATOR_HIP_YAW,
    HU_D03_ACTUATOR_KNEE,
    HU_D03_ACTUATOR_ANKLE,
    HU_D03_ACTUATOR_WAIST,
    HU_D03_ACTUATOR_HEAD,
    HU_D03_ACTUATOR_ARM,
    HU_D03_ACTUATOR_WRIST,
  ),
  soft_joint_pos_limit_factor=0.9,
)


def get_hu_d03_robot_cfg() -> EntityCfg:
  """Get a fresh HU_D03 robot configuration instance."""
  return EntityCfg(
    init_state=HOME_KEYFRAME,
    collisions=(FEET_ONLY_COLLISION,),
    spec_fn=get_spec,
    articulation=HU_D03_ARTICULATION,
  )


# Per-joint action scale, mjlab convention: 0.25 * effort_limit / stiffness
# (same formula G1 uses). With the mimic gains this yields G1-comparable leg
# authority (~0.1-0.56), enough range of motion to walk.
HU_D03_ACTION_SCALE: dict[str, float] = {}
for _a in HU_D03_ARTICULATION.actuators:
  assert isinstance(_a, BuiltinPositionActuatorCfg)
  assert _a.effort_limit is not None
  _scale = 0.25 * _a.effort_limit / _a.stiffness
  for _n in _a.target_names_expr:
    HU_D03_ACTION_SCALE[_n] = _scale


if __name__ == "__main__":
  import mujoco.viewer as viewer

  from mjlab.entity.entity import Entity

  robot = Entity(get_hu_d03_robot_cfg())
  viewer.launch(robot.spec.compile())
