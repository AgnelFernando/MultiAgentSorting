import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg
from isaacsim.storage.native import get_assets_root_path


UR10_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=get_assets_root_path() + "/Isaac/Samples/Rigging/Manipulator/configure_manipulator/ur10e/ur/ur_gripper.usd",
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=5.0,
        ),
        collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True, contact_offset=0.005, rest_offset=0.0),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        joint_pos={
            "shoulder_pan_joint": 0.0,
            "shoulder_lift_joint": -1.309,
            "elbow_joint": 2.094,
            "wrist_1_joint": -2.1817,
            "wrist_2_joint": -1.3963,
            "wrist_3_joint": 0.0,
            "finger_joint": 0.0,
            "left_inner_finger_joint": 0.0,
            "right_inner_finger_joint": 0.0,    
            "left_inner_knuckle_joint": 0.0,
            "right_inner_knuckle_joint": 0.0,
            "right_outer_knuckle_joint": 0.0,
        },
    ),
    actuators={
        "arm": ImplicitActuatorCfg(
            joint_names_expr=[
                "shoulder_pan_joint",
                "shoulder_lift_joint",
                "elbow_joint",
                "wrist_1_joint",
                "wrist_2_joint",
                "wrist_3_joint",
            ],
            velocity_limit=100.0,
            effort_limit=87.0,
            stiffness=800.0,
            damping=40.0,
        ),
        "gripper": ImplicitActuatorCfg(
            joint_names_expr=
            ["finger_joint",
            "left_inner_finger_joint",
            "right_inner_finger_joint",    
            "left_inner_knuckle_joint",
            "right_inner_knuckle_joint",
            "right_outer_knuckle_joint"],
            velocity_limit=20.0,
            effort_limit=30.0,
            stiffness=500.0,
            damping=20.0,
        ),
    },
)

