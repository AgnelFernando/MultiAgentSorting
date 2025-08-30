# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


from isaaclab.assets import ArticulationCfg, RigidObjectCfg
from isaaclab.envs import DirectMARLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass
from isaaclab.sensors import TiledCameraCfg
import isaaclab.sim as sim_utils
from source.MultiAgentSorting.MultiAgentSorting.robots.universal_robots import UR10_CFG
from isaacsim.storage.native import get_assets_root_path


@configclass
class MultiagentsortingMarlEnvCfg(DirectMARLEnvCfg):
    # env
    decimation = 2
    episode_length_s = 5.0

    tiled_camera: TiledCameraCfg = TiledCameraCfg(
        prim_path="/World/envs/env_.*/Camera",
        offset=TiledCameraCfg.OffsetCfg(pos=(-5.0, 0.0, 2.0), rot=(1.0, 0.0, 0.0, 0.0), convention="world"),
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.1, 20.0)
        ),
        width=100,
        height=100,
    )

    # multi-agent specification and spaces definition
    possible_agents = ["robot1", "robot2", "central"]
    action_spaces = {"robot1": 6, "robot2": 6, "central": 1}
    observation_spaces = {"robot1": 12, "robot2": 12, "central": int(tiled_camera.height * tiled_camera.width* 3)}
    state_space = 0

    # simulation
    sim: SimulationCfg = SimulationCfg(dt=1 / 120, render_interval=decimation)

    # table and baskets
    table_cfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Table",
        spawn=sim_utils.UsdFileCfg(
            usd_path="assets/table.usd",
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=False),
            scale=(0.75, 2.0, 1.0),
        ),
    )

    basket1_cfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Basket1",
        spawn=sim_utils.UsdFileCfg(
            usd_path="assets/basket.usd",
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            scale=(0.5, 0.6, 2.0),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.69, 0.59, table_cfg.spawn.scale[2]), rot=(0.0, 0.0, 0.0, 1.0)),
    )

    basket2_cfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Basket2",
        spawn=sim_utils.UsdFileCfg(
            usd_path="assets/basket.usd",
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            scale=(0.5, 0.6, 2.0),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(-0.69, -0.41, table_cfg.spawn.scale[2]), rot=(0.0, 0.0, 0.0, 1.0)),
    )

    # robots
    robot1_cfg: ArticulationCfg = UR10_CFG.replace(prim_path="/World/envs/env_.*/Robot1")
    robot1_cfg.init_state.pos = [0.62, 0.1, table_cfg.spawn.scale[2]]
    robot1_cfg.init_state.rot = [0.0, 0.0, 0.0, 1.0]
    robot1_cfg.init_state.joint_pos["shoulder_lift_joint"] = -1.309
    robot1_cfg.init_state.joint_pos["elbow_joint"] = 2.094
    robot1_cfg.init_state.joint_pos["wrist_1_joint"] = -2.1817
    robot1_cfg.init_state.joint_pos["wrist_2_joint"] = -1.3963
    

    robot2_cfg: ArticulationCfg = UR10_CFG.replace(prim_path="/World/envs/env_.*/Robot2")
    robot2_cfg.init_state.pos = [-0.62, 0.19, table_cfg.spawn.scale[2]]
    robot2_cfg.init_state.rot = [1.0, 0.0, 0.0, 0.0]
    robot2_cfg.init_state.joint_pos["shoulder_lift_joint"] = -1.5708
    robot2_cfg.init_state.joint_pos["elbow_joint"] = 1.8326
    robot2_cfg.init_state.joint_pos["wrist_1_joint"] = -1.5708
    robot2_cfg.init_state.joint_pos["wrist_2_joint"] = -1.745
    robot2_cfg.init_state.joint_pos["wrist_3_joint"] = 0.1745

    #blocks
    red_block_cfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/RedBlock",
        spawn=sim_utils.UsdFileCfg(
            usd_path="https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/4.5/Isaac/Props/Blocks/red_block.usd",  
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=False),
            scale=(1.5, 1.5, 1.5),
        ),
    )

    blue_block_cfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/BlueBlock",
        spawn=sim_utils.UsdFileCfg(
            usd_path="https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/4.5/Isaac/Props/Blocks/blue_block.usd",  
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=False),
            scale=(1.5, 1.5, 1.5),
        ),
    )

    num_blocks_per_robot = 10
    robot_workspace = {
        "robot1": {
            "x": (-0.35, 0.25),
            "y": (-0.37, 0.27),
        },
        "robot2": {
            "x": (-0.25, 0.35),
            "y": (-0.27, 0.37),
        },
    }


    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=2, env_spacing=6.0, replicate_physics=True)

    
    