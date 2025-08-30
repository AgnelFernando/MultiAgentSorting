# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import copy
import math
import random
from typing import List, Tuple
from source.MultiAgentSorting.MultiAgentSorting.utils.utils import sample_positions_2d, sample_points
import torch
from collections.abc import Sequence

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.envs import DirectMARLEnv
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils.math import sample_uniform
from isaaclab.assets import RigidObject
from isaaclab.sensors import TiledCamera

from .multiagentsorting_marl_env_cfg import MultiagentsortingMarlEnvCfg


class MultiagentsortingMarlEnv(DirectMARLEnv):
    cfg: MultiagentsortingMarlEnvCfg

    def __init__(self, cfg: MultiagentsortingMarlEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        self.r1_cmd_pos = self.robot1.data.joint_pos[:, :6].clone()
        self.r2_cmd_pos = self.robot1.data.joint_pos[:, :6].clone()

        self.r1_actions = torch.zeros((self.num_envs, self.cfg.action_spaces["robot1"]), device=self.device) 
        self.r1_prev_actions = torch.zeros_like(self.r1_actions) 
        
        self.r2_actions = torch.zeros((self.num_envs, self.cfg.action_spaces["robot2"]), device=self.device) 
        self.r2_prev_actions = torch.zeros_like(self.r2_actions)

        self.robot_dof_lower_limits = self.robot1.data.soft_joint_pos_limits[0, :6, 0].to(device=self.device)
        self.robot_dof_upper_limits = self.robot1.data.soft_joint_pos_limits[0, :6, 1].to(device=self.device)

        self.arm_joints_idx = self.robot1.find_joints("^(shoulder|elbow|wrist).*")[0]


    def _setup_scene(self):
        # add ground plane
        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())

        self.table = RigidObject(self.cfg.table_cfg)
        self.basket1 = RigidObject(self.cfg.basket1_cfg)
        self.basket2 = RigidObject(self.cfg.basket2_cfg)

        self.robot1 = Articulation(self.cfg.robot1_cfg)
        self.robot2 = Articulation(self.cfg.robot2_cfg)

        self._camera = TiledCamera(self.cfg.tiled_camera)

        self._create_blocks()
        
        # clone and replicate
        self.scene.clone_environments(copy_from_source=False)
        # we need to explicitly filter collisions for CPU simulation
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[])
        # add articulation to scene
        self.scene.articulations["robot1"] = self.robot1
        self.scene.articulations["robot2"] = self.robot2

        self.scene.rigid_objects["table"] = self.table
        self.scene.rigid_objects["basket1"] = self.basket1
        self.scene.rigid_objects["basket2"] = self.basket2  

        for i, block in enumerate(self.red_blocks):
            self.scene.rigid_objects[f"red_block_{i}"] = block  
        
        for i, block in enumerate(self.blue_blocks):
            self.scene.rigid_objects[f"blue_block_{i}"] = block

        self.scene.sensors["camera"] = self._camera

        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _pre_physics_step(self, actions: dict[str, torch.Tensor]) -> None:
        self.actions = actions

    def _apply_action(self) -> None:
        self.r1_cmd_pos = self.r1_cmd_pos + self.actions['robot1']
        self.r1_cmd_pos = torch.clamp(self.r1_cmd_pos, self.robot_dof_lower_limits, self.robot_dof_upper_limits)
        # self.robot1.set_joint_position_target(self.r1_cmd_pos, joint_ids=self.arm_joints_idx)
        
        self.r2_cmd_pos = self.r2_cmd_pos + self.actions['robot2']
        self.r2_cmd_pos = torch.clamp(self.r2_cmd_pos, self.robot_dof_lower_limits, self.robot_dof_upper_limits)
        # self.robot2.set_joint_position_target(self.r2_cmd_pos, joint_ids=self.arm_joints_idx)

    def _get_observations(self) -> dict[str, torch.Tensor]:
        camera_data = self._camera.data.output["rgb"] / 255.0
        mean_tensor = torch.mean(camera_data, dim=(1, 2), keepdim=True)
        camera_data -= mean_tensor
        observations = {
            "robot1": torch.cat([self.robot1.data.joint_pos[:, :6], self.robot1.data.joint_vel[:, :6]], dim=-1),
            "robot2": torch.cat([self.robot2.data.joint_pos[:, :6], self.robot2.data.joint_vel[:, :6]], dim=-1),
            "central": camera_data.view(camera_data.size(0), -1).contiguous()
        }
        return observations

    def _get_rewards(self) -> dict[str, torch.Tensor]:
        zeros = torch.zeros(self.num_envs, device=self.device)
        return {
            "robot1": zeros.clone(),
            "robot2": zeros.clone(),
            "central": zeros.clone(),
        }

    def _get_dones(self) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        time_outs = {"robot1": time_out, "robot2": time_out, "central": time_out}
        terminated = {"robot1": torch.zeros_like(time_out, dtype=torch.bool),
                      "robot2": torch.zeros_like(time_out, dtype=torch.bool),
                      "central": torch.zeros_like(time_out, dtype=torch.bool)}      
        
        return terminated, time_outs

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = self.robot1._ALL_INDICES

        if not isinstance(env_ids, torch.Tensor):
            env_ids = torch.as_tensor(env_ids, dtype=torch.long, device=self.device)
        else:
            env_ids = env_ids.to(dtype=torch.long, device=self.device)

        n = env_ids.numel()
        super()._reset_idx(env_ids)

        # robot state
        self.robot1.reset(env_ids)
        self.robot2.reset(env_ids)   

        r1_joint_pos = self.robot1.data.default_joint_pos[env_ids, :6] + sample_uniform(
            -0.125, 0.125, (n, 6), self.device,
        )
        r1_joint_pos = torch.clamp(r1_joint_pos, self.robot_dof_lower_limits, self.robot_dof_upper_limits)
        r2_joint_pos = self.robot2.data.default_joint_pos[env_ids, :6] + sample_uniform(
            -0.125, 0.125, (n, 6), self.device,
        )
        r2_joint_pos = torch.clamp(r2_joint_pos, self.robot_dof_lower_limits, self.robot_dof_upper_limits)

        joint_vel = torch.zeros_like(r1_joint_pos)
        self.robot1.write_joint_state_to_sim(r1_joint_pos, joint_vel, env_ids=env_ids, joint_ids=self.arm_joints_idx)
        self.robot2.write_joint_state_to_sim(r2_joint_pos, joint_vel, env_ids=env_ids, joint_ids=self.arm_joints_idx)

        self.r1_cmd_pos[env_ids] = r1_joint_pos
        if self.r1_actions is not None:
            self.r1_actions[env_ids] = 0.0
            self.r1_prev_actions[env_ids] = 0.0

        self.r2_cmd_pos[env_ids] = r2_joint_pos
        if self.r2_actions is not None:
            self.r2_actions[env_ids] = 0.0
            self.r2_prev_actions[env_ids] = 0.0

        # set the actuator targets equal to the (reset) command to hold still
        self.robot1.set_joint_position_target(self.r1_cmd_pos[env_ids], env_ids=env_ids, joint_ids=self.arm_joints_idx)
        self.robot2.set_joint_position_target(self.r2_cmd_pos[env_ids], env_ids=env_ids, joint_ids=self.arm_joints_idx)

        # block state 
        self._reposition_blocks(env_ids)
        
    
    def _reposition_blocks(self, env_ids: Sequence[int]):
        if not isinstance(env_ids, torch.Tensor):
            env_ids = torch.as_tensor(env_ids, dtype=torch.long, device=self.device)
        else:
            env_ids = env_ids.to(dtype=torch.long, device=self.device)

        n_env = env_ids.numel()
        n_blocks = len(self.red_blocks)  # blocks per env
        z = float(self.cfg.table_cfg.spawn.scale[2] + 0.03)

        min_sep = 0.0

        # (env, block, xyz)
        red_pos  = torch.empty((n_env, n_blocks, 3), device=self.device, dtype=torch.float32)
        blue_pos = torch.empty((n_env, n_blocks, 3), device=self.device, dtype=torch.float32)

        # Sample per env
        for e in range(n_env):
            r_pts = sample_points(self.cfg.robot_workspace["robot1"],
                                    n=n_blocks, min_separation=min_sep)
        
            for i, (x, y) in enumerate(r_pts):
                red_pos[e, i, 0] = x; red_pos[e, i, 1] = y; red_pos[e, i, 2] = z

            b_pts = sample_points(self.cfg.robot_workspace["robot2"],
                                  n=n_blocks, min_separation=min_sep)
            for i, (x, y) in enumerate(b_pts):
                blue_pos[e, i, 0] = x; blue_pos[e, i, 1] = y; blue_pos[e, i, 2] = z

        # Add env origins → world frame
        env_origins = self.scene.env_origins[env_ids].to(self.device, dtype=torch.float32)  # (env,3)
        red_pos  += env_origins.unsqueeze(1)  # (env,1,3) + (env,block,3)
        blue_pos += env_origins.unsqueeze(1)

        # Build (env, block, 7) root poses: [xyz, qw,qx,qy,qz]
        root_red  = torch.empty((n_env, n_blocks, 7), device=self.device, dtype=torch.float32)
        root_blue = torch.empty((n_env, n_blocks, 7), device=self.device, dtype=torch.float32)

        root_red[:, :, :3]  = red_pos
        root_blue[:, :, :3] = blue_pos

        # Identity quaternion (w,x,y,z) = (1,0,0,0)
        root_red[:, :, 3]  = 1.0; root_red[:, :, 4:]  = 0.0
        root_blue[:, :, 3] = 1.0; root_blue[:, :, 4:] = 0.0

        # Optional debug (small prints only)
        # print(f"env_ids {env_ids.tolist()} red block positions:\n{red_pos.detach().cpu().numpy()}")

        # Write poses block-by-block with batched env_ids
        for i in range(n_blocks):
            self.red_blocks[i].write_root_pose_to_sim(root_red[:, i, :].contiguous(),  env_ids=env_ids)
            self.blue_blocks[i].write_root_pose_to_sim(root_blue[:, i, :].contiguous(), env_ids=env_ids)



    def _create_blocks(self):
        self.red_blocks = []
        self.blue_blocks = []

        n = self.cfg.num_blocks_per_robot
        z = self.cfg.table_cfg.spawn.scale[2] + 0.03  # spawn height once

        # Tuneables in one place
        min_sep = 0.0

        # Sample positions (handles reversed bounds automatically)
        red_pts = sample_points(self.cfg.robot_workspace["robot1"],
            n=n, min_separation=min_sep)

        blue_pts = sample_points(self.cfg.robot_workspace["robot2"],
            n=n, min_separation=min_sep)


        for i, (x, y) in enumerate(red_pts):
            cfg_copy = copy.deepcopy(self.cfg.red_block_cfg)
            cfg_copy.init_state.pos = [x, y, z]
            cfg_copy.prim_path += f"{i}"
            self.red_blocks.append(RigidObject(cfg_copy))

        for i, (x, y) in enumerate(blue_pts):
            cfg_copy = copy.deepcopy(self.cfg.blue_block_cfg)
            cfg_copy.prim_path += f"{i}"
            cfg_copy.init_state.pos = [x, y, z]
            self.blue_blocks.append(RigidObject(cfg_copy))
