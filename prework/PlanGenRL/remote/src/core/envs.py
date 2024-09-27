from __future__ import annotations

import uuid
from typing import Any, SupportsFloat, Optional

import gymnasium as gym
import numpy as np
from gymnasium.core import ObsType, ActType, RenderFrame
from gymnasium.spaces import Box

from core.floor_plan import FloorPlan, FloorPlanFactory
from core.reward import get_in_bound_reward, get_area_reward, get_ratio_reward, get_invalid_reward, get_overlay_reward


class MyEnv(gym.Env):

    def __init__(self):
        super().__init__()
        self.name = f'MyEnv({uuid.uuid4()})'

        self.steps = 0
        self.observation_space = Box(0, 255, (FloorPlan.NUM_LAYERS, FloorPlan.GRID_NUM_Y, FloorPlan.GRID_NUM_X), dtype=np.uint8)  # C, H, W
        self.action_space = Box(0.0, 1.0, (6,), dtype=np.float32)

        self.fp: Optional[FloorPlan] = None

        # 缓存信息
        self.obs = None
        self.action = None
        self.done = None
        self.reward = None
        self.truncated = None
        self.reward_info = None

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None, ) -> tuple[ObsType, dict[str, Any]]:
        self.fp = FloorPlanFactory.generate_random_floor_plan(seed)
        self.steps = 0
        self.obs = self._get_observation()
        info = {}
        return self.obs, info

    def step(self, action: ActType) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        self.action = action
        self.steps += 1

        var1 = self._action_to_room_xryrti(action)
        var2 = self.fp.create_room(*var1)

        self.obs = self._get_observation()
        self.done = self.steps >= 10
        self.reward, self.reward_info = self._get_reward(var1, var2)
        self.truncated = False
        return self.obs, self.reward, self.done, self.truncated, self.reward_info

    def _get_reward(self, var1, var2):
        _ = self
        ac_x_range, ac_y_range, room_type, room_idx = var1
        x_range, y_range, overlay_area, invalid_area = var2

        ac_area = (ac_x_range[1] - ac_x_range[0]) * (ac_y_range[1] - ac_y_range[0])
        area = (x_range[1] - x_range[0]) * (y_range[1] - y_range[0]) * FloorPlan.GRID_AREA

        in_bound_reward = get_in_bound_reward(ac_area, area) * 1  # (-1 to 0) * 1
        area_reward = get_area_reward(area, target_area=self.fp.plan_area/10, loose=1) * 1  # (-1 to 1) * 2
        ratio_reward = get_ratio_reward((x_range[1] - x_range[0]) / (y_range[1] - y_range[0])) * 1  # (-1 to 1) * 1
        overlay_reward = get_overlay_reward(overlay_area, area) * 1  # (-1 to 0 or 1) * 1
        invalid_reward = get_invalid_reward(invalid_area, area) * 1  # (-1 to 0 or 1) * 1
        step_reward = 0

        # min_reward = -6 * 10
        # good_reward = 3 * 10
        # max_reward = 5 * 10

        reward = in_bound_reward + area_reward + ratio_reward + overlay_reward + invalid_reward + step_reward
        reward_info = {'in_bound_reward': in_bound_reward,
                       'area_reward': area_reward,
                       'ratio_reward': ratio_reward,
                       'overlay_reward': overlay_reward,
                       'invalid_reward': invalid_reward,
                       'step_reward': step_reward}
        return reward, reward_info

    def _get_observation(self) -> np.ndarray:
        observation = self.fp.grid
        observation = np.transpose(observation, (2, 0, 1))  # C, H, W
        return observation

    def _action_to_room_xryrti(self, action):
        _ = self
        x = action[0] * FloorPlan.GRID_NUM_X
        y = action[1] * FloorPlan.GRID_NUM_Y
        w = action[2] * (FloorPlan.GRID_NUM_X - 1) + 2
        h = action[3] * (FloorPlan.GRID_NUM_Y - 1) + 2
        t = int(action[4] * 254) + 1
        i = int(action[5] * 254) + 1
        x_range = np.array((x - w / 2, x + w / 2), dtype=int).tolist()
        y_range = np.array((y - h / 2, y + h / 2), dtype=int).tolist()
        return x_range, y_range, t, i

    def render(self) -> RenderFrame | list[RenderFrame] | None:
        return self.fp.draw()

    def close(self):
        pass
