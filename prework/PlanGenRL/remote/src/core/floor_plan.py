import math
import random
import time
from typing import Literal, Union

import numpy as np
from PIL import Image, ImageDraw

from utils.color_utils import generate_random_colors
from utils.img_utils import remap_img, generate_empty_img


class FloorPlan:
    GRID_UNIT_SIZE = 1.00  # 单位m
    GRID_AREA = GRID_UNIT_SIZE ** 2  # 单元格面积
    GRID_NUM_X = 40
    GRID_NUM_Y = 32
    NUM_LAYERS = 3  # 观察空间深度/层数
    VALID_LAYER = 0  # 第1层为是否可以填充
    TYPE_LAYER = 1  # 第2层为房间类型层，0为保留的未定义类型，房间类型序号从1开始
    IDX_LAYER = 2  # 第3层为序号层， 0 为保留的未定义序号，房间序号从1开始

    # drawing
    PIXEL_PER_GRID = 10
    COLOR_SEED = 1
    TYPE0_COLOR = (200, 200, 200)  # valid 且 type = 0 的颜色，默认为灰色，表示没有赋值
    WHITE = np.array([255, 255, 255])
    EMPTY_IMG = generate_empty_img(GRID_NUM_X * PIXEL_PER_GRID, GRID_NUM_Y * PIXEL_PER_GRID)
    @staticmethod
    def get_empty_mask() -> np.ndarray:
        """获取空蒙版"""
        return np.zeros(shape=(FloorPlan.GRID_NUM_Y, FloorPlan.GRID_NUM_X), dtype=bool)

    @staticmethod
    def set_mask_value(mask, x_range, y_range, value):
        """设置mask中某一范围的值"""
        mask[y_range[0]:y_range[1], x_range[0]: x_range[1]] = value

    def __init__(self, plan_mask: np.ndarray):
        self.num_x = FloorPlan.GRID_NUM_X
        self.num_y = FloorPlan.GRID_NUM_Y
        self.grid = np.zeros(shape=(self.num_y, self.num_x, FloorPlan.NUM_LAYERS), dtype=np.uint8)  # unit8决定了最多支持256种房间类型，最多256个房间
        self.grid_coord = np.stack(np.meshgrid(np.arange(self.num_x), np.arange(self.num_y)), axis=-1)  # (num_y, num_x, 2)其中最后一维为x和y坐标
        self.plan_mask = plan_mask
        self.plan_area = self.get_area_by_mask(self.plan_mask)
        self._set_valid_by_mask(self.plan_mask, True)

    def _set_valid_by_mask(self, mask: np.ndarray, value: bool) -> None:
        self.grid[:, :, FloorPlan.VALID_LAYER][mask] = 255 if value else 0

    def _set_type(self, x_range: tuple[int, int], y_range: tuple[int, int], value: int) -> None:
        self.grid[y_range[0]:y_range[1], x_range[0]: x_range[1], FloorPlan.TYPE_LAYER] = value

    def _set_type_by_mask(self, mask: np.ndarray, value: int) -> None:
        self.grid[mask] = value

    def _set_idx(self, x_range: tuple[int, int], y_range: tuple[int, int], value: int) -> None:
        self.grid[y_range[0]:y_range[1], x_range[0]: x_range[1], FloorPlan.IDX_LAYER] = value

    def _set_idx_by_mask(self, mask: np.ndarray, value: int) -> None:
        self.grid[mask] = value

    def create_room(self, x_range: tuple[int, int], y_range: tuple[int, int], room_type: int, room_idx: int):
        """创建房间"""
        if x_range[0] < 0 or x_range[1] > self.num_x:
            x_range = (max(0, x_range[0]), min(x_range[1], self.num_x))

        if y_range[0] < 0 or y_range[1] > self.num_y:
            y_range = (max(0, y_range[0]), min(y_range[1], self.num_y))

        # 计算重叠面积
        org_room_mask = self.grid[:, :, FloorPlan.TYPE_LAYER] != 0

        new_room_mask = np.zeros(shape=(self.num_y, self.num_x), dtype=bool)
        new_room_mask[y_range[0]:y_range[1], x_range[0]: x_range[1]] = True
        overlay_mask = org_room_mask & new_room_mask  # 计算原有的房间区域和新建的房间区域的重叠mask
        overlay_area = np.sum(overlay_mask) * FloorPlan.GRID_AREA  # 计算重叠区域面积
        invalid_mask = ~self.plan_mask & new_room_mask
        invalid_area = np.sum(invalid_mask) * FloorPlan.GRID_AREA  # 计算超出边界的面积

        self._set_type(x_range, y_range, room_type)  # 这里的操作会覆盖原有的信息
        self._set_idx(x_range, y_range, room_idx)

        return x_range, y_range, overlay_area, invalid_area

    def get_valid(self, x, y) -> bool:
        return bool(self.grid[y, x, FloorPlan.VALID_LAYER])

    def get_type(self, x, y):
        return self.grid[y, x, FloorPlan.TYPE_LAYER]

    def get_idx(self, x, y):
        return self.grid[y, x, FloorPlan.IDX_LAYER]

    # region 整体平面层级
    def get_total_room_area(self):
        """房间总面积"""
        return np.sum(self.grid[:, :, FloorPlan.TYPE_LAYER] != 0) * FloorPlan.GRID_AREA

    def get_invalid_room_area(self):
        # 获取超出有效区域的room的面积，即无效区域但是有room type的
        non_plan_mask = self.grid[:, :, FloorPlan.VALID_LAYER] == 0
        room_mask = self.grid[:, :, FloorPlan.TYPE_LAYER] != 0
        return np.sum(non_plan_mask & room_mask) * FloorPlan.GRID_AREA

    def get_plan_area(self):
        """建筑平面总面积"""
        return np.sum(self.grid[:, :, FloorPlan.VALID_LAYER] != 0) * FloorPlan.GRID_AREA

    # endregion

    # region 房间类型层级
    def get_all_types(self):
        """获取所有被创建的类型"""
        unique_types = np.unique(self.grid[:, :, FloorPlan.TYPE_LAYER])
        return unique_types.tolist()

    def get_mask_by_type(self, room_type: int):
        """从int类型的room_type获取mask"""
        return self.grid[:, :, FloorPlan.TYPE_LAYER] == room_type

    # endregion

    # region 房间个体层级
    def get_all_idx_of_type(self, room_type: int):
        """获取某一类房间类型的所有idx"""
        type_mask = self.grid[:, :, FloorPlan.TYPE_LAYER] == room_type
        unique_idx = np.unique(self.grid[:, :, FloorPlan.IDX_LAYER][type_mask]).tolist()
        return unique_idx

    def get_room_mask_by_type_and_idx(self, room_type: int, room_idx: int):
        """请不要针对每个type和每个index循环调用这个方法"""
        type_mask = self.grid[:, :, FloorPlan.TYPE_LAYER] == room_type
        idx_mask = self.grid[:, :, FloorPlan.IDX_LAYER] == room_idx
        room_mask = type_mask & idx_mask
        return room_mask

    # endregion
    # region 通用
    @staticmethod
    def get_area_by_mask(mask):
        return np.sum(mask) * FloorPlan.GRID_AREA

    @staticmethod
    def is_mask_continuous(mask):
        n, m = mask.shape
        visited = np.zeros_like(mask, dtype=bool)

        # Find a starting point (a 'True' in the mask)
        start_point = None
        for i in range(n):
            for j in range(m):
                if mask[i, j]:
                    start_point = (i, j)
                    break
            if start_point:
                break

        if not start_point:
            # No 'True' in the mask
            return False

        # Use DFS to visit all connected 'True' cells
        stack = [start_point]
        visited[start_point] = True
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # Up, Down, Left, Right

        while stack:
            x, y = stack.pop()
            for dx, dy in directions:
                nx, ny = x + dx, y + dy
                if 0 <= nx < n and 0 <= ny < m and mask[nx, ny] and not visited[nx, ny]:
                    stack.append((nx, ny))
                    visited[nx, ny] = True

        # Check if all 'True' cells are visited
        return np.all(mask == visited)

    @staticmethod
    def get_mask_bbox(mask):
        # 找到所有True元素的行和列的索引
        rows, cols = np.where(mask)

        if len(rows) == 0 or len(cols) == 0:
            return None  # 如果没有True元素，返回None

        min_row, min_col = np.min(rows), np.min(cols)
        max_row, max_col = np.max(rows), np.max(cols)

        return (min_row, min_col), (max_row, max_col)

    def draw(self) -> np.ndarray:
        # create canvas
        canvas = Image.new('RGB', (self.num_x * FloorPlan.PIXEL_PER_GRID, self.num_y * FloorPlan.PIXEL_PER_GRID), 'white')
        draw = ImageDraw.Draw(canvas)

        # get color for type
        max_type = np.max(self.grid[:, :, FloorPlan.TYPE_LAYER])
        colors_arr = np.array([FloorPlan.TYPE0_COLOR] + generate_random_colors(max_type, seed=FloorPlan.COLOR_SEED))
        grid_colors = colors_arr[self.grid[:, :, FloorPlan.TYPE_LAYER]]  # (num_y, num_x, 3)

        # get grid coord
        top_left = self.grid_coord * FloorPlan.PIXEL_PER_GRID  # (num_y, num_x, 2) in pixel
        bot_right = (self.grid_coord + 1) * FloorPlan.PIXEL_PER_GRID  # (num_y, num_x, 2) in pixel
        xyxys = np.concatenate((top_left, bot_right), axis=-1)  # (num_y, num_x, 4)
        for y in range(self.num_y):
            for x in range(self.num_x):
                xyxy = xyxys[y, x].tolist()
                valid = self.get_valid(x, y)
                room_type = self.get_type(x, y)
                if valid:
                    if room_type:
                        # valid and room type
                        draw.rectangle(xyxy, outline='gray', fill=tuple(grid_colors[y, x]))
                    else:
                        # valid and no room type
                        draw.rectangle(xyxy, outline='gray', fill=FloorPlan.TYPE0_COLOR)
                else:
                    if room_type:
                        # not valid and room type
                        draw.rectangle(xyxy, outline='gray', fill=tuple(np.array(0.5 * grid_colors[y, x] + 0.5 * FloorPlan.WHITE, dtype=np.uint8)))
                    else:
                        # not valid and no room type
                        draw.rectangle(xyxy, outline='gray', fill=None)

        # 在Jupyter Notebook中显示图像
        return np.array(canvas)

    def get_raw_data_img(self, channel: Union[Literal['r', 'g', 'b', 'rgb'], int] = 'rgb') -> np.ndarray:
        """channel可以是str类型的r、g、b、rgb， 也可以是int类型的FloorPlan.VALID_LAYER、FloorPlan.TYPE_LAYER、FloorPlan.IDX_LAYER"""
        if isinstance(channel, str):
            if channel == 'r':
                arr = self.grid[:, :, 0]
            elif channel == 'g':
                arr = self.grid[:, :, 1]
            elif channel == 'b':
                arr = self.grid[:, :, 2]
            else:
                arr = self.grid
        else:
            arr = self.grid[:, :, channel]

        scaled_arr = remap_img(arr, 0, np.max(arr), 0, 255)

        img = Image.fromarray(scaled_arr)
        enlarged_img = img.resize((img.width * FloorPlan.PIXEL_PER_GRID, img.height * FloorPlan.PIXEL_PER_GRID), Image.NEAREST)
        return np.array(enlarged_img)

    def get_info(self):
        info = {}
        info['plan_area'] = self.plan_area
        info['total_room_area'] = self.get_total_room_area()
        info['invalid_room_area'] = self.get_invalid_room_area()
        info['room_types'] = {}
        room_types = self.get_all_types()
        for room_type in room_types:
            info['room_types'][room_type] = {}
            room_type_info = info['room_types'][room_type]
            type_mask = self.get_mask_by_type(room_type)
            type_area = self.get_area_by_mask(type_mask)
            room_type_info['total_area'] = type_area
            room_type_info['rooms'] = {}
            room_idxs = self.get_all_idx_of_type(room_type)
            for i, room_idx in enumerate(room_idxs):
                idx_mask = self.grid[:, :, FloorPlan.IDX_LAYER] == room_idx
                room_mask = type_mask & idx_mask
                area = self.get_area_by_mask(room_mask)
                bbox = self.get_mask_bbox(room_mask)
                vec = (np.array(bbox[1]) + 1) - np.array(bbox[0])
                bbox_area = vec[0] * vec[1] * FloorPlan.GRID_AREA

                info['room_types'][room_type]['rooms'][i] = {}
                room_info = info['room_types'][room_type]['rooms'][i]
                room_info['idx'] = room_idx
                room_info['area'] = area
                room_info['continuous'] = self.is_mask_continuous(room_mask)
                room_info['bbox_area'] = bbox_area
                room_info['fill_ratio'] = area / bbox_area
                # room_info['mask'] = room_mask
        return info


class FloorPlanFactory:
    @staticmethod
    def generate_random_floor_plan(seed=None, min_ratio=0.3, max_ratio=0.8) -> FloorPlan:
        assert 0 < min_ratio <= 1, 'min_ratio should be between 0 and 1'
        assert 0 < max_ratio <= 1, 'max_ratio should be between 0 and 1'
        assert min_ratio <= max_ratio, 'min_ratio less than max ratio'
        if seed is None or seed == -1:
            seed = int(time.time() * 1000)  # Use current time in milliseconds as seed
        random.seed(seed)
        w = random.randrange(int(FloorPlan.GRID_NUM_X * min_ratio), int(math.ceil(FloorPlan.GRID_NUM_X * max_ratio)))
        h = random.randrange(int(FloorPlan.GRID_NUM_Y * min_ratio), int(math.ceil(FloorPlan.GRID_NUM_Y * max_ratio)))
        w = max(w, 1)
        h = max(h, 1)
        x = random.randrange(0, FloorPlan.GRID_NUM_X - w)
        y = random.randrange(0, FloorPlan.GRID_NUM_Y - h)

        plan_mask = FloorPlan.get_empty_mask()
        FloorPlan.set_mask_value(plan_mask, (x, x + w), (y, y + h), True)
        fp = FloorPlan(plan_mask)
        return fp

    @staticmethod
    def generate_random_rooms(fp, seed):
        random.seed(seed)
        fp.create_room((0, 10), (0, 12), room_type=1, room_idx=1)

        fp.create_room((0, 6), (12, 15), room_type=1, room_idx=2)
        fp.create_room((0, 8), (15, 20), room_type=1, room_idx=2)

        fp.create_room((8, 20), (0, 10), room_type=2, room_idx=3)
        fp.create_room((21, 30), (0, 5), room_type=2, room_idx=3)

        fp.create_room((12, 22), (8, 18), room_type=3, room_idx=4)
        fp.create_room((25, FloorPlan.GRID_NUM_X + 5),(FloorPlan.GRID_NUM_Y - 5, FloorPlan.GRID_NUM_Y + 5), room_type=3, room_idx=5)