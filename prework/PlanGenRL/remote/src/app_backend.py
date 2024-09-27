import math
import os
import sys
import uuid
from datetime import datetime
from typing import Union, Literal, Optional

import gymnasium as gym
import pandas as pd
from gymnasium.envs.registration import register
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback, CallbackList
from stable_baselines3.common.monitor import Monitor
from streamlit.runtime.uploaded_file_manager import UploadedFile
from torchinfo import summary

from core.envs import MyEnv
from core.floor_plan import FloorPlan, FloorPlanFactory
from core.train import create_model, load_model

register(
    id='MyEnv-v0',
    entry_point='core.envs:MyEnv',
)


class Backend:
    _env = MyEnv()
    _model: Optional[PPO] = None
    _model_name = None

    _model_settings = {'create_or_load_model_op': 'Create New Model',
                       'enable_tb_log': False,
                       'model_n_steps': 2048,
                       'model_verbose': 1}
    _training_settings = {'total_timesteps': 4096,
                          'enable_eval': False,
                          'use_custom_task_name': False,
                          'save_final_model': False,
                          'eval_freq_mul': 1,
                          'task_name': None,
                          'callback_update_freq': 1024}
    _cached_model_file_id = None
    _cached_model_summary = None

    _cached_log_folder_name = None

    _train_next_frame = False
    _is_training = False
    _training_complete = False

    _training_progress_bar_obj = None
    _training_status_obj = None
    _training_info_container = None

    @classmethod
    def ST_OnResetEnvButtonPressed(cls, seed):
        cls._obs, _ = cls._env.reset(seed=seed)

    @classmethod
    def ST_OnNewEnvButtonPressed(cls):
        cls._env = MyEnv()
        cls.ST_RemoveModel()

    @classmethod
    def ST_OnGenerateRandomRoomButtonPressed(cls, seed):
        if cls._env.fp is None:
            return
        FloorPlanFactory.generate_random_rooms(cls._env.fp, seed)

    @classmethod
    def ST_SetModelSetting(cls, key, value):
        if key not in cls._model_settings:
            raise KeyError(f'Model setting {key} is not defined')
        cls._model_settings[key] = value

    @classmethod
    def ST_SetTrainingSetting(cls, key, value):
        if key not in cls._training_settings:
            raise KeyError(f'Training setting {key} is not defined')
        cls._training_settings[key] = value

    # region create new model
    @classmethod
    def ST_OnNewModelButtonPressed(cls):

        cls._model = create_model(cls._env,
                                  n_steps=cls._model_settings['model_n_steps'],
                                  verbose=cls._model_settings['model_verbose'],
                                  tensorboard_log='./logs/tensorboard/' if cls._model_settings['enable_tb_log'] else None)
        cls._model_name = f'default PPO({uuid.uuid4()})'

        cls._cached_model_file_id = None
        cls._cached_model_summary = None

    @classmethod
    def ST_LoadModelAuto(cls, file: UploadedFile, verbose=1, enable_tb_log=False):
        if file is None:
            return False
        if file.file_id == cls._cached_model_file_id:
            return False
        cls._model = load_model(file,
                                env=cls._env,
                                verbose=verbose,
                                tensorboard_log='./logs/tensorboard/' if enable_tb_log else None)
        cls._model_name = f"loaded PPO ({file.name.split('.')[0]})"

        cls._cached_model_file_id = file.file_id
        cls._cached_model_summary = None

        return True

    # endregion

    @classmethod
    def ST_HasModel(cls):
        return cls._model is not None

    @classmethod
    def ST_RemoveModel(cls):
        cls._model = None
        cls._model_name = None

        cls.ST_StopTraining()

    @classmethod
    def ST_OnTakeActionButtonPressed(cls):
        if cls._model is None:
            return
        if cls._env.fp is None:
            return

        action, _ = cls._model.predict(cls._env.obs)
        cls._env.step(action)

    @classmethod
    def ST_StartTraining(cls):
        if cls._model is None:
            return
        # 仅做标记，不训练
        cls._is_training = True
        cls._training_complete = False
        # 标记到下一次刷新时训练
        cls._train_next_frame = True

    @classmethod
    def ST_HandleTrainingAuto(cls):
        if not cls._train_next_frame:
            return False
        cls._train_next_frame = False
        if cls._model is None:
            return False
        task_name = cls._training_settings['task_name']
        task_name = None if task_name == "" else task_name

        cls._save_final_model = cls._training_settings['save_final_model']

        log_folder_name = f"{task_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}" if task_name is not None else datetime.now().strftime("%Y%m%d_%H%M%S")
        cls._cached_log_folder_name = log_folder_name

        # 创建callback
        callbacks = []
        custom_callback = CustomCallback(update_freq=cls._training_settings['callback_update_freq'], verbose=0)
        callbacks += [custom_callback]
        if cls._training_settings['enable_eval']:
            best_model_save_path = os.path.join('./logs', log_folder_name, 'models')
            log_path = os.path.join('./logs', log_folder_name, 'eval_results')
            eval_callback = EvalCallback(Monitor(gym.make("MyEnv-v0")),
                                         n_eval_episodes=5,
                                         best_model_save_path=best_model_save_path,
                                         log_path=log_path,
                                         eval_freq=cls._model.n_steps * cls._training_settings['eval_freq_mul']
                                         )
            callbacks += [eval_callback]

        callback = CallbackList(callbacks)

        # 计算真实的训练次数
        total_timesteps = int(cls._model.n_steps * math.ceil(cls._training_settings['total_timesteps'] / cls._model.n_steps))

        # 更新控件状态
        cls._training_status_obj.update(label="Training...", expanded=True, state='running')
        cls._training_status_obj.text(f"Task name: {log_folder_name}")
        # 开始训练
        cls._model.learn(total_timesteps, callback=callback, tb_log_name=log_folder_name)

        # 训练完成或提前结束
        if cls._training_settings['save_final_model']:
            final_model_path = os.path.join('./logs', cls._cached_log_folder_name, 'models', 'final_model.zip')
            cls.ST_SaveModel(final_model_path)
        # 训练完成返回True
        return True

    @classmethod
    def CB_OnProgressChanged(cls, callback: 'CustomCallback'):
        percent = min(1.0, callback.num_timesteps / callback.total_timesteps)
        readable_elapsed_time = format_timedelta(datetime.now() - callback.start_time)

        cls._training_progress_bar_obj.progress(value=percent)
        print(sys.getrefcount(cls._training_status_obj))
        cls._training_status_obj.update(label=f"Training... {int(percent * 100)}%({callback.num_timesteps}/{callback.total_timesteps}) - ({readable_elapsed_time})",expanded =True, state='running')

    @classmethod
    def ST_SaveModel(cls, path):
        cls._model.save(path)

    @classmethod
    def ST_StopTraining(cls):
        cls._training_complete = False
        cls._is_training = False

    @classmethod
    def ST_RegisterTrainingObjs(cls, status):
        print('registered')
        cls._training_status_obj = status
        cls._training_progress_bar_obj = cls._training_status_obj.progress(value=0.0)

    @classmethod
    def ST_GetEnvName(cls):
        if cls._env is None:
            return 'No Env'
        return cls._env.name

    @classmethod
    def ST_GetEnvInfo(cls):
        if cls._env is None:
            return None
        info = {'steps': str(cls._env.steps),
                'last_action': str(cls._env.action),
                'reward': str(cls._env.reward),
                'done': str(cls._env.done),
                'reward_info': str(cls._env.reward_info)
                }
        df = pd.DataFrame.from_dict(info, orient='index', columns=['value'])
        return df

    @classmethod
    def ST_GetEnvRewardInfo(cls):
        if cls._env is None:
            return None
        info = cls._env.reward_info
        if info is None:
            return
        df = pd.DataFrame.from_dict(info, orient='index', columns=['value'])
        return df

    @classmethod
    def ST_GetModelName(cls):
        if cls._model is None:
            return 'No Model'
        return cls._model_name

    @classmethod
    def ST_GetModelSummary(cls):
        if cls._model is None:
            return None
        if cls._cached_model_summary is not None:
            return cls._cached_model_summary
        shape = cls._model.policy.observation_space.shape
        cls._cached_model_summary = summary(cls._model.policy, [1] + list(shape))  # 增加一个batch层
        return cls._cached_model_summary

    @classmethod
    def ST_GetModelInfo(cls):
        if cls._model is None:
            return None
        info = {"total_timesteps": cls._model._total_timesteps,
                "num_timesteps": cls._model.num_timesteps,
                "n_steps": cls._model.n_steps,
                "device": cls._model.device,
                "tensorboard_log": cls._model.tensorboard_log,
                "verbose": cls._model.verbose,
                "seed": cls._model.seed,
                "start_time": cls._model.start_time}
        df = pd.DataFrame.from_dict(info, orient='index', columns=['value'])
        return df

    @classmethod
    def ST_GetModel(cls):
        return cls._model

    @classmethod
    def ST_GetModelSetting(cls, key):
        return cls._model_settings.get(key)

    @classmethod
    def ST_GetTrainingSetting(cls, key):
        return cls._training_settings.get(key)

    @classmethod
    def ST_IsTraining(cls):
        return cls._is_training

    @classmethod
    def ST_IsTrainComplete(cls):
        return cls._training_complete

    @classmethod
    def ST_GetCachedLogFolderName(cls):
        return cls._cached_log_folder_name

    @classmethod
    def ST_GetFloorPlanImage(cls):
        if cls._env.fp is None:
            return FloorPlan.EMPTY_IMG
        return cls._env.fp.draw()

    @classmethod
    def ST_GetFloorPlanStaticInfo(cls):
        info = {'GRID_UNIT_SIZE': FloorPlan.GRID_UNIT_SIZE,
                'GRID_AREA': FloorPlan.GRID_AREA,
                'GRID_NUM_X': FloorPlan.GRID_NUM_X,
                'GRID_NUM_Y': FloorPlan.GRID_NUM_Y,
                'NUM_LAYERS': FloorPlan.NUM_LAYERS
                }
        return info

    @classmethod
    def ST_GetFloorPlanRawDataImage(cls, channel: Union[Literal['r', 'g', 'b', 'rgb'], int] = 'rgb'):
        if cls._env.fp is None:
            return FloorPlan.EMPTY_IMG
        return cls._env.fp.get_raw_data_img(channel)

    @classmethod
    def ST_GetFloorPlanInfo(cls) -> dict[str, any]:
        if cls._env.fp is None:
            return {}
        return cls._env.fp.get_info()


class CustomCallback(BaseCallback):
    """
    A custom callback that derives from ``BaseCallback``.

    :param verbose: Verbosity level: 0 for no output, 1 for info messages, 2 for debug messages
    """

    def __init__(self, update_freq=1024, verbose: int = 0):
        super().__init__(verbose)
        self.update_freq = update_freq
        # Those variables will be accessible in the callback
        # (they are defined in the base class)
        # The RL model
        # self.model = None  # type: BaseAlgorithm
        # An alias for self.model.get_env(), the environment used for training
        # self.training_env # type: VecEnv
        # Number of time the callback was called
        # self.n_calls = 0  # type: int
        # num_timesteps = n_envs * n times env.step() was called
        # self.num_timesteps = 0  # type: int
        # local and global variables
        # self.locals = {}  # type: Dict[str, Any]
        # self.globals = {}  # type: Dict[str, Any]
        # The logger object, used to report things in the terminal
        # self.logger # type: stable_baselines3.common.logger.Logger
        # Sometimes, for event callback, it is useful
        # to have access to the parent object
        # self.parent = None  # type: Optional[BaseCallback]
        self.total_timesteps = -1
        self.start_time = datetime.now()

    def _on_training_start(self) -> None:
        self.total_timesteps = self.locals['total_timesteps']

    def _on_rollout_start(self) -> None:
        pass

    def _on_step(self) -> bool:
        """
        This method will be called by the model after each call to `env.step()`.

        For child callback (of an `EventCallback`), this will be called
        when the event is triggered.

        :return: If the callback returns False, training is aborted early.
        """
        if self.num_timesteps % self.update_freq == 0:
            Backend.CB_OnProgressChanged(self)
            if not Backend.ST_IsTraining():
                return False
        return True

    def _on_rollout_end(self) -> None:
        pass

    def _on_training_end(self) -> None:
        Backend._is_training = False
        Backend._training_complete = True


PERIODS = [
    ('year', 60 * 60 * 24 * 365),
    ('month', 60 * 60 * 24 * 30),
    ('day', 60 * 60 * 24),
    ('hour', 60 * 60),
    ('minute', 60),
    ('second', 1)
]


def format_timedelta(delta):
    total_seconds = int(delta.total_seconds())
    time_string = []
    for period_name, period_seconds in PERIODS:
        if total_seconds >= period_seconds:
            period_value, total_seconds = divmod(total_seconds, period_seconds)
            if period_value > 1:
                period_name += 's'
            time_string.append(f"{period_value} {period_name}")

    return ', '.join(time_string)
