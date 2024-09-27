# pip install streamlit
# python -m streamlit run app.py --server.address 0.0.0.0
import math
from typing import Optional

import streamlit as st
import streamlit_authenticator as stauth
import yaml
from streamlit.runtime.uploaded_file_manager import UploadedFile
from yaml.loader import SafeLoader

from app_backend import Backend as ab


def env_region():
    _is_model_training = ab.ST_IsTraining()

    # -----------------------------------------------------------------------------------------------------

    st.header("# Create Gym Env (Optional)")
    env_create_area = st.container()
    env_name_area = st.container()

    # -----------------------------------------------------------------------------------------------------
    with env_create_area:
        if st.button("New Env", type='secondary', use_container_width=True, disabled=_is_model_training):
            ab.ST_OnNewEnvButtonPressed()
    with env_name_area:
        col1, col2 = st.columns([1, 20])
        with col1:
            st.image("./images/gymlogo_normal.png", width=24)
        with col2:
            st.text(f'Env: {ab.ST_GetEnvName()}')


def model_region():
    _is_model_training = ab.ST_IsTraining()
    _modeL_region_first_loop = "model_region_first_loop" not in st.session_state
    if _modeL_region_first_loop:
        st.session_state.model_region_first_loop = False
    # -----------------------------------------------------------------------------------------------------

    st.header("# Create Or Load A Model")
    model_settings_area = st.container()
    model_info_area = st.container()

    # -----------------------------------------------------------------------------------------------------
    with model_settings_area:
        # settings
        OPTIONS_CREATE_OR_LOAD_MODEL = ['Create New Model', 'Load Pretrained Model']
        OPTIONS_MODEL_N_STEPS = [int(math.pow(2, n)) for n in range(8, 15)]
        OPTIONS_MODEL_VERBOSE = [0, 1, 2]

        if _modeL_region_first_loop:
            st.session_state.create_or_load_model_op_default = OPTIONS_CREATE_OR_LOAD_MODEL.index(ab.ST_GetModelSetting("create_or_load_model_op"))
            st.session_state.model_n_steps_default = OPTIONS_MODEL_N_STEPS.index(ab.ST_GetModelSetting("model_n_steps"))
            st.session_state.model_verbose_default = OPTIONS_MODEL_VERBOSE.index(ab.ST_GetModelSetting("model_verbose"))
            st.session_state.enable_tb_log_default = ab.ST_GetModelSetting("enable_tb_log")

        create_or_load_model_op = st.selectbox('Create Or Load Model',
                                               options=OPTIONS_CREATE_OR_LOAD_MODEL,
                                               disabled=_is_model_training,
                                               index=st.session_state.create_or_load_model_op_default)
        create_or_load_model_op_changed = create_or_load_model_op != ab.ST_GetModelSetting("create_or_load_model_op")

        model_n_steps = st.selectbox("N Steps (default: 2048)",
                                     options=OPTIONS_MODEL_N_STEPS,
                                     disabled=_is_model_training,
                                     index=st.session_state.model_n_steps_default)
        model_n_steps_changed = model_n_steps != ab.ST_GetModelSetting("model_n_steps")

        model_verbose = st.selectbox("Model Verbose (default: 1)",
                                     options=OPTIONS_MODEL_VERBOSE,
                                     disabled=_is_model_training,
                                     index=st.session_state.model_verbose_default)

        model_verbose_changed = model_verbose != ab.ST_GetModelSetting("model_verbose")

        enable_tb_log = st.checkbox("Enable Tensorboard Log",
                                    disabled=_is_model_training,
                                    value=st.session_state.enable_tb_log_default)
        enable_tensorboard_log_changed = enable_tb_log != ab.ST_GetModelSetting("enable_tb_log")

        # create or load model buttons

        if create_or_load_model_op == OPTIONS_CREATE_OR_LOAD_MODEL[0]:
            if st.button("Create New Model", use_container_width=True, type='secondary', disabled=_is_model_training):
                ab.ST_OnNewModelButtonPressed()
        elif create_or_load_model_op == OPTIONS_CREATE_OR_LOAD_MODEL[1]:
            file: Optional[UploadedFile] = st.file_uploader("Choose a file", type="zip", disabled=_is_model_training)
            try:
                success = ab.ST_LoadModelAuto(file,
                                              verbose=model_verbose,
                                              enable_tb_log=enable_tb_log)
                if success:
                    st.toast(f'Successfully Uploaded {file.name}')
            except Exception as e:
                st.error(str(e))
    # 逻辑代码:检测模型设置是否被改变
    model_settings_changed = create_or_load_model_op_changed | enable_tensorboard_log_changed | model_n_steps_changed | model_verbose_changed
    if model_settings_changed:
        ab.ST_SetModelSetting("create_or_load_model_op", create_or_load_model_op)
        ab.ST_SetModelSetting("enable_tb_log", enable_tb_log)
        ab.ST_SetModelSetting("model_n_steps", model_n_steps)
        ab.ST_SetModelSetting("model_verbose", model_verbose)
        ab.ST_SetTrainingSetting("total_timesteps", model_n_steps)
        if ab.ST_HasModel():
            st.warning("Model Settings has been changed, the original model has been removed. You need to create a new model")
            ab.ST_RemoveModel()
    with model_info_area:
        # model name area
        col1, col2 = st.columns([1, 20])
        with col1:
            if ab.ST_HasModel():
                st.image("./images/model_normal.png", width=24)
            else:
                st.image("./images/model_disabled.png", width=24)
        with col2:
            st.text(f'Model: {ab.ST_GetModelName()}')
        # model info area
        with st.expander('Model Info'):
            if (model_info := ab.ST_GetModelInfo()) is not None:
                st.text("Overview: ")
                st.dataframe(model_info, use_container_width=True)
            if (model_summary := ab.ST_GetModelSummary()) is not None:
                st.text("summary: ")
                st.text(model_summary)


def training_region():
    # -----------------------------------------------------------------------------------------------------
    st.header("# Training")
    if not ab.ST_HasModel():
        st.info("Please Create or Load Model First")
        return

    training_settings_area = st.container()
    training_info_area = st.container()
    training_button_area = st.container()

    # -----------------------------------------------------------------------------------------------------
    _is_model_training = ab.ST_IsTraining()
    _training_region_first_loop = "training_region_first_loop" not in st.session_state
    if _training_region_first_loop:
        st.session_state.training_region_first_loop = False
    if 'training_complete_in_last_frame' not in st.session_state:
        st.session_state.training_complete_in_last_frame = False

    with training_settings_area:

        if _training_region_first_loop:
            st.session_state.total_timesteps = ab.ST_GetTrainingSetting('total_timesteps')
            st.session_state.enable_eval = ab.ST_GetTrainingSetting('enable_eval')
            st.session_state.use_custom_task_name = ab.ST_GetTrainingSetting('use_custom_task_name')
            st.session_state.save_final_model = ab.ST_GetTrainingSetting('save_final_model')
            st.session_state.eval_freq_mul = ab.ST_GetTrainingSetting('eval_freq_mul')
            st.session_state.task_name = ab.ST_GetTrainingSetting('task_name')
            st.session_state.callback_update_freq = ab.ST_GetTrainingSetting('callback_update_freq')
        st.text("Training Settings")
        model_n_steps = ab.ST_GetModelSetting("model_n_steps")
        total_timesteps = st.slider('Total Timesteps',
                                    min_value=model_n_steps,
                                    max_value=model_n_steps * 1000,
                                    value=st.session_state.total_timesteps,
                                    step=model_n_steps)

        ab.ST_SetTrainingSetting('total_timesteps', total_timesteps)

        col1, col2 = st.columns([1, 1])
        with col1:
            enable_eval = st.checkbox("Enable Eval", value=st.session_state.enable_eval)
            ab.ST_SetTrainingSetting('enable_eval', enable_eval)
            eval_freq_mul = st.number_input("Eval Freq Mul", 1, 100, step=1, placeholder="Eval Freq Mul", label_visibility='collapsed') if enable_eval else 1
            ab.ST_SetTrainingSetting('eval_freq_mul', eval_freq_mul)
            use_custom_task_name = st.checkbox("Use Custom Task Name", value=st.session_state.use_custom_task_name)
            ab.ST_SetTrainingSetting('use_custom_task_name', use_custom_task_name)
            task_name = st.text_input("Custom Task Name", placeholder="Custom TaskName", label_visibility='collapsed') if use_custom_task_name else None
            ab.ST_SetTrainingSetting('task_name', task_name)
            save_final_model = st.checkbox("Save Final Model", value=st.session_state.save_final_model)
            ab.ST_SetTrainingSetting('save_final_model', save_final_model)

        with col2:
            with st.popover("Advanced Training Settings"):
                callback_update_freq = st.number_input(label="Callback Update Frequency", min_value=1, value=st.session_state.callback_update_freq)
                ab.ST_SetTrainingSetting('callback_update_freq', callback_update_freq)


    with training_info_area:
        st_training_status = st.status("No Task", state="error")
        if ab.ST_IsTrainComplete():
            st_training_status.update(label="Complete", state='complete', expanded=True)
        if (log_folder_name := ab.ST_GetCachedLogFolderName()) is not None and not ab.ST_IsTraining():
            st_training_status.text(f"Last Task Name: {log_folder_name}")

        ab.ST_RegisterTrainingObjs(st_training_status)

    with training_button_area:
        if ab.ST_IsTraining():
            if st.button("Stop", use_container_width=True, type='primary'):
                ab.ST_StopTraining()
                st.rerun()
        else:
            if st.button("Train", use_container_width=True, type='primary'):
                ab.ST_StartTraining()  # 这里不会直接开始， 而是做一个标记
                st.rerun()
        if ab.ST_HasModel() and not ab.ST_IsTraining():
            if st.button("Download Current Model"):
                ab.ST_GetModel().save('./tmp/model.zip')
                with open('./tmp/model.zip', 'rb') as f:
                    st.download_button("Click to download", f, file_name="model.zip")

    # training
    if ab.ST_HandleTrainingAuto():  # 训练过程，耗时操作
        st.session_state.training_complete_in_last_frame = True
        st.rerun()

    # on training complete
    if st.session_state.training_complete_in_last_frame:
        st.session_state.training_complete_in_last_frame = False
        st.toast("Training Complete")
        st.balloons()


def testing_region():
    st.header("# Testing")
    # -----------------------------------------------------------------------------------------------------
    if not ab.ST_HasModel():
        st.info("Please Create or Load Model First")
        return

    testing_op_area = st.container()
    testing_info_area = st.container()

    with testing_op_area:
        seed = st.number_input("seed (set to -1 to random)", -1, 1000)

        col1, col2, col3 = st.columns([3, 3, 4])
        with col1:
            if st.button("Reset Env"):
                ab.ST_OnResetEnvButtonPressed(seed)
        with col2:
            if st.button("Take Action"):
                ab.ST_OnTakeActionButtonPressed()
        with col3:
            if st.button("Generate Example Rooms (Debug)"):
                ab.ST_OnGenerateRandomRoomButtonPressed(seed)
    with testing_info_area:
        with st.expander('Env Info'):
            st.dataframe(ab.ST_GetEnvInfo(), use_container_width=True)
            st.dataframe(ab.ST_GetEnvRewardInfo(), use_container_width=True)
        st.subheader("Render Image")
        st.image(ab.ST_GetFloorPlanImage())

        st.subheader('Observation(Scaled to 0 to 255)')
        info = ab.ST_GetFloorPlanStaticInfo()
        st.text(str(info))
        col1, col2, col3 = st.columns(3)
        with col1:
            st.image(ab.ST_GetFloorPlanRawDataImage('r'), caption="Valid Layer")
        with col2:
            st.image(ab.ST_GetFloorPlanRawDataImage('g'), caption="Type Layer")
        with col3:
            st.image(ab.ST_GetFloorPlanRawDataImage('b'), caption="Idx Layer")
        st.subheader('Floor Plan Info')
        with st.expander("expand"):
            st.write(ab.ST_GetFloorPlanInfo())


def about_region():
    # -----------------------------------------------------------------------------------------------------
    st.header("# About")
    st.text("Room Plan Generation using Reinforcement Learning")
    st.text("Group Menber: Feng Yiheng, Liu Jianglong")


def main_page():
    st.title('PlanGenRL')
    st.text("Room Plan Generation using Reinforcement Learning")
    env_region()
    model_region()
    training_region()
    testing_region()
    about_region()


if __name__ == '__main__':
    with open('./config.yaml') as file:
        config = yaml.load(file, Loader=SafeLoader)
    authenticator = stauth.Authenticate(
        config['credentials'],
        config['cookie']['name'],
        config['cookie']['key'],
        config['cookie']['expiry_days'],
        config['pre-authorized']
    )
    authenticator.login()

    if st.session_state["authentication_status"]:
        st.write(f'Welcome *{st.session_state["name"]}*')
        main_page()
        authenticator.logout(location='main')
    elif st.session_state["authentication_status"] is False:
        st.error('Username/password is incorrect')
    elif st.session_state["authentication_status"] is None:
        st.warning('Please enter your username and password')
