import warnings

warnings.filterwarnings('ignore', category=DeprecationWarning)  # message=r'.*jsonschema.RefResolver is deprecated.*')

from abc import abstractmethod, ABC  # noqa
import neptune  # noqa
from datetime import datetime  # noqa
import logging  # noqa
from stable_baselines3.common.callbacks import BaseCallback  # noqa
from pathlib import Path  # noqa
import numpy as np  # noqa
from typing import Literal, Union  # noqa

logger = logging.getLogger(__name__)


def init_neptune_run(run_name: Union[str, int, None], neptune_cfg, read_only: bool = False) -> neptune.Run:
    """Initialize a neptune run. If neptune_run is None, create a new run. Else, tryies to resume the run.

    Args:
        neptune_run (str | int | None): Name of the neptune run to load. If None, create a new run.
        neptune_cfg (OmegaDict): Neptune config

    Returns:
        neptune.Run: _description_
    """
    # Load existing neptune run
    if run_name is not None:
        run_id = f'PH-{run_name}'
        logger.info(f'Loading existing run: {run_id}')
        mode = 'read-only' if read_only else 'async'
        run = neptune.init_run(
            with_id=run_id,
            project=neptune_cfg.project_name,
            api_token=neptune_cfg.api_token,
            mode=mode,
        )

    # Create new neptune run
    else:
        run_name = datetime.today().strftime('%Y-%m-%d_%H-%M_PH')

        logger.info('Initialization of neptune run')
        run = neptune.init_run(
            project=neptune_cfg.project_name,
            api_token=neptune_cfg.api_token,
            name=run_name,
        )

    run.wait()
    logger.info(f"Run id: {run['sys/id'].fetch()}")
    logger.info(f"Run name: {run['sys/name'].fetch()}")

    return run


class NeptuneCallback(ABC):
    """
    Callback class for logging and saving data to Neptune.

    Two types of data:
    - Env data: Data coming from the environment (Observations, actions, rewards, ...).
                Logged at every env_log_freq steps
                if log_env is True.

    - Episode data: Summary of the episode (Total reward, avg force, avg torque, ...).
                    Logged at the end of each episode.

    During an episode, the data is stored in the episode_log dict. The data is sent to neptune at the end of each
    episode.

    Args:
        neptune_run (neptune.Run): Neptune run object.
        run_type (Literal['test', 'train'], optional): Type of run ('test' or 'train'). Defaults to 'train'.
        env_log_freq (int, optional): Frequency (in ts) at which to log the environment data. Defaults to 10.
        save_freq (int, optional): Time step frequency at which to save the model. Defaults to 10000.
        save_path (Union[Path, None], optional): Path to save the model. Defaults to None.
        save_name_prefix (str, optional): Prefix of the saved model. Defaults to 'rl_model'.
        log_env (bool, optional): Whether to log the environment data. Defaults to True.
        save_replay_buffer (bool, optional): Whether to save the replay buffer. Defaults to True.
        save_vecnormalize (bool, optional): Whether to save the VecNormalize. Defaults to False.
        verbose (int, optional): Verbosity level. Defaults to 1.
        start_timestep (int, optional): Number of time the callback was called. Defaults to 0.
    """

    def __init__(
        self,
        neptune_run: neptune.Run,
        run_type: Literal['test', 'train'] = 'train',
        env_log_freq: int = 10,
        save_freq: int = 10_000,
        save_path: Union[Path, None] = None,
        save_name_prefix: str = 'rl_model',
        log_env: bool = True,
        save_replay_buffer: bool = True,
        save_vecnormalize: bool = False,
        neptune_n_episodes: int = 10,
        verbose: int = 1,
        start_timestep: int = 0,
    ):
        self.neptune_run = neptune_run
        self.run_type = run_type  # 'test' or 'train'

        # Base parameters
        self.num_timesteps = start_timestep  # Number of time the callback was called
        self.n_episodes = 0  # Number of episodes
        self.ep_n_steps = 0  # Number of steps in the current episode

        self.verbose = verbose  # Verbosity level

        self.log_env = log_env  # Whether to log the environment data
        self.env_log_freq = env_log_freq  # Frequency (in ts) at which to log the environment data (observations, ...)

        # Saving
        self.save_freq = save_freq  # Time step frequency at which to save the model
        self.save_name_prefix = save_name_prefix  # Prefix of the saved model
        self.save_path = save_path  # Path to save the model. Doesn't save if it's None
        self.save_replay_buffer = save_replay_buffer  # Whether to save the replay buffer
        self.save_vecnormalize = save_vecnormalize  # Whether to save the VecNormalize
        self.neptune_n_episodes = (
            neptune_n_episodes  # Number of the last episodes to keep in neptune. Needed or neptune max reached
        )

        # Logging
        self.step_env = {}  # Values of the env at the current step

        self.ep_reward_list = []  # Total reward of each ep
        self.end_depth_list = []  # End depth of each ep. List of 3 values (x, z, rot)

        self.force_norm_tot_list = []  # total force norm of each ep
        self.force_norm_avg_list = []  # avg force norm of each ep
        self.force_norm_rms_list = []  # rms force norm of each ep
        self.force_norm_max_list = []  # max force norm of each ep

        self.torque_norm_tot_list = []  # total torque norm of each ep
        self.torque_norm_avg_list = []  # avg torque norm of each ep
        self.torque_norm_rms_list = []  # rms torque norm of each ep
        self.torque_norm_max_list = []  # max torque norm of each ep

        self.episode_log = {
            'step': [],
            'obs': [],
            'command': [],
            'action': [],
            'reward': [],
            'plug_force': [],
            'plug_torque': [],
            'insertion_depth': [],
        }

    def _checkpoint_path(self, checkpoint_type: str = '', extension: str = '') -> str:
        """
        Helper to get checkpoint path for each type of checkpoint.

        Args:
            checkpoint_type (str, optional): Empty for the model, "replay_buffer_" or "vecnormalize_" for the other checkpoints.
            extension (str, optional): Checkpoint file extension (zip for model, pkl for others).

        Returns:
            str: Path to the checkpoint.
        """
        path = self.save_path / f'{self.save_name_prefix}_{checkpoint_type}{self.num_timesteps}_steps.{extension}'
        # path = self.save_path / f'{self.name_prefix}_{checkpoint_type}.{extension}'

        return path

    def step_callback(self, step_env: dict) -> bool:
        """
        Callback function called at each step.

        Args:
            step_env (dict): Values of the environment at the current step.

        Returns:
            bool: True.
        """
        # TODO: If multiple envs, need to check the ones that are done and send their data
        self.step_env = step_env

        # Record environment data
        if self.num_timesteps % self.env_log_freq == 0:
            self.record_env_step()

        # If episode over, send data to neptune
        episode_done = self.step_env.get('done', False) or self.step_env.get('truncated', False)

        if episode_done:
            self.send_ep_to_neptune()
            # self.n_episodes += np.sum(self.locals['dones']).item()
            self.n_episodes += 1

        # Save model
        if self.save_path is not None and self.num_timesteps % self.save_freq == 0:
            self.save_model()

        return True

    @abstractmethod
    def _on_step(self) -> bool:
        """To be defined specifcally in the child class (Train or Test)."""
        ...

    def record_env_step(self):
        """Record environment data. Add the important values to the episode_log dict."""

        # Observations
        obs = self.step_env.get('obs', None)
        infos = self.step_env.get('infos', None)
        command = infos['command']
        plug_force = infos['plug_force']
        plug_torque = infos['plug_torque']
        insertion_depth = infos['insertion_depth']

        action = self.step_env.get('action', None)
        reward = self.step_env.get('reward', None)

        self.ep_n_steps += self.env_log_freq

        log_dict = {
            'step': self.num_timesteps,
            'obs': obs,
            'command': command,
            'action': action,
            'reward': reward,
            'plug_force': plug_force,
            'plug_torque': plug_torque,
            'insertion_depth': insertion_depth,
        }

        for param, val in log_dict.items():
            self.episode_log[param].append(val)

    def send_ep_to_neptune(self):
        """Send episode data to neptune."""
        # If no data, exit (occurs if resets before log_env_freq steps)
        if len(self.episode_log['step']) == 0:
            return

        # ----- Episode observations -----
        if self.log_env:
            ep_id = self.n_episodes % self.neptune_n_episodes

            if self.n_episodes >= self.neptune_n_episodes:
                try:
                    del self.neptune_run[f'episode/{ep_id}']
                except neptune.exceptions.NeptuneException as e:
                    logger.warning(f'Could not delete episode {ep_id} from neptune: {e}')

            ep_logger = self.neptune_run[f'episode/{ep_id}']

            # Raw values
            obs = np.vstack(self.episode_log['obs'])
            command = np.vstack(self.episode_log['command'])
            action = np.vstack(self.episode_log['action'])
            reward = np.vstack(self.episode_log['reward'])
            plug_force = np.vstack(self.episode_log['plug_force'])
            plug_torque = np.vstack(self.episode_log['plug_torque'])
            insertion_depth = np.vstack(self.episode_log['insertion_depth'])

            # Computations
            plug_force_norm = np.linalg.norm(plug_force, axis=1)
            plug_torque_norm = np.linalg.norm(plug_torque, axis=1)

            # Log dict
            log_dict = {
                'step': self.episode_log['step'],
                'j2_pos': obs[:, 0],
                'j4_pos': obs[:, 1],
                'j6_pos': obs[:, 2],
                'j2_vel': obs[:, 0 + 3],
                'j4_vel': obs[:, 1 + 3],
                'j6_vel': obs[:, 2 + 3],
                'j2_ideal_vel': obs[:, 0 + 6],
                'j4_ideal_vel': obs[:, 1 + 6],
                'j6_ideal_vel': obs[:, 2 + 6],
                'j2_torque': obs[:, 0 + 9],
                'j4_torque': obs[:, 1 + 9],
                'j6_torque': obs[:, 2 + 9],
                'j2_cmd': command[:, 0],
                'j4_cmd': command[:, 1],
                'j6_cmd': command[:, 2],
                'j2_act': action[:, 0],
                'j6_act': action[:, 1],
                'reward': reward[:, 0],
                'plug_force_norm': plug_force_norm,
                'plug_force_x': plug_force[:, 0],
                'plug_force_y': plug_force[:, 1],
                'plug_force_z': plug_force[:, 2],
                'plug_torque_norm': plug_torque_norm,
                'plug_torque_x': plug_torque[:, 0],
                'plug_torque_y': plug_torque[:, 1],
                'plug_torque_z': plug_torque[:, 2],
                'insertion_depth_x': insertion_depth[:, 0],
                'insertion_depth_z': insertion_depth[:, 1],
                'insertion_depth_rot': insertion_depth[:, 2],
            }

            for param, val in log_dict.items():
                ep_logger[param].extend(list(val))  # TODO: Add timestep to values

            # Reset infos
            reset_infos = self.step_env['reset_info']  # self.training_env.reset_infos[0]
            self.neptune_run[f'episode/misaligment/{ep_id}'] = reset_infos['insertion_misalignment']
            self.neptune_run['data/misaligment'].append(reset_infos['insertion_misalignment'])

        # ----- Episode summary -----
        ep_reward = np.sum(reward * self.env_log_freq)  # Episodic reward

        end_insert_depth = insertion_depth[-1]

        ep_total_force = np.sum(plug_force_norm)
        ep_total_torque = np.sum(plug_torque_norm)

        ep_avg_force = np.mean(plug_force_norm)
        ep_avg_torque = np.mean(plug_torque_norm)

        ep_max_force = np.max(plug_force_norm)
        ep_max_torque = np.max(plug_torque_norm)

        ep_rms_force = np.sqrt(np.mean(plug_force_norm**2))
        ep_rms_torque = np.sqrt(np.mean(plug_torque_norm**2))

        # Add the evaluation values to the lists
        self.ep_reward_list.append(ep_reward)
        self.end_depth_list.append(end_insert_depth)

        self.force_norm_tot_list.append(ep_total_force)
        self.torque_norm_tot_list.append(ep_total_torque)

        self.force_norm_avg_list.append(ep_avg_force)
        self.torque_norm_avg_list.append(ep_avg_torque)

        self.force_norm_max_list.append(ep_max_force)
        self.torque_norm_max_list.append(ep_max_torque)

        self.force_norm_rms_list.append(ep_rms_force)
        self.torque_norm_rms_list.append(ep_rms_torque)

        # Rolling averages (Useful to check the learning curve)
        mean_reward = np.mean(self.ep_reward_list[-100:])  # Mean reward of the last 100 episodes
        mean_total_force = np.mean(self.force_norm_tot_list[-100:])  # Mean total force of the last 100 episodes
        mean_total_torque = np.mean(self.torque_norm_tot_list[-100:])  # Mean total torque of the last 100 episodes
        mean_end_depth_z = np.mean(
            [x[1] for x in self.end_depth_list[-100:]]
        )  # Mean end depth z of the last 100 episodes

        # Send the evaluation values to neptune
        ep_data = {
            'ep_reward': ep_reward,
            'ep_total_force': ep_total_force,
            'ep_total_torque': ep_total_torque,
            'ep_avg_force': ep_avg_force,
            'ep_avg_torque': ep_avg_torque,
            'ep_max_force': ep_max_force,
            'ep_max_torque': ep_max_torque,
            'ep_rms_force': ep_rms_force,
            'ep_rms_torque': ep_rms_torque,
            'ep_end_depth_x': end_insert_depth[0],
            'depth_z_target': 0.0156,
            'ep_end_depth_z': end_insert_depth[1],
            'ep_end_depth_rot': end_insert_depth[2],
            'mean_reward': mean_reward,
            'mean_total_force': mean_total_force,
            'mean_total_torque': mean_total_torque,
            'mean_end_depth_z': mean_end_depth_z,
        }

        ep_logger = self.neptune_run['data']
        for param, val in ep_data.items():
            ep_logger[param].append(value=val, step=self.num_timesteps)

        ep_logger['timestep'] = self.num_timesteps  # Update last timestep

        # ----- Reset episodic data -----
        self.episode_log = {
            'step': [],
            'obs': [],
            'command': [],
            'action': [],
            'reward': [],
            'plug_force': [],
            'plug_torque': [],
            'insertion_depth': [],
        }
        self.ep_n_steps = 0

    def save_model(self):
        """Save model and replay buffer. From sb3.callbacks.CheckpointCallback."""
        if self.save_path is None:
            return

        model_path = self._checkpoint_path(extension='zip')
        self.model.save(model_path)

        self.neptune_run[f'model_checkpoints/{self.num_timesteps}/model'].upload(model_path.as_posix())

        if self.verbose >= 2:
            print(f'Saving model checkpoint to {model_path}')

        if self.save_replay_buffer and hasattr(self.model, 'replay_buffer') and self.model.replay_buffer is not None:
            # If model has a replay buffer, save it too
            replay_buffer_path = self._checkpoint_path('replay_buffer_', extension='pkl')
            self.model.save_replay_buffer(replay_buffer_path)  # type: ignore[attr-defined]
            if self.verbose > 1:
                print(f'Saving model replay buffer checkpoint to {replay_buffer_path}')

            self.neptune_run['model_checkpoints/buffer'].upload(replay_buffer_path.as_posix())

        if self.save_vecnormalize and self.model.get_vec_normalize_env() is not None:
            # Save the VecNormalize statistics
            vec_normalize_path = self._checkpoint_path('vecnormalize_', extension='pkl')
            self.model.get_vec_normalize_env().save(vec_normalize_path)  # type: ignore[union-attr]
            if self.verbose >= 2:
                print(f'Saving model VecNormalize to {vec_normalize_path}')

            self.neptune_run['model_checkpoints/vec_normalize'].upload(vec_normalize_path.as_posix())

    def _on_run_end(self) -> None:
        # Compute the mean of the evaluation values
        force_norm_avg = np.mean(self.force_norm_avg_list)
        force_norm_rms = np.mean(self.force_norm_rms_list)
        force_norm_max = np.mean(self.force_norm_max_list)
        torque_norm_avg = np.mean(self.torque_norm_avg_list)
        torque_norm_rms = np.mean(self.torque_norm_rms_list)
        torque_norm_max = np.mean(self.torque_norm_max_list)

        eval_values_dict = {
            'reward_avg': np.mean(self.ep_reward_list),
            'force_norm_avg': force_norm_avg,
            'force_norm_rms': force_norm_rms,
            'force_norm_max': force_norm_max,
            'torque_norm_avg': torque_norm_avg,
            'torque_norm_rms': torque_norm_rms,
            'torque_norm_max': torque_norm_max,
        }

        eval_logger = self.neptune_run['eval']

        for param, val in eval_values_dict.items():
            eval_logger[param] = val


class NeptuneTrainCallback(NeptuneCallback, BaseCallback):
    def __init__(
        self,
        neptune_run: neptune.Run,
        env_log_freq: int = 10,
        save_freq: int = 10_000,
        save_path: Union[Path, None] = None,
        save_name_prefix: str = 'rl_model',
        log_env: bool = True,
        save_replay_buffer: bool = True,
        save_vecnormalize: bool = False,
        neptune_n_episodes: int = 10,
        verbose: int = 1,
        start_timestep: int = 0,
    ):
        BaseCallback.__init__(self, verbose)

        NeptuneCallback.__init__(
            self,
            neptune_run,
            run_type='train',
            env_log_freq=env_log_freq,
            save_freq=save_freq,
            save_path=save_path,
            save_name_prefix=save_name_prefix,
            log_env=log_env,
            save_replay_buffer=save_replay_buffer,
            save_vecnormalize=save_vecnormalize,
            neptune_n_episodes=neptune_n_episodes,
            verbose=verbose,
            start_timestep=start_timestep,
        )

    def _init_callback(self) -> None:
        # Create folder if needed
        if self.save_path is not None:
            self.save_path.mkdir(parents=True, exist_ok=True)

        self._load_existing_run()

        if self.num_timesteps != 0:
            logger.info(f'Deleting data after timestep {self.num_timesteps} from neptune run')

            # Delete all data after the start timestep
            data_keys = list(self.neptune_run.get_structure()['data'].keys())

            for each_key in data_keys:
                if each_key in ['timestep']:
                    continue

                vals = self.neptune_run[f'data/{each_key}'].fetch_values()
                vals = vals[vals['step'] <= self.num_timesteps]

                del self.neptune_run[f'data/{each_key}']
                self.neptune_run[f'data/{each_key}'].extend(
                    values=vals['value'].to_list(), steps=vals['step'].to_list()
                )

    def _on_training_end(self) -> None:
        self.save_model()

    def _on_step(self):
        """Called at each step by SB3 during training"""
        # TODO: Support vectorized envs
        assert 'dones' in self.locals, '`dones` variable is not defined'
        self.num_timesteps += 1

        step_env = {
            'obs': self.locals.get('new_obs', None),
            'reward': self.locals.get('rewards', None),
            'done': self.locals.get('dones', None)[0],
            'truncated': None,  # TODO
            'infos': self.locals.get('infos', None)[0],
            'action': self.locals.get('actions', None),
            'reset_info': self.training_env.reset_infos[0],
        }

        return self.step_callback(step_env)

    def _load_existing_run(self):
        """Load past run if needed."""
        if self.num_timesteps != 0:
            logger.info(f'Loading existing run information from timestep {self.num_timesteps}')

            self.ep_reward_list = self.neptune_run['data/ep_reward'].fetch_values()['value'].values[-100:].tolist()

            self.force_norm_tot_list = (
                self.neptune_run['data/ep_total_force'].fetch_values()['value'].values[-100:].tolist()
            )
            self.force_norm_avg_list = (
                self.neptune_run['data/ep_avg_force'].fetch_values()['value'].values[-100:].tolist()
            )
            self.force_norm_rms_list = (
                self.neptune_run['data/ep_rms_force'].fetch_values()['value'].values[-100:].tolist()
            )
            self.force_norm_max_list = (
                self.neptune_run['data/ep_max_force'].fetch_values()['value'].values[-100:].tolist()
            )

            self.torque_norm_tot_list = (
                self.neptune_run['data/ep_total_torque'].fetch_values()['value'].values[-100:].tolist()
            )
            self.torque_norm_avg_list = (
                self.neptune_run['data/ep_avg_torque'].fetch_values()['value'].values[-100:].tolist()
            )
            self.torque_norm_rms_list = (
                self.neptune_run['data/ep_rms_torque'].fetch_values()['value'].values[-100:].tolist()
            )
            self.torque_norm_max_list = (
                self.neptune_run['data/ep_max_torque'].fetch_values()['value'].values[-100:].tolist()
            )

            end_depth_x = self.neptune_run['data/ep_end_depth_x'].fetch_values()['value'].values[-100:].tolist()
            end_depth_z = self.neptune_run['data/ep_end_depth_z'].fetch_values()['value'].values[-100:].tolist()
            end_depth_rot = self.neptune_run['data/ep_end_depth_rot'].fetch_values()['value'].values[-100:].tolist()

            self.end_depth_list = [np.array([x, z, r]) for x, z, r in zip(end_depth_x, end_depth_z, end_depth_rot)]


class NeptuneTestCallback(NeptuneCallback):
    def __init__(
        self,
        neptune_run: neptune.Run,
        env_log_freq: int = 10,
        log_env: bool = True,
        neptune_n_episodes: int = 10,
        verbose: int = 1,
    ):
        NeptuneCallback.__init__(
            self,
            neptune_run,
            run_type='test',
            env_log_freq=env_log_freq,
            save_path=None,
            log_env=log_env,
            save_replay_buffer=False,
            save_vecnormalize=False,
            neptune_n_episodes=neptune_n_episodes,
            verbose=verbose,
            start_timestep=0,
        )

    def _on_step(self, obs, reward, terminated, truncated, info, action, reset_info) -> bool:
        """Need to be called manually at each step during testing."""
        self.num_timesteps += 1

        step_env = {
            'obs': obs,
            'reward': reward,
            'done': terminated,
            'truncated': truncated,
            'infos': info,
            'action': action,
            'reset_info': reset_info,
        }

        return self.step_callback(step_env)

    def _on_test_end(self) -> None:
        """To compute the average, over all test runs, of the evaluation data.

        Called at the end of the test.

        Evaluations data:
            - reward_avg
            - force_norm_tot
            - force_norm_avg
            - force_norm_rms
            - force_norm_max
            - torque_norm_tot
            - torque_norm_avg
            - torque_norm_rms
            - torque_norm_max

        """
        force_norm_tot = np.mean(self.force_norm_tot_list)
        force_norm_avg = np.mean(self.force_norm_avg_list)
        force_norm_rms = np.mean(self.force_norm_rms_list)
        force_norm_max = np.mean(self.force_norm_max_list)
        torque_norm_tot = np.mean(self.torque_norm_tot_list)
        torque_norm_avg = np.mean(self.torque_norm_avg_list)
        torque_norm_rms = np.mean(self.torque_norm_rms_list)
        torque_norm_max = np.mean(self.torque_norm_max_list)

        eval_values_dict = {
            'reward_avg': np.mean(self.ep_reward_list),
            'force_norm_tot': force_norm_tot,
            'force_norm_avg': force_norm_avg,
            'force_norm_rms': force_norm_rms,
            'force_norm_max': force_norm_max,
            'torque_norm_tot': torque_norm_tot,
            'torque_norm_avg': torque_norm_avg,
            'torque_norm_rms': torque_norm_rms,
            'torque_norm_max': torque_norm_max,
        }

        eval_logger = self.neptune_run['eval']

        for param, val in eval_values_dict.items():
            eval_logger[param] = val
