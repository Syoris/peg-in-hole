import warnings

warnings.filterwarnings('ignore', category=DeprecationWarning)  # message=r'.*jsonschema.RefResolver is deprecated.*')

import neptune  # noqa
from datetime import datetime  # noqa
import logging  # noqa
from stable_baselines3.common.callbacks import BaseCallback  # noqa
from pathlib import Path  # noqa
import numpy as np  # noqa
from typing import Union  # noqa

logger = logging.getLogger(__name__)


def init_neptune_run(run_name: Union[str, None], neptune_cfg, read_only: bool = False) -> neptune.Run:
    """Initialize a neptune run. If neptune_run is None, create a new run. Else, tryies to resume the run.

    Args:
        neptune_run (str | None): Name of the neptune run to resume. If None, create a new run.
        neptune_cfg (OmegaDict): Neptune config

    Returns:
        neptune.Run: _description_
    """
    # Create new neptune run
    if run_name is not None:
        logger.info(f'Loading existing run: {run_name}')
        mode = 'read-only' if read_only else 'async'
        run = neptune.init_run(
            with_id=run_name,
            project=neptune_cfg.project_name,
            api_token=neptune_cfg.api_token,
            mode=mode,
        )

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


# TODO: Merge NeptuneTrainCallback and NeptuneTestCallback
class NeptuneTrainCallback(BaseCallback):
    def __init__(
        self,
        neptune_run: neptune.Run,
        env_log_freq: int,
        save_freq: int,
        save_path: Path,
        name_prefix: str = 'rl_model',
        log_env: bool = True,
        save_replay_buffer: bool = False,
        save_vecnormalize: bool = False,
        verbose: int = 1,
        start_timestep: int = 0,
    ):
        super().__init__(verbose)
        self.neptune_run = neptune_run

        self.log_env = log_env
        self.env_log_freq = env_log_freq
        self.save_freq = save_freq

        self.name_prefix = name_prefix
        self.save_path = save_path
        self.save_replay_buffer = save_replay_buffer
        self.save_vecnormalize = save_vecnormalize

        self.neptune_n_episodes = 10  # Number of the last episodes to keep in neptune. Needed or neptune max reached

        self.n_episodes = 0
        self.episodic_reward = 0
        self.episodic_force = 0
        self.episodic_torque = 0
        self.ep_n_steps = 0
        self.num_timesteps = start_timestep
        self.rewards_list = []  # TODO: Load past reward on resume

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

    def _init_callback(self) -> None:
        # Create folder if needed
        if self.save_path is not None:
            self.save_path.mkdir(parents=True, exist_ok=True)

        if self.num_timesteps != 0:
            logger.debug(f'Deleting data after timestep {self.num_timesteps} from neptune run')

            # Delete all data after the start timestep
            data_keys = list(self.neptune_run.get_structure()['data'].keys())

            for each_key in data_keys:
                vals = self.neptune_run[f'data/{each_key}'].fetch_values()
                vals = vals[vals['step'] <= self.num_timesteps]

                del self.neptune_run[f'data/{each_key}']
                self.neptune_run[f'data/{each_key}'].extend(
                    values=vals['value'].to_list(), steps=vals['step'].to_list()
                )

    def _checkpoint_path(self, checkpoint_type: str = '', extension: str = '') -> str:
        """
        Helper to get checkpoint path for each type of checkpoint.

        :param checkpoint_type: empty for the model, "replay_buffer_"
            or "vecnormalize_" for the other checkpoints.
        :param extension: Checkpoint file extension (zip for model, pkl for others)
        :return: Path to the checkpoint
        """
        path = self.save_path / f'{self.name_prefix}_{checkpoint_type}{self.num_timesteps}_steps.{extension}'
        # path = self.save_path / f'{self.name_prefix}_{checkpoint_type}.{extension}'

        return path

    def _on_training_end(self) -> None:
        self.save_model()

    def _on_step(self) -> bool:
        # Record environment data
        if self.n_calls % self.env_log_freq == 0:
            self.record_env_step()

        # If episode over, send data to neptune
        assert 'dones' in self.locals, '`dones` variable is not defined'
        episode_done = self.locals.get('dones', None)[0]
        if episode_done:
            # TODO: If multiple envs, need to check the ones that are done and send their data
            self.send_ep_to_neptune()
            self.n_episodes += np.sum(self.locals['dones']).item()

        # Save model
        if self.n_calls % self.save_freq == 0:
            self.save_model()

        return True

    def record_env_step(self):
        if not self.log_env:
            return

        # Observations
        obs = self.locals.get('new_obs', None)
        infos = self.locals.get('infos', None)[0]
        command = infos['command']
        plug_force = infos['plug_force']
        plug_torque = infos['plug_torque']
        insertion_depth = infos['insertion_depth']

        action = self.locals.get('actions', None)
        reward = self.locals.get('rewards', None)

        force_norm = np.linalg.norm(plug_force)
        torque_norm = np.linalg.norm(plug_torque)
        self.episodic_reward += reward
        self.episodic_force += force_norm
        self.episodic_torque += torque_norm
        self.ep_n_steps += self.env_log_freq

        log_dict = {
            'step': self.n_calls,
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
        # ----- Episode observations -----
        if self.log_env:
            ep_id = self.n_episodes % self.neptune_n_episodes

            # if self.n_episodes > self.neptune_n_episodes - 1:  # -1 bc of 0-index
            #     del_idx = self.n_episodes - self.neptune_n_episodes
            #     try:
            #         del self.neptune_run[f'episode/{del_idx}']
            #     except neptune.exceptions.NeptuneException as e:
            #         logger.warning(f'Could not delete episode {del_idx} from neptune: {e}')

            if self.n_episodes >= self.neptune_n_episodes:
                try:
                    del self.neptune_run[f'episode/{ep_id}']
                except neptune.exceptions.NeptuneException as e:
                    logger.warning(f'Could not delete episode {ep_id} from neptune: {e}')

            ep_logger = self.neptune_run[f'episode/{ep_id}']

            obs = np.vstack(self.episode_log['obs'])
            command = np.vstack(self.episode_log['command'])
            action = np.vstack(self.episode_log['action'])
            reward = np.vstack(self.episode_log['reward'])
            plug_force = np.vstack(self.episode_log['plug_force'])
            plug_torque = np.vstack(self.episode_log['plug_torque'])
            insertion_depth = np.vstack(self.episode_log['insertion_depth'])

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
                'plug_force_x': plug_force[:, 0],
                'plug_force_y': plug_force[:, 1],
                'plug_force_z': plug_force[:, 2],
                'plug_torque_x': plug_torque[:, 0],
                'plug_torque_y': plug_torque[:, 1],
                'plug_torque_z': plug_torque[:, 2],
                'insertion_depth_x': insertion_depth[:, 0],
                'insertion_depth_z': insertion_depth[:, 1],
                'insertion_depth_rot': insertion_depth[:, 2],
            }

            for param, val in log_dict.items():
                ep_logger[param].extend(list(val))

            # Reset infos
            reset_infos = self.training_env.reset_infos[0]
            self.neptune_run[f'episode/misaligment/{ep_id}'] = reset_infos['insertion_misalignment']
            self.neptune_run['data/misaligment'].append(reset_infos['insertion_misalignment'])

        # ----- Episode summary -----
        last_insert_depth = insertion_depth[-1]
        ep_avg_force = self.episodic_force / self.ep_n_steps
        ep_avg_torque = self.episodic_torque / self.ep_n_steps

        ep_data = {
            'ep_reward': self.episodic_reward,
            'ep_avg_force': ep_avg_force,
            'ep_avg_torque': ep_avg_torque,
            'ep_end_depth_x': last_insert_depth[0],
            'ep_end_depth_z': last_insert_depth[1],
            'ep_end_depth_rot': last_insert_depth[2],
        }

        ep_logger = self.neptune_run['data']
        for param, val in ep_data.items():
            ep_logger[param].append(value=val, step=self.num_timesteps)

        ep_logger['timestep'] = self.num_timesteps

        # Mean reward of the last 100 episodes
        self.rewards_list.append(self.episodic_reward)
        mean_reward = np.mean(self.rewards_list[-100:])
        self.neptune_run['data/mean_ep_reward'].append(mean_reward, step=self.num_timesteps)

        # Reset episodic data
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
        self.episodic_reward = 0
        self.episodic_force = 0
        self.episodic_torque = 0

    def save_model(self):
        """Save model and replay buffer. From sb3.callbacks.CheckpointCallback."""
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


class NeptuneTestCallback:
    def __init__(
        self,
        neptune_run: neptune.Run,
        env_log_freq: int,
        verbose: int = 1,
        start_timestep: int = 0,
    ):
        # Base parameters
        # Number of time the callback was called
        self.num_timesteps = 0  # type: int
        self.verbose = verbose

        self.neptune_run = neptune_run

        self.env_log_freq = env_log_freq

        self.n_episodes = 0
        self.episodic_reward = 0
        self.episodic_force = 0
        self.episodic_torque = 0
        self.ep_n_steps = 0
        self.num_timesteps = start_timestep
        self.step_env = {}  # Values of the env at the current step

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

    def _on_test_end(self) -> None:
        ...

    def on_step(self, obs, reward, terminated, truncated, info, action, reset_info) -> bool:
        self.num_timesteps += 1

        self.step_env = {
            'obs': obs,
            'reward': reward,
            'done': terminated,
            'truncated': truncated,
            'infos': info,
            'action': action,
            'reset_info': reset_info,
        }

        # Record environment data
        if self.num_timesteps % self.env_log_freq == 0:
            self.record_env_step()

        # If episode over, send data to neptune
        assert 'done' in self.step_env, '`done` variable is not defined'
        episode_done = self.step_env.get('done', None)
        if episode_done:
            self.send_ep_to_neptune()
            self.n_episodes += 1

        return True

    def record_env_step(self):
        """
        Records the environment step by storing relevant information such as observations, actions, rewards,
        forces, torques, and insertion depth in the episode log.

        Args:
            None

        Returns:
            None
        """
        # Observations
        obs = self.step_env.get('obs', None)
        infos = self.step_env.get('infos', None)
        command = infos['command']
        plug_force = infos['plug_force']
        plug_torque = infos['plug_torque']
        insertion_depth = infos['insertion_depth']

        action = self.step_env.get('action', None)
        reward = self.step_env.get('reward', None)

        force_norm = np.linalg.norm(plug_force)
        torque_norm = np.linalg.norm(plug_torque)
        self.episodic_reward += reward
        self.episodic_force += force_norm
        self.episodic_torque += torque_norm
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
        # ----- Episode observations -----
        ep_logger = self.neptune_run[f'episode/{self.n_episodes}']

        obs = np.vstack(self.episode_log['obs'])
        command = np.vstack(self.episode_log['command'])
        action = np.vstack(self.episode_log['action'])
        reward = np.vstack(self.episode_log['reward'])
        plug_force = np.vstack(self.episode_log['plug_force'])
        plug_torque = np.vstack(self.episode_log['plug_torque'])
        insertion_depth = np.vstack(self.episode_log['insertion_depth'])

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
            'plug_force_x': plug_force[:, 0],
            'plug_force_y': plug_force[:, 1],
            'plug_force_z': plug_force[:, 2],
            'plug_torque_x': plug_torque[:, 0],
            'plug_torque_y': plug_torque[:, 1],
            'plug_torque_z': plug_torque[:, 2],
            'insertion_depth_x': insertion_depth[:, 0],
            'insertion_depth_z': insertion_depth[:, 1],
            'insertion_depth_rot': insertion_depth[:, 2],
        }

        for param, val in log_dict.items():
            ep_logger[param].extend(list(val))

        # Reset infos
        self.neptune_run[f'episode/misaligment/{self.n_episodes}'] = self.step_env['reset_info'][
            'insertion_misalignment'
        ]
        # ep_logger['insertion_misalignment'] = self.step_env['reset_info']['insertion_misalignment']

        # ----- Episode summary -----
        last_insert_depth = insertion_depth[-1]
        ep_avg_force = self.episodic_force / self.ep_n_steps
        ep_avg_torque = self.episodic_torque / self.ep_n_steps

        ep_data = {
            'ep_reward': self.episodic_reward,
            'ep_avg_force': ep_avg_force,
            'ep_avg_torque': ep_avg_torque,
            'ep_end_depth_x': last_insert_depth[0],
            'ep_end_depth_z': last_insert_depth[1],
            'ep_end_depth_rot': last_insert_depth[2],
        }

        ep_logger = self.neptune_run['data']
        for param, val in ep_data.items():
            ep_logger[param].append(value=val, step=self.num_timesteps)

        ep_logger['timestep'] = self.num_timesteps

        # Reset episodic data
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
        self.episodic_reward = 0
        self.episodic_force = 0
        self.episodic_torque = 0
