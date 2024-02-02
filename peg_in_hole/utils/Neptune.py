import warnings

warnings.filterwarnings('ignore', category=DeprecationWarning)  # message=r'.*jsonschema.RefResolver is deprecated.*')

import neptune  # noqa
from datetime import datetime  # noqa
import logging  # noqa
from stable_baselines3.common.callbacks import BaseCallback  # noqa
from pathlib import Path  # noqa
import numpy as np  # noqa

logger = logging.getLogger(__name__)


def new_neptune_run(neptune_cfg):
    # Create new neptune run
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


class NeptuneCallback(BaseCallback):
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

    def _checkpoint_path(self, checkpoint_type: str = '', extension: str = '') -> str:
        """
        Helper to get checkpoint path for each type of checkpoint.

        :param checkpoint_type: empty for the model, "replay_buffer_"
            or "vecnormalize_" for the other checkpoints.
        :param extension: Checkpoint file extension (zip for model, pkl for others)
        :return: Path to the checkpoint
        """
        # path = self.save_path / f'{self.name_prefix}_{checkpoint_type}{self.num_timesteps}_steps.{extension}'
        path = self.save_path / f'{self.name_prefix}_{checkpoint_type}.{extension}'

        return path

    def _on_step(self) -> bool:
        # Record environment data
        self.record_env_step()

        # If episode over, send data to neptune
        assert 'dones' in self.locals, '`dones` variable is not defined'
        episode_done = self.locals.get('dones', None)[0]
        if episode_done:
            # TODO: If multiple envs, need to check the ones that are done and send their data
            self.send_ep_to_neptune()
            self.n_episodes += np.sum(self.locals['dones']).item()

        # Save model
        self.save_model()

        return True

    def record_env_step(self):
        if not self.log_env:
            return

        # Observations
        if self.n_calls % self.env_log_freq == 0:
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
        # Episode observations
        if self.log_env:
            if self.n_episodes > self.neptune_n_episodes - 1:  # -1 bc of 0-index
                del_idx = self.n_episodes - self.neptune_n_episodes
                try:
                    del self.neptune_run[f'episode/{del_idx}']
                except neptune.exceptions.NeptuneException as e:
                    logger.warning(f'Could not delete episode {del_idx} from neptune: {e}')

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

        # Episode summary
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
            ep_logger[param].append(val)

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
        if self.n_calls % self.save_freq == 0:
            model_path = self._checkpoint_path(extension='zip')
            self.model.save(model_path)
            self.neptune_run['model_checkpoints/model'].upload(model_path.as_posix())

            if self.verbose >= 2:
                print(f'Saving model checkpoint to {model_path}')

            if (
                self.save_replay_buffer
                and hasattr(self.model, 'replay_buffer')
                and self.model.replay_buffer is not None
            ):
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
