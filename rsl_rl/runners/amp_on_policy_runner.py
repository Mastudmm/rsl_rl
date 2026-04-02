from __future__ import annotations

import os
import time
import glob
import torch

from rsl_rl.algorithms import AMPPPO
from rsl_rl.env import VecEnv
from rsl_rl.models import MLPModel
from rsl_rl.runners.on_policy_runner import OnPolicyRunner
from rsl_rl.utils import check_nan, resolve_callable
from rsl_rl.utils.logger import Logger
from rsl_rl.utils import AMPLoader, Normalizer, validate_amp_pipeline

class AmpOnPolicyRunner(OnPolicyRunner):
    """On-policy runner for AMP reinforcement learning algorithms."""

    alg: AMPPPO
    """The actor-critic algorithm with AMP."""

    def __init__(self, env: VecEnv, train_cfg: dict, log_dir: str | None = None, device: str = "cpu") -> None:
        """Construct the runner, algorithm, and logging stack with AMP additions."""
        self.env = env
        if not hasattr(self.env, "get_amp_obs_for_expert_trans"):
            raise TypeError(
                "AmpOnPolicyRunner requires an AMP-capable env wrapper with "
                "get_amp_obs_for_expert_trans()."
            )
        if not hasattr(self.env, "amp_enabled"):
            raise TypeError(
                "AmpOnPolicyRunner requires env.amp_enabled toggle. "
                "Please use RslRlVecEnvWrapper with amp_enabled support."
            )
        self.env.amp_enabled = True
        self.cfg = train_cfg
        self.device = device
        
        self._configure_multi_gpu()
        obs = self.env.get_observations()

        raw_motion_files = train_cfg.get("amp_motion_files")
        if isinstance(raw_motion_files, str):
            raw_motion_files = [raw_motion_files]

        if not raw_motion_files:
            motion_files = sorted(glob.glob("dataset/*.json"))
            if not motion_files:
                motion_files = glob.glob("datasets/motion_amp_expert/*")
        else:
            motion_files = []
            for pattern in raw_motion_files:
                matched = glob.glob(pattern)
                if matched:
                    motion_files.extend(matched)
                else:
                    motion_files.append(pattern)
            motion_files = sorted(set(motion_files))

        if not motion_files:
            raise ValueError(
                "No AMP motion files found. Set train_cfg['amp_motion_files'] or place files in "
                "dataset/ (e.g. dataset/*.json) or datasets/motion_amp_expert/."
            )

        amp_num_preload_transitions = int(train_cfg.get("amp_num_preload_transitions", 1_000_000))
        amp_reward_coef = float(train_cfg.get("amp_reward_coef", 0.3))
        amp_discr_hidden_dims = train_cfg.get("amp_discr_hidden_dims", [1024, 512])
        amp_task_reward_lerp = float(train_cfg.get("amp_task_reward_lerp", 0.0))
        amp_joint_pos_mode = str(train_cfg.get("amp_joint_pos_mode", "relative"))
        amp_joint_pos_offset = train_cfg.get("amp_joint_pos_offset")

        amp_data = AMPLoader(
            device,
            time_between_frames=self.env.step_dt,
            preload_transitions=True,
            num_preload_transitions=amp_num_preload_transitions,
            motion_files=motion_files,
            joint_pos_mode=amp_joint_pos_mode,
            joint_pos_offset=amp_joint_pos_offset,
        )

        if bool(train_cfg.get("amp_preflight_check", True)):
            preflight = validate_amp_pipeline(
                self.env,
                motion_files,
                amp_data,
                strict=bool(train_cfg.get("amp_preflight_strict", True)),
                max_files_to_scan=int(train_cfg.get("amp_preflight_max_files", 8)),
                expected_obs_dim=train_cfg.get("amp_expected_obs_dim"),
            )
            print(
                "[AMP preflight] OK "
                f"(env_dim={preflight['env_obs_dim']}, expert_dim={preflight['expert_obs_dim']}, "
                f"files={preflight['num_motion_files']}, scanned={preflight['scanned_motion_files']}, "
                f"warnings={len(preflight['warnings'])})"
            )

        amp_normalizer = Normalizer(amp_data.observation_dim)
        
        # Instantiate Discriminator (Assumes Discriminator model is provided by rsl_rl.models)
        from rsl_rl.models.discriminator import Discriminator
        discriminator = Discriminator(
            amp_data.observation_dim * 2,
            amp_reward_coef,
            amp_discr_hidden_dims,
            device,
            amp_task_reward_lerp,
        ).to(device)

        alg_class: type[AMPPPO] = resolve_callable(self.cfg["algorithm"]["class_name"])
        self.alg = alg_class.construct_algorithm(obs, self.env, self.cfg, self.device, discriminator, amp_data, amp_normalizer)

        self.logger = Logger(
            log_dir=log_dir,
            cfg=self.cfg,
            env_cfg=self.env.cfg,
            num_envs=self.env.num_envs,
            is_distributed=self.is_distributed,
            gpu_world_size=self.gpu_world_size,
            gpu_global_rank=self.gpu_global_rank,
            device=self.device,
        )

        self.current_learning_iteration = 0

    def learn(self, num_learning_iterations: int, init_at_random_ep_len: bool = False) -> None:
        """Run the learning loop for the specified number of iterations with AMP observations."""
        # Overrides standard PPO learn loop to include amp_obs processing and logging
        if init_at_random_ep_len:
            self.env.episode_length_buf = torch.randint_like(self.env.episode_length_buf, high=int(self.env.max_episode_length))
        
        obs = self.env.get_observations().to(self.device)
        # Fetch initial AMP obs from the environment
        amp_obs = self.env.get_amp_obs_for_expert_trans().to(self.device)
        
        self.alg.train_mode()
        
        if self.is_distributed:
            self.alg.broadcast_parameters()
            
        self.logger.init_logging_writer()

        start_it = self.current_learning_iteration
        total_it = start_it + num_learning_iterations
        for it in range(start_it, total_it):
            start = time.time()
            with torch.inference_mode():
                for _ in range(self.cfg["num_steps_per_env"]):
                    actions = self.alg.act(obs, amp_obs)
                    obs, rewards, dones, extras = self.env.step(actions.to(self.env.device))
                    next_amp_obs = extras.get("amp_obs")
                    if next_amp_obs is None:
                        next_amp_obs = self.env.get_amp_obs_for_expert_trans()
                    
                    if self.cfg.get("check_for_nan", True):
                        check_nan(obs, rewards, dones)
                        
                    obs = obs.to(self.device)
                    rewards = rewards.to(self.device)
                    dones = dones.to(self.device)
                    next_amp_obs = next_amp_obs.to(self.device)

                    # Handle terminal transitions for AMP discriminator targets.
                    next_amp_obs_with_term = next_amp_obs.clone()
                    reset_env_ids = getattr(self.env, "reset_env_ids", None)
                    if reset_env_ids is not None and reset_env_ids.numel() > 0:
                        terminal_amp_states = self.env.get_amp_obs_for_expert_trans()[reset_env_ids].to(self.device)
                        next_amp_obs_with_term[reset_env_ids] = terminal_amp_states

                    # Replace task reward with AMP-combined reward used by PPO update.
                    rewards = self.alg.discriminator.predict_amp_reward(
                        amp_obs,
                        next_amp_obs_with_term,
                        rewards,
                        normalizer=self.alg.amp_normalizer,
                    )[0]

                    self.alg.process_env_step(obs, rewards, dones, extras, next_amp_obs_with_term)
                    amp_obs = next_amp_obs.clone()
                    
                    intrinsic_rewards = self.alg.intrinsic_rewards if self.cfg["algorithm"].get("rnd_cfg") else None
                    self.logger.process_env_step(rewards, dones, extras, intrinsic_rewards)

                stop = time.time()
                collect_time = stop - start
                start = stop
                self.alg.compute_returns(obs)

            loss_dict = self.alg.update()
            
            stop = time.time()
            learn_time = stop - start
            self.current_learning_iteration = it
            
            self.logger.log(
                it=it,
                start_it=start_it,
                total_it=total_it,
                collect_time=collect_time,
                learn_time=learn_time,
                loss_dict=loss_dict,
                learning_rate=self.alg.learning_rate,
                action_std=self.alg.get_policy().output_std,
                rnd_weight=self.alg.rnd.weight if self.cfg["algorithm"].get("rnd_cfg") else None,
            )

            if self.logger.writer is not None and it % self.cfg["save_interval"] == 0:
                log_dir = self.logger.log_dir
                if log_dir is not None:
                    self.save(os.path.join(log_dir, f"model_{it}.pt"))

        if self.logger.writer is not None:
            log_dir = self.logger.log_dir
            if log_dir is not None:
                self.save(os.path.join(log_dir, f"model_{self.current_learning_iteration}.pt"))
            self.logger.stop_logging_writer()
