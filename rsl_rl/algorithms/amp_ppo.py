from __future__ import annotations

import torch
import torch.nn as nn
from itertools import chain
from typing import Any
from tensordict import TensorDict

from rsl_rl.algorithms.ppo import PPO
from rsl_rl.env import VecEnv
from rsl_rl.extensions import resolve_rnd_config, resolve_symmetry_config
from rsl_rl.models import MLPModel
from rsl_rl.storage import RolloutStorage, ReplayBuffer
from rsl_rl.utils import resolve_callable, resolve_obs_groups, resolve_optimizer

class AMPPPO(PPO):
    """
    AMP PPO logic adapted for RSL-RL v5.0.1.
    Inherits from standard v5.0.1 PPO and implements the complete Adversarial Motion Priors training loop,
    including full support for RND and Symmetry data augmentation logic natively supported in v5.0.1.
    """
    def __init__(
        self,
        actor: MLPModel,
        critic: MLPModel,
        discriminator: nn.Module,
        storage: RolloutStorage,
        amp_data,
        amp_normalizer,
        amp_replay_buffer_size: int = 100000,
        min_std: torch.Tensor | None = None,
        **kwargs
    ) -> None:
        # Initialize standard PPO (which handles RND, Symmetry, storage, standard optimizer prep)
        super().__init__(actor, critic, storage, **kwargs)

        if discriminator is None:
            raise ValueError("AMPPPO requires a valid discriminator instance.")
        if amp_data is None:
            raise ValueError("AMPPPO requires amp_data for expert transitions.")
        
        # Override optimizer to explicitly include discriminator parameters
        optimizer_name = kwargs.get("optimizer", "adam")
        optimizer_cls = resolve_optimizer(optimizer_name)
        self.optimizer = optimizer_cls(
            [
                {"params": self.actor.parameters(), "name": "actor"},
                {"params": self.critic.parameters(), "name": "critic"},
                {"params": discriminator.trunk.parameters(), "weight_decay": 1e-3, "name": "amp_trunk"},
                {"params": discriminator.amp_linear.parameters(), "weight_decay": 1e-1, "name": "amp_head"},
            ],
            lr=self.learning_rate,
        )

        self.discriminator = discriminator.to(self.device)
        self.amp_data = amp_data
        self.amp_normalizer = amp_normalizer
        self.amploss_coef = 1.0
        self.min_std = min_std

        self._last_amp_obs: torch.Tensor | None = None
        amp_obs_dim = int(self.discriminator.input_dim // 2)
        self.amp_storage = ReplayBuffer(amp_obs_dim, amp_replay_buffer_size, self.device)
        
    def act(self, obs: TensorDict, amp_obs: torch.Tensor | None = None) -> torch.Tensor:
        """Sample actions and store transition data including amp observations."""
        actions = super().act(obs)
        if amp_obs is not None:
            self._last_amp_obs = amp_obs.detach()
        return actions

    def process_env_step(
        self, obs: TensorDict, rewards: torch.Tensor, dones: torch.Tensor, extras: dict[str, torch.Tensor], amp_obs: torch.Tensor | None = None
    ) -> None:
        """Record one environment step and update the normalizers and amp buffers."""
        super().process_env_step(obs, rewards, dones, extras)
        if amp_obs is not None and self._last_amp_obs is not None:
            self.amp_storage.insert(self._last_amp_obs, amp_obs)
            self._last_amp_obs = amp_obs.detach()

    def update(self) -> dict[str, float]:
        """Run optimization epochs over stored batches and return mean losses with complete AMP integration."""
        mean_value_loss = 0
        mean_surrogate_loss = 0

        mean_entropy = 0
        mean_amp_loss = 0
        mean_grad_pen_loss = 0
        mean_policy_pred = 0
        mean_expert_pred = 0

        mean_rnd_loss = 0 if self.rnd else None
        mean_symmetry_loss = 0 if self.symmetry else None

        if self.actor.is_recurrent or self.critic.is_recurrent:
            generator = self.storage.recurrent_mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        else:
            generator = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)

        amp_policy_generator = self.amp_storage.feed_forward_generator(
            self.num_learning_epochs * self.num_mini_batches,
            self.storage.num_envs * self.storage.num_transitions_per_env // self.num_mini_batches,
        )
        amp_expert_generator = self.amp_data.feed_forward_generator(
            self.num_learning_epochs * self.num_mini_batches,
            self.storage.num_envs * self.storage.num_transitions_per_env // self.num_mini_batches,
        )

        for batch, sample_amp_policy, sample_amp_expert in zip(generator, amp_policy_generator, amp_expert_generator):
            original_batch_size = batch.observations.batch_size[0]
            rnd_loss = torch.zeros((), device=self.device)
            symmetry_loss = torch.zeros((), device=self.device)
            data_augmentation_func = None

            if self.normalize_advantage_per_mini_batch:
                with torch.no_grad():
                    batch.advantages = (batch.advantages - batch.advantages.mean()) / (batch.advantages.std() + 1e-8)  # type: ignore

            # Perform symmetric augmentation
            if self.symmetry and self.symmetry["use_data_augmentation"]:
                data_augmentation_func = self.symmetry["data_augmentation_func"]
                batch.observations, batch.actions = data_augmentation_func(
                    env=self.symmetry["_env"],
                    obs=batch.observations,
                    actions=batch.actions,
                )
                num_aug = int(batch.observations.batch_size[0] / original_batch_size)
                batch.old_actions_log_prob = batch.old_actions_log_prob.repeat(num_aug, 1)
                batch.values = batch.values.repeat(num_aug, 1)
                batch.advantages = batch.advantages.repeat(num_aug, 1)
                batch.returns = batch.returns.repeat(num_aug, 1)

            self.actor(
                batch.observations,
                masks=batch.masks,
                hidden_state=batch.hidden_states[0],
                stochastic_output=True,
            )
            actions_log_prob = self.actor.get_output_log_prob(batch.actions)  # type: ignore
            values = self.critic(batch.observations, masks=batch.masks, hidden_state=batch.hidden_states[1])
            distribution_params = tuple(p[:original_batch_size] for p in self.actor.output_distribution_params)
            entropy = self.actor.output_entropy[:original_batch_size]

            # KL divergence calculation
            if self.desired_kl is not None and self.schedule == "adaptive":
                with torch.inference_mode():
                    kl = self.actor.get_kl_divergence(batch.old_distribution_params, distribution_params)  # type: ignore
                    kl_mean = torch.mean(kl)

                    if self.is_multi_gpu:
                        torch.distributed.all_reduce(kl_mean, op=torch.distributed.ReduceOp.SUM)
                        kl_mean /= self.gpu_world_size

                    if self.gpu_global_rank == 0:
                        if kl_mean > self.desired_kl * 2.0:
                            self.learning_rate = max(1e-5, self.learning_rate / 1.5)
                        elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
                            self.learning_rate = min(1e-2, self.learning_rate * 1.5)

                    if self.is_multi_gpu:
                        lr_tensor = torch.tensor(self.learning_rate, device=self.device)
                        torch.distributed.broadcast(lr_tensor, src=0)
                        self.learning_rate = lr_tensor.item()

                    for param_group in self.optimizer.param_groups:
                        param_group["lr"] = self.learning_rate

            # PPO Core Losses
            ratio = torch.exp(actions_log_prob - torch.squeeze(batch.old_actions_log_prob))  # type: ignore
            surrogate = -torch.squeeze(batch.advantages) * ratio  # type: ignore
            surrogate_clipped = -torch.squeeze(batch.advantages) * torch.clamp(  # type: ignore
                ratio, 1.0 - self.clip_param, 1.0 + self.clip_param
            )
            surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

            if self.use_clipped_value_loss:
                value_clipped = batch.values + (values - batch.values).clamp(-self.clip_param, self.clip_param)
                value_losses = (values - batch.returns).pow(2)
                value_losses_clipped = (value_clipped - batch.returns).pow(2)
                value_loss = torch.max(value_losses, value_losses_clipped).mean()
            else:
                value_loss = (batch.returns - values).pow(2).mean()

            loss = surrogate_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy.mean()

            # Symmetry loss
            if self.symmetry:
                if not self.symmetry["use_data_augmentation"]:
                    data_augmentation_func = self.symmetry["data_augmentation_func"]
                    batch.observations, _ = data_augmentation_func(
                        obs=batch.observations, actions=None, env=self.symmetry["_env"]
                    )
                mean_actions = self.actor(batch.observations.detach().clone())
                action_mean_orig = mean_actions[:original_batch_size]
                _, actions_mean_symm = data_augmentation_func(  # type: ignore[misc]
                    obs=None, actions=action_mean_orig, env=self.symmetry["_env"]
                )
                mse_loss = torch.nn.MSELoss()
                symmetry_loss = mse_loss(
                    mean_actions[original_batch_size:], actions_mean_symm.detach()[original_batch_size:]
                )
                if self.symmetry["use_mirror_loss"]:
                    loss += self.symmetry["mirror_loss_coeff"] * symmetry_loss
                else:
                    symmetry_loss = symmetry_loss.detach()

            # RND loss
            if self.rnd:
                with torch.no_grad():
                    rnd_state = self.rnd.get_rnd_state(batch.observations[:original_batch_size])  # type: ignore
                    rnd_state = self.rnd.state_normalizer(rnd_state)
                predicted_embedding = self.rnd.predictor(rnd_state)
                target_embedding = self.rnd.target(rnd_state).detach()
                mseloss = torch.nn.MSELoss()
                rnd_loss = mseloss(predicted_embedding, target_embedding)

            # --- COMPLETING AMP DISCRIMINATOR LOSS ---
            policy_state, policy_next_state = sample_amp_policy
            expert_state, expert_next_state = sample_amp_expert
            
            if self.amp_normalizer is not None:
                with torch.no_grad():
                    policy_state = self.amp_normalizer.normalize_torch(policy_state, self.device)
                    policy_next_state = self.amp_normalizer.normalize_torch(policy_next_state, self.device)
                    expert_state = self.amp_normalizer.normalize_torch(expert_state, self.device)
                    expert_next_state = self.amp_normalizer.normalize_torch(expert_next_state, self.device)
            
            # Predict labels for (s, s') pairs
            policy_d = self.discriminator(torch.cat([policy_state, policy_next_state], dim=-1))
            expert_d = self.discriminator(torch.cat([expert_state, expert_next_state], dim=-1))
            
            # Expert wants to be pushed to 1, Policy pseudo-agent to -1
            expert_loss = torch.nn.MSELoss()(expert_d, torch.ones(expert_d.size(), device=self.device))
            policy_loss = torch.nn.MSELoss()(policy_d, -1 * torch.ones(policy_d.size(), device=self.device))
            amp_loss = 0.5 * (expert_loss + policy_loss)
            
            # Gradient penalty formulation
            grad_pen_loss = self.discriminator.compute_grad_pen(expert_state, expert_next_state, lambda_=10)
            
            # Add to total loss
            loss += self.amploss_coef * amp_loss + self.amploss_coef * grad_pen_loss

            # Optimization Steps
            self.optimizer.zero_grad()
            loss.backward()
            
            if self.rnd:
                self.rnd_optimizer.zero_grad()
                rnd_loss.backward()

            if self.is_multi_gpu:
                self.reduce_parameters()

            nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
            nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
            self.optimizer.step()

            if self.min_std is not None and hasattr(self.actor, "distribution") and self.actor.distribution is not None:
                if hasattr(self.actor.distribution, "std_param"):
                    with torch.no_grad():
                        min_std = self.min_std.to(self.device)
                        self.actor.distribution.std_param.data = torch.maximum(  # type: ignore[attr-defined]
                            self.actor.distribution.std_param.data, min_std
                        )
            
            if self.rnd_optimizer:
                self.rnd_optimizer.step()

            # Update AMP normalizers
            if self.amp_normalizer is not None:
                self.amp_normalizer.update(policy_state.detach().cpu().numpy())
                self.amp_normalizer.update(expert_state.detach().cpu().numpy())

            # Metrics
            mean_value_loss += value_loss.item()
            mean_surrogate_loss += surrogate_loss.item()
            mean_entropy += entropy.mean().item()
            mean_amp_loss += amp_loss.item()
            mean_grad_pen_loss += grad_pen_loss.item()
            mean_policy_pred += policy_d.mean().item()
            mean_expert_pred += expert_d.mean().item()
            if mean_rnd_loss is not None:
                mean_rnd_loss += rnd_loss.item()
            if mean_symmetry_loss is not None:
                mean_symmetry_loss += symmetry_loss.item()

        num_updates = self.num_learning_epochs * self.num_mini_batches
        mean_value_loss /= num_updates
        mean_surrogate_loss /= num_updates
        mean_entropy /= num_updates
        mean_amp_loss /= num_updates
        mean_grad_pen_loss /= num_updates
        mean_policy_pred /= num_updates
        mean_expert_pred /= num_updates
        if mean_rnd_loss is not None:
            mean_rnd_loss /= num_updates
        if mean_symmetry_loss is not None:
            mean_symmetry_loss /= num_updates
        
        self.storage.clear()

        loss_dict = {
            "value": mean_value_loss,
            "surrogate": mean_surrogate_loss,
            "entropy": mean_entropy,
            "amp": mean_amp_loss,
            "amp_grad_pen": mean_grad_pen_loss,
            "amp_policy_pred": mean_policy_pred,
            "amp_expert_pred": mean_expert_pred,
        }
        if self.rnd:
            loss_dict["rnd"] = mean_rnd_loss
        if self.symmetry:
            loss_dict["symmetry"] = mean_symmetry_loss
        
        return loss_dict

    def train_mode(self) -> None:
        """Set train mode for learnable models including discriminator."""
        super().train_mode()
        self.discriminator.train()

    def eval_mode(self) -> None:
        """Set evaluation mode for learnable models including discriminator."""
        super().eval_mode()
        self.discriminator.eval()

    def save(self) -> dict:
        """Return a dict of all models for saving, adding discriminator specifics."""
        saved_dict = super().save()
        saved_dict["discriminator_state_dict"] = self.discriminator.state_dict()
        return saved_dict

    def load(self, loaded_dict: dict, load_cfg: dict | None, strict: bool) -> bool:
        """Load specified models from a saved dict."""
        iteration = super().load(loaded_dict, load_cfg, strict)
        if load_cfg is None or load_cfg.get("discriminator"):
            self.discriminator.load_state_dict(loaded_dict["discriminator_state_dict"], strict=strict)
        return iteration

    def reduce_parameters(self) -> None:
        """Collect gradients from all GPUs including Discriminator and average them."""
        all_params = chain(self.actor.parameters(), self.critic.parameters(), self.discriminator.parameters())
        if self.rnd:
            all_params = chain(all_params, self.rnd.parameters())
            
        all_params = list(all_params)
        grads = [param.grad.view(-1) for param in all_params if param.grad is not None]
        if not grads:
            return
        all_grads = torch.cat(grads)
        torch.distributed.all_reduce(all_grads, op=torch.distributed.ReduceOp.SUM)
        all_grads /= self.gpu_world_size
        offset = 0
        for param in all_params:
            if param.grad is not None:
                numel = param.numel()
                param.grad.data.copy_(all_grads[offset : offset + numel].view_as(param.grad.data))
                offset += numel

    def broadcast_parameters(self) -> None:
        """Broadcast learnable model parameters to all GPUs including discriminator."""
        model_params: list[dict[str, Any]] = [
            self.actor.state_dict(),
            self.critic.state_dict(),
            self.discriminator.state_dict(),
        ]
        if self.rnd:
            model_params.append(self.rnd.predictor.state_dict())
        torch.distributed.broadcast_object_list(model_params, src=0)
        self.actor.load_state_dict(model_params[0])
        self.critic.load_state_dict(model_params[1])
        self.discriminator.load_state_dict(model_params[2])
        if self.rnd:
            self.rnd.predictor.load_state_dict(model_params[3])

    @staticmethod
    def construct_algorithm(
        obs: TensorDict,
        env: VecEnv,
        cfg: dict,
        device: str,
        discriminator: nn.Module | None = None,
        amp_data=None,
        amp_normalizer=None,
    ) -> AMPPPO:
        """Construct the AMPPPO algorithm with AMP specific dependencies."""
        if discriminator is None:
            raise ValueError("construct_algorithm requires a discriminator for AMPPPO.")
        alg_class: type[AMPPPO] = resolve_callable(cfg["algorithm"].pop("class_name"))
        actor_class: type[MLPModel] = resolve_callable(cfg["actor"].pop("class_name"))
        critic_class: type[MLPModel] = resolve_callable(cfg["critic"].pop("class_name"))

        # Drop optional config keys that some model classes (e.g. MLPModel) don't accept.
        if cfg["actor"].get("cnn_cfg") is None:
            cfg["actor"].pop("cnn_cfg", None)
        if cfg["critic"].get("cnn_cfg") is None:
            cfg["critic"].pop("cnn_cfg", None)

        default_sets = ["actor", "critic"]
        if "rnd_cfg" in cfg["algorithm"] and cfg["algorithm"]["rnd_cfg"] is not None:
            default_sets.append("rnd_state")
        cfg["obs_groups"] = resolve_obs_groups(obs, cfg["obs_groups"], default_sets)

        cfg["algorithm"] = resolve_rnd_config(cfg["algorithm"], obs, cfg["obs_groups"], env)
        cfg["algorithm"] = resolve_symmetry_config(cfg["algorithm"], env)

        actor: MLPModel = actor_class(obs, cfg["obs_groups"], "actor", env.num_actions, **cfg["actor"]).to(device)
        if cfg["algorithm"].pop("share_cnn_encoders", None):
            cfg["critic"]["cnns"] = actor.cnns
        critic: MLPModel = critic_class(obs, cfg["obs_groups"], "critic", 1, **cfg["critic"]).to(device)

        storage = RolloutStorage("rl", env.num_envs, cfg["num_steps_per_env"], obs, [env.num_actions], device)

        alg: AMPPPO = alg_class(
            actor=actor, 
            critic=critic, 
            discriminator=discriminator,
            storage=storage, 
            amp_data=amp_data,
            amp_normalizer=amp_normalizer,
            device=device, 
            **cfg["algorithm"], 
            multi_gpu_cfg=cfg["multi_gpu"]
        )

        return alg

