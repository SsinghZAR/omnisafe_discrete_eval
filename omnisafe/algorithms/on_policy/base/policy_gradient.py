# Copyright 2023 OmniSafe Team. All Rights Reserved.
# Additional Modifications Copyright YEAR YOUR NAME/ORGANIZATION for morality evaluation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Implementation of the Policy Gradient algorithm with optional periodic morality evaluation."""

from __future__ import annotations

import os # Added for path joining in morality eval
import time
from typing import Any

import torch
import torch.nn as nn
import pandas as pd # Added for morality eval results saving
import numpy as np  # Added for morality eval policy function

from rich.progress import track
from torch.nn.utils.clip_grad import clip_grad_norm_
from torch.utils.data import DataLoader, TensorDataset

from omnisafe.adapter import OnPolicyAdapter
from omnisafe.algorithms import registry
from omnisafe.algorithms.base_algo import BaseAlgo
from omnisafe.common.buffer import VectorOnPolicyBuffer
from omnisafe.common.logger import Logger
from omnisafe.models.actor_critic.constraint_actor_critic import ConstraintActorCritic
from omnisafe.utils import distributed

# Optional imports for morality_gym evaluation
_MORALITY_GYM_AVAILABLE = False
try:
    from experiments.cost_function import Cost
    from experiments.baselines.common.setup import make_experiment
    from experiments.baselines.common.evaluate import eval_multi_variants
    from morality_gym.setup.setup import make as env_mt_make
    from morality_gym.wrappers.evaluation_wrapper import EvaluationEnvWrapper
    from experiments.baselines.common.evaluate import eval_multi_variants
    # from morality_gym.utils.common import join_paths # Using os.path.join directly
    _MORALITY_GYM_AVAILABLE = True
except ImportError:
    # This is not a critical error if morality evaluation is not used.
    # A warning will be printed in _init if morality eval is configured but imports failed.
    pass 


@registry.register
# pylint: disable-next=too-many-instance-attributes,too-few-public-methods,line-too-long
class PolicyGradient(BaseAlgo):
    """The Policy Gradient algorithm.

    References:
        - Title: Policy Gradient Methods for Reinforcement Learning with Function Approximation
        - Authors: Richard S. Sutton, David McAllester, Satinder Singh, Yishay Mansour.
        - URL: `PG <https://proceedings.neurips.cc/paper/1999/file64d828b85b0bed98e80ade0a5c43b0f-Paper.pdf>`_
    """

    def _init_env(self) -> None:
        """Initialize the environment.

        OmniSafe uses :class:`omnisafe.adapter.OnPolicyAdapter` to adapt the environment to the
        algorithm.

        User can customize the environment by inheriting this method.

        Examples:
            >>> def _init_env(self) -> None:
            ...     self._env = CustomAdapter()

        Raises:
            AssertionError: If the number of steps per epoch is not divisible by the number of
                environments.
        """
        self._env: OnPolicyAdapter = OnPolicyAdapter(
            self._env_id,
            self._cfgs.train_cfgs.vector_env_nums,
            self._seed,
            self._cfgs,
        )
        assert (self._cfgs.algo_cfgs.steps_per_epoch) % (
            distributed.world_size() * self._cfgs.train_cfgs.vector_env_nums
        ) == 0, 'The number of steps per epoch is not divisible by the number of environments.'
        self._steps_per_epoch: int = (
            self._cfgs.algo_cfgs.steps_per_epoch
            // distributed.world_size()
            // self._cfgs.train_cfgs.vector_env_nums
        )

    def _init_model(self) -> None:
        """Initialize the model.

        OmniSafe uses :class:`omnisafe.models.actor_critic.constraint_actor_critic.ConstraintActorCritic`
        as the default model.

        User can customize the model by inheriting this method.

        Examples:
            >>> def _init_model(self) -> None:
            ...     self._actor_critic = CustomActorCritic()
        """
        self._actor_critic: ConstraintActorCritic = ConstraintActorCritic(
            obs_space=self._env.observation_space,
            act_space=self._env.action_space,
            model_cfgs=self._cfgs.model_cfgs,
            epochs=self._cfgs.train_cfgs.epochs,
        ).to(self._device)

        if distributed.world_size() > 1:
            distributed.sync_params(self._actor_critic)

        if self._cfgs.model_cfgs.exploration_noise_anneal:
            self._actor_critic.set_annealing(
                epochs=[0, self._cfgs.train_cfgs.epochs],
                std=self._cfgs.model_cfgs.std_range,
            )

    def _init(self) -> None:
        """The initialization of the algorithm.

        User can define the initialization of the algorithm by inheriting this method.

        Examples:
            >>> def _init(self) -> None:
            ...     super()._init()
            ...     self._buffer = CustomBuffer()
            ...     self._model = CustomModel()
        """
        self._buf: VectorOnPolicyBuffer = VectorOnPolicyBuffer(
            obs_space=self._env.observation_space,
            act_space=self._env.action_space,
            size=self._steps_per_epoch,
            gamma=self._cfgs.algo_cfgs.gamma,
            lam=self._cfgs.algo_cfgs.lam,
            lam_c=self._cfgs.algo_cfgs.lam_c,
            advantage_estimator=self._cfgs.algo_cfgs.adv_estimation_method,
            standardized_adv_r=self._cfgs.algo_cfgs.standardized_rew_adv,
            standardized_adv_c=self._cfgs.algo_cfgs.standardized_cost_adv,
            penalty_coefficient=self._cfgs.algo_cfgs.penalty_coef,
            num_envs=self._cfgs.train_cfgs.vector_env_nums,
            device=self._device,
        )

    def _init_log(self) -> None:
        """Log info about epoch.

        +-----------------------+----------------------------------------------------------------------+
        | Things to log         | Description                                                          |
        +=======================+======================================================================+
        | Train/Epoch           | Current epoch.                                                       |
        +-----------------------+----------------------------------------------------------------------+
        | Metrics/EpCost        | Average cost of the epoch.                                           |
        +-----------------------+----------------------------------------------------------------------+
        | Metrics/EpRet         | Average return of the epoch.                                         |
        +-----------------------+----------------------------------------------------------------------+
        | Metrics/EpLen         | Average length of the epoch.                                         |
        +-----------------------+----------------------------------------------------------------------+
        | Values/reward         | Average value in :meth:`rollout` (from critic network) of the epoch. |
        +-----------------------+----------------------------------------------------------------------+
        | Values/cost           | Average cost in :meth:`rollout` (from critic network) of the epoch.  |
        +-----------------------+----------------------------------------------------------------------+
        | Values/Adv            | Average reward advantage of the epoch.                               |
        +-----------------------+----------------------------------------------------------------------+
        | Loss/Loss_pi          | Loss of the policy network.                                          |
        +-----------------------+----------------------------------------------------------------------+
        | Loss/Loss_cost_critic | Loss of the cost critic network.                                     |
        +-----------------------+----------------------------------------------------------------------+
        | Train/Entropy         | Entropy of the policy network.                                       |
        +-----------------------+----------------------------------------------------------------------+
        | Train/StopIters       | Number of iterations of the policy network.                          |
        +-----------------------+----------------------------------------------------------------------+
        | Train/PolicyRatio     | Ratio of the policy network.                                         |
        +-----------------------+----------------------------------------------------------------------+
        | Train/LR              | Learning rate of the policy network.                                 |
        +-----------------------+----------------------------------------------------------------------+
        | Misc/Seed             | Seed of the experiment.                                              |
        +-----------------------+----------------------------------------------------------------------+
        | Misc/TotalEnvSteps    | Total steps of the experiment.                                       |
        +-----------------------+----------------------------------------------------------------------+
        | Time                  | Total time.                                                          |
        +-----------------------+----------------------------------------------------------------------+
        | FPS                   | Frames per second of the epoch.                                      |
        +-----------------------+----------------------------------------------------------------------+
        """
        self._logger = Logger(
            output_dir=self._cfgs.logger_cfgs.log_dir,
            exp_name=self._cfgs.exp_name,
            seed=self._cfgs.seed,
            use_tensorboard=self._cfgs.logger_cfgs.use_tensorboard,
            use_wandb=self._cfgs.logger_cfgs.use_wandb,
            config=self._cfgs,
        )

        what_to_save: dict[str, Any] = {}
        what_to_save['pi'] = self._actor_critic.actor
        if self._cfgs.algo_cfgs.obs_normalize:
            obs_normalizer = self._env.save()['obs_normalizer']
            what_to_save['obs_normalizer'] = obs_normalizer
        self._logger.setup_torch_saver(what_to_save)
        self._logger.torch_save()

        self._logger.register_key('Metrics/EpRet', window_length=50)
        self._logger.register_key('Metrics/EpCost', window_length=50)
        self._logger.register_key('Metrics/EpLen', window_length=50)

        self._logger.register_key('Train/Epoch')
        self._logger.register_key('Train/Entropy')
        self._logger.register_key('Train/KL')
        self._logger.register_key('Train/StopIter')
        self._logger.register_key('Train/PolicyRatio', min_and_max=True)
        self._logger.register_key('Train/LR')
        if self._cfgs.model_cfgs.actor_type == 'gaussian_learning':
            self._logger.register_key('Train/PolicyStd')

        self._logger.register_key('TotalEnvSteps')

        # log information about actor
        self._logger.register_key('Loss/Loss_pi', delta=True)
        self._logger.register_key('Value/Adv')

        # log information about critic
        self._logger.register_key('Loss/Loss_reward_critic', delta=True)
        self._logger.register_key('Value/reward')

        if self._cfgs.algo_cfgs.use_cost:
            # log information about cost critic
            self._logger.register_key('Loss/Loss_cost_critic', delta=True)
            self._logger.register_key('Value/cost')

        self._logger.register_key('Time/Total')
        self._logger.register_key('Time/Rollout')
        self._logger.register_key('Time/Update')
        self._logger.register_key('Time/Epoch')
        self._logger.register_key('Time/FPS')

        # --- Initialization for optional periodic morality evaluation ---
        # Look for config nested under algo_cfgs now
        algo_cfgs_obj = getattr(self._cfgs, 'algo_cfgs', None)
        self._morality_eval_cfgs = getattr(algo_cfgs_obj, 'morality_eval_cfgs', {}) if algo_cfgs_obj else {}
        
        self._morality_eval_freq: int = self._morality_eval_cfgs.get('eval_freq_epochs', 0)
        self._morality_exp_name: str | None = self._morality_eval_cfgs.get('experiment_name', None)
        self._intermediate_morality_results: list[dict] = []

        if self._morality_eval_freq > 0:
            if not _MORALITY_GYM_AVAILABLE:
                self._logger.log("ERROR: Morality Gym components not available, but morality evaluation was configured. Disabling periodic evaluation.")
                self._morality_eval_freq = 0 # Disable
            elif not self._morality_exp_name:
                raise ValueError("morality_eval_cfgs.experiment_name is required when morality_eval_freq_epochs > 0.")
            else:
                if hasattr(self._cfgs, 'env_id') and isinstance(self._cfgs.env_id, str):
                    try:
                        # env_id must be "ExperimentName::TreeId::RepeatIdx"
                        exp_name_from_env, self._eval_morality_tree_id, self._eval_repeat_idx_str = self._cfgs.env_id.split('::')
                        self._eval_repeat_idx = int(self._eval_repeat_idx_str)
                    except ValueError:
                        raise ValueError(
                            f"Invalid env_id format ('{self._cfgs.env_id}'). Expected 'ExpName::TreeId::RepeatIdx' for morality evaluation."
                        )

                    if self._morality_exp_name != exp_name_from_env:
                        raise ValueError(
                            f"morality_eval_cfgs.experiment_name ('{self._morality_exp_name}') must match the training experiment name ('{exp_name_from_env}') parsed from env_id."
                        )

                    # Validate the experiment config exists; raise if not found
                    try:
                        _ = make_experiment(self._morality_exp_name)  # type: ignore
                    except Exception as e:
                        raise FileNotFoundError(
                            f"Morality evaluation experiment '{self._morality_exp_name}' could not be loaded via make_experiment: {e}"
                        ) from e

                    self._logger.log(
                        f"Periodic morality evaluation enabled: Freq={self._morality_eval_freq} epochs, ExpName={self._morality_exp_name}, Evaluating on variant part of {self._cfgs.env_id}"
                    )
                else:
                    raise ValueError("env_id not found or not a string in configs; required for morality evaluation.")

            if self._morality_eval_freq > 0: # If still enabled after checks
                self._logger.register_key('Time/MoralityEval')

    def learn(self) -> tuple[float, float, float]:
        """This is main function for algorithm update.

        It is divided into the following steps:

        - :meth:`rollout`: collect interactive data from environment.
        - :meth:`update`: perform actor/critic updates.
        - :meth:`log`: epoch/update information for visualization and terminal log print.

        Returns:
            ep_ret: Average episode return in final epoch.
            ep_cost: Average episode cost in final epoch.
            ep_len: Average episode length in final epoch.
        """
        start_time = time.time()
        self._logger.log('INFO: Start training')

        for epoch in range(self._cfgs.train_cfgs.epochs):
            epoch_time = time.time()

            rollout_time = time.time()
            self._env.rollout(
                steps_per_epoch=self._steps_per_epoch,
                agent=self._actor_critic,
                buffer=self._buf,
                logger=self._logger,
            )
            self._logger.store({'Time/Rollout': time.time() - rollout_time})

            update_time = time.time()
            self._update()
            self._logger.store({'Time/Update': time.time() - update_time})

            if self._cfgs.model_cfgs.exploration_noise_anneal:
                self._actor_critic.annealing(epoch)

            if self._cfgs.model_cfgs.actor.lr is not None:
                self._actor_critic.actor_scheduler.step()

            self._logger.store(
                {
                    'TotalEnvSteps': (epoch + 1) * self._cfgs.algo_cfgs.steps_per_epoch,
                    'Time/FPS': self._cfgs.algo_cfgs.steps_per_epoch / (time.time() - epoch_time),
                    'Time/Total': (time.time() - start_time),
                    'Time/Epoch': (time.time() - epoch_time),
                    'Train/Epoch': epoch,
                    'Train/LR': 0.0
                    if self._cfgs.model_cfgs.actor.lr is None
                    else self._actor_critic.actor_scheduler.get_last_lr()[0],
                },
            )

            self._logger.dump_tabular()

            # --- Optional: Periodic Morality Evaluation ---
            if self._morality_eval_freq > 0: # Check if enabled first
                perform_eval_this_epoch = (epoch + 1) % self._morality_eval_freq == 0
                is_last_epoch = epoch == self._cfgs.train_cfgs.epochs - 1
                if perform_eval_this_epoch or is_last_epoch:
                    self._logger.log(f"INFO: Performing morality evaluation at epoch {epoch+1}")
                    eval_start_time = time.time()
                    self._perform_morality_evaluation(current_epoch=epoch)
                    self._logger.store({'Time/MoralityEval': time.time() - eval_start_time})
                    # self._logger.dump_tabular() # Optionally dump again if morality eval uses self._logger.store for its metrics

                            # --- Save intermediate morality results if any ---
                    if self._intermediate_morality_results:
                        try:
                            results_df = pd.DataFrame(self._intermediate_morality_results)
                            intermediate_eval_path = os.path.join(self._logger.log_dir, 'intermediate_morality_evals.csv')
                            results_df.to_csv(intermediate_eval_path, index=False)
                            self._logger.log(f"INFO: Intermediate morality evaluation results saved to {intermediate_eval_path}")
                        except Exception as e:
                            self._logger.log(f"ERROR: Could not save intermediate morality evaluation results: {e}")
            # save model to disk
            if (epoch + 1) % self._cfgs.logger_cfgs.save_model_freq == 0 or \
               (epoch == self._cfgs.train_cfgs.epochs - 1): # Ensure last epoch model is saved
                self._logger.torch_save()

        ep_ret = self._logger.get_stats('Metrics/EpRet')[0]
        ep_cost = self._logger.get_stats('Metrics/EpCost')[0]
        ep_len = self._logger.get_stats('Metrics/EpLen')[0]

        self._logger.close()

        return ep_ret, ep_cost, ep_len

    def _update(self) -> None:
        """Update actor, critic.

        -  Get the ``data`` from buffer

        .. hint::

            +----------------+------------------------------------------------------------------+
            | obs            | ``observation`` sampled from buffer.                             |
            +================+==================================================================+
            | act            | ``action`` sampled from buffer.                                  |
            +----------------+------------------------------------------------------------------+
            | target_value_r | ``target reward value`` sampled from buffer.                     |
            +----------------+------------------------------------------------------------------+
            | target_value_c | ``target cost value`` sampled from buffer.                       |
            +----------------+------------------------------------------------------------------+
            | logp           | ``log probability`` sampled from buffer.                         |
            +----------------+------------------------------------------------------------------+
            | adv_r          | ``estimated advantage`` (e.g. **GAE**) sampled from buffer.      |
            +----------------+------------------------------------------------------------------+
            | adv_c          | ``estimated cost advantage`` (e.g. **GAE**) sampled from buffer. |
            +----------------+------------------------------------------------------------------+


        -  Update value net by :meth:`_update_reward_critic`.
        -  Update cost net by :meth:`_update_cost_critic`.
        -  Update policy net by :meth:`_update_actor`.

        The basic process of each update is as follows:

        #. Get the data from buffer.
        #. Shuffle the data and split it into mini-batch data.
        #. Get the loss of network.
        #. Update the network by loss.
        #. Repeat steps 2, 3 until the number of mini-batch data is used up.
        #. Repeat steps 2, 3, 4 until the KL divergence violates the limit.
        """
        data = self._buf.get()
        obs, act, logp, target_value_r, target_value_c, adv_r, adv_c = (
            data['obs'],
            data['act'],
            data['logp'],
            data['target_value_r'],
            data['target_value_c'],
            data['adv_r'],
            data['adv_c'],
        )

        original_obs = obs
        old_distribution = self._actor_critic.actor(obs)

        dataloader = DataLoader(
            dataset=TensorDataset(obs, act, logp, target_value_r, target_value_c, adv_r, adv_c),
            batch_size=self._cfgs.algo_cfgs.batch_size,
            shuffle=True,
        )

        update_counts = 0
        final_kl = 0.0

        for i in track(range(self._cfgs.algo_cfgs.update_iters), description='Updating...'):
            for (
                obs,
                act,
                logp,
                target_value_r,
                target_value_c,
                adv_r,
                adv_c,
            ) in dataloader:
                self._update_reward_critic(obs, target_value_r)
                if self._cfgs.algo_cfgs.use_cost:
                    self._update_cost_critic(obs, target_value_c)
                self._update_actor(obs, act, logp, adv_r, adv_c)

            new_distribution = self._actor_critic.actor(original_obs)

            kl = (
                torch.distributions.kl.kl_divergence(old_distribution, new_distribution)
                .sum(-1, keepdim=True)
                .mean()
            )
            kl = distributed.dist_avg(kl)

            final_kl = kl.item()
            update_counts += 1

            if self._cfgs.algo_cfgs.kl_early_stop and kl.item() > self._cfgs.algo_cfgs.target_kl:
                self._logger.log(f'Early stopping at iter {i + 1} due to reaching max kl')
                break

        self._logger.store(
            {
                'Train/StopIter': update_counts,  # pylint: disable=undefined-loop-variable
                'Value/Adv': adv_r.mean().item(),
                'Train/KL': final_kl,
            },
        )

    def _update_reward_critic(self, obs: torch.Tensor, target_value_r: torch.Tensor) -> None:
        r"""Update value network under a double for loop.

        The loss function is ``MSE loss``, which is defined in ``torch.nn.MSELoss``.
        Specifically, the loss function is defined as:

        .. math::

            L = \frac{1}{N} \sum_{i=1}^N (\hat{V} - V)^2

        where :math:`\hat{V}` is the predicted cost and :math:`V` is the target cost.

        #. Compute the loss function.
        #. Add the ``critic norm`` to the loss function if ``use_critic_norm`` is ``True``.
        #. Clip the gradient if ``use_max_grad_norm`` is ``True``.
        #. Update the network by loss function.

        Args:
            obs (torch.Tensor): The ``observation`` sampled from buffer.
            target_value_r (torch.Tensor): The ``target_value_r`` sampled from buffer.
        """
        self._actor_critic.reward_critic_optimizer.zero_grad()
        loss = nn.functional.mse_loss(self._actor_critic.reward_critic(obs)[0], target_value_r)

        if self._cfgs.algo_cfgs.use_critic_norm:
            for param in self._actor_critic.reward_critic.parameters():
                loss += param.pow(2).sum() * self._cfgs.algo_cfgs.critic_norm_coef

        loss.backward()

        if self._cfgs.algo_cfgs.use_max_grad_norm:
            clip_grad_norm_(
                self._actor_critic.reward_critic.parameters(),
                self._cfgs.algo_cfgs.max_grad_norm,
            )
        distributed.avg_grads(self._actor_critic.reward_critic)
        self._actor_critic.reward_critic_optimizer.step()

        self._logger.store({'Loss/Loss_reward_critic': loss.mean().item()})

    def _update_cost_critic(self, obs: torch.Tensor, target_value_c: torch.Tensor) -> None:
        r"""Update value network under a double for loop.

        The loss function is ``MSE loss``, which is defined in ``torch.nn.MSELoss``.
        Specifically, the loss function is defined as:

        .. math::

            L = \frac{1}{N} \sum_{i=1}^N (\hat{V} - V)^2

        where :math:`\hat{V}` is the predicted cost and :math:`V` is the target cost.

        #. Compute the loss function.
        #. Add the ``critic norm`` to the loss function if ``use_critic_norm`` is ``True``.
        #. Clip the gradient if ``use_max_grad_norm`` is ``True``.
        #. Update the network by loss function.

        Args:
            obs (torch.Tensor): The ``observation`` sampled from buffer.
            target_value_c (torch.Tensor): The ``target_value_c`` sampled from buffer.
        """
        self._actor_critic.cost_critic_optimizer.zero_grad()
        loss = nn.functional.mse_loss(self._actor_critic.cost_critic(obs)[0], target_value_c)

        if self._cfgs.algo_cfgs.use_critic_norm:
            for param in self._actor_critic.cost_critic.parameters():
                loss += param.pow(2).sum() * self._cfgs.algo_cfgs.critic_norm_coef

        loss.backward()

        if self._cfgs.algo_cfgs.use_max_grad_norm:
            clip_grad_norm_(
                self._actor_critic.cost_critic.parameters(),
                self._cfgs.algo_cfgs.max_grad_norm,
            )
        distributed.avg_grads(self._actor_critic.cost_critic)
        self._actor_critic.cost_critic_optimizer.step()

        self._logger.store({'Loss/Loss_cost_critic': loss.mean().item()})

    def _update_actor(  # pylint: disable=too-many-arguments
        self,
        obs: torch.Tensor,
        act: torch.Tensor,
        logp: torch.Tensor,
        adv_r: torch.Tensor,
        adv_c: torch.Tensor,
    ) -> None:
        """Update policy network under a double for loop.

        #. Compute the loss function.
        #. Clip the gradient if ``use_max_grad_norm`` is ``True``.
        #. Update the network by loss function.

        .. warning::
            For some ``KL divergence`` based algorithms (e.g. TRPO, CPO, etc.),
            the ``KL divergence`` between the old policy and the new policy is calculated.
            And the ``KL divergence`` is used to determine whether the update is successful.
            If the ``KL divergence`` is too large, the update will be terminated.

        Args:
            obs (torch.Tensor): The ``observation`` sampled from buffer.
            act (torch.Tensor): The ``action`` sampled from buffer.
            logp (torch.Tensor): The ``log_p`` sampled from buffer.
            adv_r (torch.Tensor): The ``reward_advantage`` sampled from buffer.
            adv_c (torch.Tensor): The ``cost_advantage`` sampled from buffer.
        """
        adv = self._compute_adv_surrogate(adv_r, adv_c)
        loss = self._loss_pi(obs, act, logp, adv)
        self._actor_critic.actor_optimizer.zero_grad()
        loss.backward()
        if self._cfgs.algo_cfgs.use_max_grad_norm:
            clip_grad_norm_(
                self._actor_critic.actor.parameters(),
                self._cfgs.algo_cfgs.max_grad_norm,
            )
        distributed.avg_grads(self._actor_critic.actor)
        self._actor_critic.actor_optimizer.step()

    def _compute_adv_surrogate(  # pylint: disable=unused-argument
        self,
        adv_r: torch.Tensor,
        adv_c: torch.Tensor,
    ) -> torch.Tensor:
        """Compute surrogate loss.

        Policy Gradient only use reward advantage.

        Args:
            adv_r (torch.Tensor): The ``reward_advantage`` sampled from buffer.
            adv_c (torch.Tensor): The ``cost_advantage`` sampled from buffer.

        Returns:
            The advantage function of reward to update policy network.
        """
        return adv_r

    def _loss_pi(
        self,
        obs: torch.Tensor,
        act: torch.Tensor,
        logp: torch.Tensor,
        adv: torch.Tensor,
    ) -> torch.Tensor:
        r"""Computing pi/actor loss.

        In Policy Gradient, the loss is defined as:

        .. math::

            L = -\underset{s_t \sim \rho_{\theta}}{\mathbb{E}} [
                \sum_{t=0}^T ( \frac{\pi^{'}_{\theta}(a_t|s_t)}{\pi_{\theta}(a_t|s_t)} )
                 A^{R}_{\pi_{\theta}}(s_t, a_t)
            ]

        where :math:`\pi_{\theta}` is the policy network, :math:`\pi^{'}_{\theta}`
        is the new policy network, :math:`A^{R}_{\pi_{\theta}}(s_t, a_t)` is the advantage.

        Args:
            obs (torch.Tensor): The ``observation`` sampled from buffer.
            act (torch.Tensor): The ``action`` sampled from buffer.
            logp (torch.Tensor): The ``log probability`` of action sampled from buffer.
            adv (torch.Tensor): The ``advantage`` processed. ``reward_advantage`` here.

        Returns:
            The loss of pi/actor.
        """
        distribution = self._actor_critic.actor(obs)
        logp_ = self._actor_critic.actor.log_prob(act)
        ratio = torch.exp(logp_ - logp)
        loss = -(ratio * adv).mean()
        entropy = distribution.entropy().mean().item()
        if self._cfgs.model_cfgs.actor_type == 'gaussian_learning':
            self._logger.store({'Train/PolicyStd': self._actor_critic.actor.std})
        self._logger.store(
            {
                'Train/Entropy': entropy,
                'Train/PolicyRatio': ratio,
                'Loss/Loss_pi': loss.mean().item(),
            },
        )
        return loss

    # +++ Methods for Morality Evaluation (added) +++
    def _perform_morality_evaluation(self, current_epoch: int) -> None:
        """
        Performs morality evaluation using eval_multi_variants and logs the results.
        This method is called periodically during training if configured.
        """
        if not _MORALITY_GYM_AVAILABLE: # Should have been caught in _init_log, but double check
            self._logger.log("CRITICAL_ERROR: _perform_morality_evaluation called but Morality Gym components not available.")
            return
        
        self._logger.log(f"--- Starting Morality Evaluation for Epoch {current_epoch + 1} ---")
        
        # Ensure necessary attributes were set in _init_log
        if not (self._morality_exp_name and hasattr(self, '_eval_morality_tree_id') and hasattr(self, '_eval_repeat_idx')):
            self._logger.log("WARNING: Morality evaluation skipped due to missing configuration (exp_name, tree_id, or repeat_idx for eval variant).")
            return

        eval_env = None # For finally block
        try:
            # Still need make_experiment to get config details
            all_variant_make_configs, final_eval_args, learn_eval_multi_kwargs, seeds = make_experiment(self._morality_exp_name) # type: ignore
        except Exception as e:
            self._logger.log(f"ERROR: Failed during make_experiment('{self._morality_exp_name}') for morality eval: {e}")
            return

        # Config for the specific variant the agent is training on
        current_variant_key = (self._eval_morality_tree_id, self._eval_repeat_idx)
        if current_variant_key not in all_variant_make_configs:
            self._logger.log(f"ERROR: Variant key {current_variant_key} not found in make_configs from make_experiment('{self._morality_exp_name}'). Cannot perform periodic evaluation.")
            return
        specific_variant_config = all_variant_make_configs[current_variant_key]
        eval_base_env_id = specific_variant_config['env_id'] 
        eval_env_kwargs = specific_variant_config['env_kwargs'] 
        eval_morality_tree_id_for_make = specific_variant_config['morality_tree_id']

        try:
            # Create the specific base environment and morality tree
            eval_env, eval_mt = env_mt_make(
                env_id=eval_base_env_id,
                morality_tree_id=eval_morality_tree_id_for_make,
                env_kwargs=eval_env_kwargs
            ) # type: ignore
        except Exception as e:
            self._logger.log(f"ERROR: Failed to create base env/mt for morality eval: {e}")
            return 
            
        self._logger.log(f"Morality Tree for evaluation: {eval_morality_tree_id_for_make if eval_mt else 'None'}")

        # Define policy function using current actor
        def omnisafe_policy_fn(obs_original):
            if isinstance(obs_original, dict):
                if "observation" in obs_original and isinstance(obs_original["observation"], np.ndarray):
                    obs_flat = obs_original["observation"].ravel()
                else:
                    # Basic fallback for dict observations if specific key is missing
                    obs_flat = np.concatenate([v.flatten() if isinstance(v, np.ndarray) else np.array([v]).flatten() for v in obs_original.values()]).astype(np.float32)
            else:
                obs_flat = obs_original.astype(np.float32)

            obs_tensor = torch.as_tensor(obs_flat, dtype=torch.float32, device=self._device).unsqueeze(0)
            action_tensor = self._actor_critic.actor.predict(obs_tensor,deterministic=True)
            action_item = action_tensor.cpu().numpy().item()
            return action_item

        # Extract evaluation parameters from the loaded experiment config's final eval section
        try:
            max_steps = final_eval_args['max_episode_steps']
            n_repeats_for_avg = final_eval_args['n_mt_repeats'] # Number of episodes to average over
            handle_trunc = final_eval_args['handle_trunc']
        except KeyError as e:
             self._logger.log(f"ERROR: Missing expected key {e} in final_eval_args from make_experiment. Cannot run evaluate_morality_metric.")
             if eval_env is not None: eval_env.close()
             return

        try:
            # Call evaluate_morality_metric directly on the specific eval_mt and eval_env
            # Pass reset_kwargs={}, assuming eval_env is already configured correctly.

            #TODO SET COST NORMLAISTION THROUGH ENV CONFIGS.
            eval_cost_obj = Cost(eval_mt,normalise_cost=True)
            wrapped_eval_env = EvaluationEnvWrapper(eval_env, eval_cost_obj)
            wrapped_eval_env.all_episode_costs = []
            wrapped_eval_env.reset()

            agr_info_keys = ["cost"]
            morality_metric_learn, morality_functions_learn, avg_returns_learn, info_learn = eval_multi_variants(
               policy=omnisafe_policy_fn, 
                env=wrapped_eval_env,#eval_env,
                morality_tree=eval_mt,
                agr_info_keys=agr_info_keys,
                **learn_eval_multi_kwargs,
            ) # type: ignore
            
            all_costs = wrapped_eval_env.all_episode_costs
            avg_cost = np.mean(all_costs) if all_costs else 0.0

            # Log/Store results
            self._logger.log(f"Epoch {current_epoch + 1} Morality Eval - Avg Return: {avg_returns_learn}, Morality Metric: {morality_metric_learn}, Avg Cost: {avg_cost}")
            
            result_summary = {
                "epoch": current_epoch + 1,
                "avg_return_morality_eval": avg_returns_learn,
                "morality_metric_eval": morality_metric_learn,
                "avg_cost_morality_eval": avg_cost,
                **morality_functions_learn 
            }
            self._intermediate_morality_results.append(result_summary)

        except Exception as e:
            self._logger.log(f"ERROR: Exception during evaluate_morality_metric: {e}")
            import traceback
            self._logger.log(traceback.format_exc())
            #exit()
        finally:
            if eval_env is not None:
                eval_env.close() # type: ignore
            self._logger.log(f"--- Finished Morality Evaluation for Epoch {current_epoch + 1} ---")

    # --- End of Morality Evaluation Methods ---
