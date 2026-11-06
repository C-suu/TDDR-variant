"DASC target mixing with predictive representation learning."

import copy
from dataclasses import dataclass
from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F

import buffer


@dataclass
class Hyperparameters:
    """Architecture and optimizer settings for representation learning."""

    zs_dim: int = 256
    enc_hdim: int = 256
    enc_activ: Callable = F.elu
    encoder_lr: float = 0.0003
    critic_activ: Callable = F.elu
    critic_lr: float = 0.001
    actor_activ: Callable = F.relu
    actor_lr: float = 0.001


def AvgL1Norm(x, eps=1e-08):
    """Normalize each feature vector by its mean absolute value."""
    return x / x.abs().mean(-1, keepdim=True).clamp(min=eps)


class Actor(nn.Module):
    """Deterministic actor using augmented state features."""

    def __init__(
        self, state_dim, action_dim, zs_dim=256, hdim=[400, 300], activ=F.relu
    ):
        super(Actor, self).__init__()
        self.activ = activ
        input_dim = state_dim + zs_dim
        self.l0 = nn.Linear(input_dim, hdim[0])
        self.l1 = nn.Linear(hdim[0], hdim[1])
        self.l2 = nn.Linear(hdim[1], action_dim)

    def forward(self, enhanced_state):
        a = self.activ(self.l0(enhanced_state))
        a = self.activ(self.l1(a))
        return torch.tanh(self.l2(a))


class Critic(nn.Module):
    """Action-value estimator using state and state-action features."""

    def __init__(self, state_dim, action_dim, zs_dim=256, hdim=[400, 300], activ=F.elu):
        super(Critic, self).__init__()
        self.activ = activ
        input_dim = state_dim + zs_dim + action_dim + zs_dim
        self.q1 = nn.Linear(input_dim, hdim[0])
        self.q2 = nn.Linear(hdim[0], hdim[1])
        self.q3 = nn.Linear(hdim[1], 1)

    def forward(self, enhanced_state, action, zsa):
        all_inputs = torch.cat([enhanced_state, action, zsa], 1)
        q = self.activ(self.q1(all_inputs))
        q = self.activ(self.q2(q))
        q = self.q3(q)
        return q

    def Q1(self, enhanced_state, action, zsa):
        all_inputs = torch.cat([enhanced_state, action, zsa], 1)
        q = self.activ(self.q1(all_inputs))
        q = self.activ(self.q2(q))
        q = self.q3(q)
        return q


class Encoder(nn.Module):
    """State and state-action encoders for next-state feature prediction."""

    def __init__(self, state_dim, action_dim, zs_dim=256, hdim=256, activ=F.elu):
        super(Encoder, self).__init__()
        self.state_dim = state_dim
        self.activ = activ
        self.zs1 = nn.Linear(state_dim, hdim)
        self.zs2 = nn.Linear(hdim, hdim)
        self.zs3 = nn.Linear(hdim, zs_dim)
        self.zsa1 = nn.Linear(zs_dim + action_dim, hdim)
        self.zsa2 = nn.Linear(hdim, hdim)
        self.zsa3 = nn.Linear(hdim, zs_dim)

    def zs(self, state):
        """Encode states with average-L1 normalization."""
        zs = self.activ(self.zs1(state))
        zs = self.activ(self.zs2(zs))
        zs = AvgL1Norm(self.zs3(zs))
        return zs

    def zsa(self, zs, action):
        """Predict next-state features from state features and actions."""
        zsa = self.activ(self.zsa1(torch.cat([zs, action], 1)))
        zsa = self.activ(self.zsa2(zsa))
        zsa = self.zsa3(zsa)
        return zsa


class DASC_R(object):
    """DASC target construction with learned representations."""

    def __init__(
        self,
        state_dim,
        action_dim,
        max_action,
        q_weight,
        batch_size=128,
        buffer_size=1000000,
        discount=0.99,
        target_update_rate=250,
        exploration_noise=0.1,
        tau=0.005,
        target_policy_noise=0.2,
        noise_clip=0.5,
        hdim=[400, 300],
        hp=Hyperparameters(),
    ):
        self.batch_size = batch_size
        self.buffer_size = buffer_size
        self.discount = discount
        self.target_update_rate = target_update_rate
        self.exploration_noise = exploration_noise
        self.tau = tau
        self.q_weight = q_weight
        self.target_policy_noise = target_policy_noise
        self.noise_clip = noise_clip
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.hp = hp
        self.actor1 = Actor(state_dim, action_dim, hp.zs_dim, hdim, hp.actor_activ).to(
            self.device
        )
        self.actor1_optimizer = torch.optim.Adam(
            self.actor1.parameters(), lr=hp.actor_lr
        )
        self.actor1_target = copy.deepcopy(self.actor1)
        self.actor2 = Actor(state_dim, action_dim, hp.zs_dim, hdim, hp.actor_activ).to(
            self.device
        )
        self.actor2_optimizer = torch.optim.Adam(
            self.actor2.parameters(), lr=hp.actor_lr
        )
        self.actor2_target = copy.deepcopy(self.actor2)
        self.critic1 = Critic(
            state_dim, action_dim, hp.zs_dim, hdim, hp.critic_activ
        ).to(self.device)
        self.critic1_optimizer = torch.optim.Adam(
            self.critic1.parameters(), lr=hp.critic_lr
        )
        self.critic1_target = copy.deepcopy(self.critic1)
        self.critic2 = Critic(
            state_dim, action_dim, hp.zs_dim, hdim, hp.critic_activ
        ).to(self.device)
        self.critic2_optimizer = torch.optim.Adam(
            self.critic2.parameters(), lr=hp.critic_lr
        )
        self.critic2_target = copy.deepcopy(self.critic2)
        self.encoder = Encoder(
            state_dim, action_dim, hp.zs_dim, hp.enc_hdim, hp.enc_activ
        ).to(self.device)
        self.encoder_optimizer = torch.optim.Adam(
            self.encoder.parameters(), lr=hp.encoder_lr
        )
        self.fixed_encoder = copy.deepcopy(self.encoder)
        self.fixed_encoder_target = copy.deepcopy(self.encoder)
        self.max_action = max_action
        self.training_steps = 0
        self.replay_buffer = buffer.ReplayBuffer(
            state_dim, action_dim, max_size=int(buffer_size)
        )

    def select_action(self, state, use_exploration=True):
        """Select an action and optionally add exploration noise."""
        with torch.no_grad():
            state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
            zs = self.fixed_encoder.zs(state)
            enhanced_state = torch.cat([state, zs], 1)
            action1 = self.actor1(enhanced_state)
            action2 = self.actor2(enhanced_state)
            zsa1 = self.fixed_encoder.zsa(zs, action1)
            zsa2 = self.fixed_encoder.zsa(zs, action2)
            q1 = self.critic1(enhanced_state, action1, zsa1)
            q2 = self.critic2(enhanced_state, action2, zsa2)
            action = action1 if q1 >= q2 else action2
            if use_exploration:
                action = action + torch.randn_like(action) * self.exploration_noise
            return action.clamp(-1, 1).cpu().data.numpy().flatten() * self.max_action

    def train(self, replay_buffer, batch_size=128):
        """Update both actor-critic pairs using replay samples."""
        self.train_one_q_and_pi(replay_buffer, True, batch_size=batch_size)
        self.train_one_q_and_pi(replay_buffer, False, batch_size=batch_size)

    def train_one_q_and_pi(self, replay_buffer, update_a1, batch_size=128):
        """Update the encoder, one actor-critic pair, and target networks."""
        self.training_steps += 1
        (state, action, next_state, reward, not_done) = replay_buffer.sample(batch_size)
        with torch.no_grad():
            next_zs = self.encoder.zs(next_state)
        zs = self.encoder.zs(state)
        pred_zs = self.encoder.zsa(zs, action)
        encoder_loss = F.mse_loss(pred_zs, next_zs)
        self.encoder_optimizer.zero_grad()
        encoder_loss.backward()
        self.encoder_optimizer.step()
        with torch.no_grad():
            fixed_target_zs = self.fixed_encoder_target.zs(next_state)
            enhanced_next_state = torch.cat([next_state, fixed_target_zs], 1)
            noise = (torch.randn_like(action) * self.target_policy_noise).clamp(
                -self.noise_clip, self.noise_clip
            )
            next_action1 = (self.actor1_target(enhanced_next_state) + noise).clamp(
                -1, 1
            )
            next_action2 = (self.actor2_target(enhanced_next_state) + noise).clamp(
                -1, 1
            )
            fixed_target_zsa1 = self.fixed_encoder_target.zsa(
                fixed_target_zs, next_action1
            )
            fixed_target_zsa2 = self.fixed_encoder_target.zsa(
                fixed_target_zs, next_action2
            )
            # Evaluate both target critics at each actor action.
            next_Q1_a1 = self.critic1_target(
                enhanced_next_state, next_action1, fixed_target_zsa1
            )
            next_Q2_a1 = self.critic2_target(
                enhanced_next_state, next_action1, fixed_target_zsa1
            )
            next_Q1_a2 = self.critic1_target(
                enhanced_next_state, next_action2, fixed_target_zsa2
            )
            next_Q2_a2 = self.critic2_target(
                enhanced_next_state, next_action2, fixed_target_zsa2
            )
            next_Q1 = torch.min(next_Q1_a1, next_Q2_a1)
            next_Q2 = torch.min(next_Q1_a2, next_Q2_a2)
            fixed_zs = self.fixed_encoder.zs(state)
            enhanced_state = torch.cat([state, fixed_zs], 1)
            fixed_zsa = self.fixed_encoder.zsa(fixed_zs, action)
            current_Q1 = self.critic1_target(enhanced_state, action, fixed_zsa)
            current_Q2 = self.critic2_target(enhanced_state, action, fixed_zsa)
            current_Q = torch.min(current_Q1, current_Q2)
            # Compare TD discrepancies for the candidate actions.
            Delta_1 = reward + not_done * self.discount * next_Q1 - current_Q
            Delta_2 = reward + not_done * self.discount * next_Q2 - current_Q
            # DASC target combination.
            next_Q3 = self.q_weight * next_Q1 + (1 - self.q_weight) * next_Q1_a2
            next_Q4 = self.q_weight * next_Q2 + (1 - self.q_weight) * next_Q1_a1
            # Select the mixture with the smaller absolute TD discrepancy.
            comparison_tensor = torch.abs(Delta_1) <= torch.abs(Delta_2)
            next_Q = torch.where(comparison_tensor, next_Q3, next_Q4)
            Q_target = reward + not_done * self.discount * next_Q
        # Update the selected actor-critic pair.
        if update_a1:
            current_Q1 = self.critic1(enhanced_state, action, fixed_zsa)
            critic1_loss = F.mse_loss(current_Q1, Q_target)
            self.critic1_optimizer.zero_grad()
            critic1_loss.backward()
            self.critic1_optimizer.step()
            actor1_action = self.actor1(enhanced_state)
            actor1_zsa = self.fixed_encoder.zsa(fixed_zs, actor1_action)
            Q = self.critic1(enhanced_state, actor1_action, actor1_zsa)
            actor1_loss = -Q.mean()
            self.actor1_optimizer.zero_grad()
            actor1_loss.backward()
            self.actor1_optimizer.step()
            for (param, target_param) in zip(
                self.critic1.parameters(), self.critic1_target.parameters()
            ):
                target_param.data.copy_(
                    self.tau * param.data + (1 - self.tau) * target_param.data
                )
            for (param, target_param) in zip(
                self.actor1.parameters(), self.actor1_target.parameters()
            ):
                target_param.data.copy_(
                    self.tau * param.data + (1 - self.tau) * target_param.data
                )
        else:
            current_Q2 = self.critic2(enhanced_state, action, fixed_zsa)
            critic2_loss = F.mse_loss(current_Q2, Q_target)
            self.critic2_optimizer.zero_grad()
            critic2_loss.backward()
            self.critic2_optimizer.step()
            actor2_action = self.actor2(enhanced_state)
            actor2_zsa = self.fixed_encoder.zsa(fixed_zs, actor2_action)
            Q = self.critic2(enhanced_state, actor2_action, actor2_zsa)
            actor2_loss = -Q.mean()
            self.actor2_optimizer.zero_grad()
            actor2_loss.backward()
            self.actor2_optimizer.step()
            for (param, target_param) in zip(
                self.critic2.parameters(), self.critic2_target.parameters()
            ):
                target_param.data.copy_(
                    self.tau * param.data + (1 - self.tau) * target_param.data
                )
            for (param, target_param) in zip(
                self.actor2.parameters(), self.actor2_target.parameters()
            ):
                target_param.data.copy_(
                    self.tau * param.data + (1 - self.tau) * target_param.data
                )
        # Refresh the fixed encoder snapshots at the configured interval.
        if self.training_steps % self.target_update_rate == 0:
            self.fixed_encoder_target.load_state_dict(self.fixed_encoder.state_dict())
            self.fixed_encoder.load_state_dict(self.encoder.state_dict())
