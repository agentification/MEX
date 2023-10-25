import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Implementation of Twin Delayed Deep Deterministic Policy Gradients (TD3)
# Paper: https://arxiv.org/abs/1802.09477


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()

        self.l1 = nn.Linear(state_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, action_dim)

        self.max_action = max_action

    def forward(self, state):
        a = F.relu(self.l1(state))
        a = F.relu(self.l2(a))
        return self.max_action * torch.tanh(self.l3(a))


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()

        # Q1 architecture
        self.l1 = nn.Linear(state_dim + action_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, 1)

        # Q2 architecture
        self.l4 = nn.Linear(state_dim + action_dim, 256)
        self.l5 = nn.Linear(256, 256)
        self.l6 = nn.Linear(256, 1)

    def forward(self, state, action):
        sa = torch.cat([state, action], 1)

        q1 = F.relu(self.l1(sa))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)

        q2 = F.relu(self.l4(sa))
        q2 = F.relu(self.l5(q2))
        q2 = self.l6(q2)
        return q1, q2

    def Q1(self, state, action):
        sa = torch.cat([state, action], 1)

        q1 = F.relu(self.l1(sa))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)
        return q1


class IO(object):
    def __init__(
            self,
            state_dim,
            action_dim,
            max_action,
            discount=0.99,
            tau=0.005,
            policy_noise=0.2,
            noise_clip=0.5,
            policy_freq=2,
            eta=0.001,
            use_baseline=False,
    ):

        self.actor = Actor(state_dim, action_dim, max_action).to(device)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=3e-4)

        self.critic = Critic(state_dim, action_dim).to(device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=3e-4)

        self.max_action = max_action
        self.discount = discount
        self.tau = tau
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_freq = policy_freq
        self.eta = eta
        self.use_baseline = use_baseline
        print("use baseline", self.use_baseline)
        self.total_it = 0

    def select_action(self, state):
        state = torch.FloatTensor(state.reshape(1, -1)).to(device)
        return self.actor(state).cpu().data.numpy().flatten()

    def train(self, replay_buffer, batch_size=256):
        self.total_it += 1
        info = {}
        # Sample replay buffer
        state, action, next_state, reward, not_done = replay_buffer.sample(batch_size)

        with torch.no_grad():
            # Select action according to policy and add clipped noise
            noise = (
                    torch.randn_like(action) * self.policy_noise
            ).clamp(-self.noise_clip, self.noise_clip)

            next_action = (
                    self.actor_target(next_state) + noise
            ).clamp(-self.max_action, self.max_action)

            # Compute the target Q value
            target_Q1, target_Q2 = self.critic_target(next_state, next_action)
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = reward + not_done * self.discount * target_Q

        # Get current Q estimates
        current_Q1, current_Q2 = self.critic(state, action)
        info['train/q_value_0'] = current_Q1.mean()

        # Compute critic loss
        mse_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)
        if self.use_baseline:
            random_action = torch.rand(action.size()).to(device)*2.-1. # range (-1,1)
            pi_action= self.actor(state)
            pi_action_next = self.actor(next_state)
            Q_pi_1, Q_pi_2 = self.critic(state, pi_action)
            Q_pi_next_1, Q_pi_next_2 = self.critic(state, pi_action_next)
            Q_random_1, Q_random_2 = self.critic(state, random_action)
            # random_Q1, random_Q2 = self.critic(state, random_action)
            info['train/random_q_value_0'] = Q_random_1.mean()
            info['train/q_pi_value_0'] = Q_pi_1.mean()
            info['train/q_pi_next_value_0'] = Q_random_1.mean()
            cat_Q1 = torch.concat([current_Q1, Q_pi_next_1, Q_random_1],dim=1)
            cat_Q2 = torch.concat([current_Q1, Q_pi_next_2, Q_random_2],dim=1)
            # cat_Q1 = torch.concat([Q_pi_1, Q_pi_next_1, Q_random_1],dim=1)
            # cat_Q2 = torch.concat([Q_pi_2, Q_pi_next_2, Q_random_2],dim=1)
            # print(cat_Q1.shape, cat_Q2.shape)
            info['train/q_cat_logsumexp_0'] = torch.logsumexp(cat_Q1, dim=1).mean()
            info['train/q_cat_std_0'] = torch.std(cat_Q1, dim=1).mean()
            # delta_Q1 = torch.logsumexp(cat_Q1, dim=1) - current_Q1
            # delta_Q2 = torch.logsumexp(cat_Q2, dim=1) - current_Q2
            delta_Q1 = Q_pi_1 - torch.logsumexp(cat_Q1, dim=1)
            delta_Q2 = Q_pi_2 - torch.logsumexp(cat_Q2, dim=1)
            optimistic_loss = self.eta * (delta_Q1 + delta_Q2).mean()
        else:
            optimistic_loss = self.eta * (current_Q1 + current_Q2).mean()
        critic_loss = mse_loss - optimistic_loss

        info['train/q_mse_loss'] = mse_loss
        info['train/q_optimistic_loss'] = optimistic_loss
        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Delayed policy updates
        if self.total_it % self.policy_freq == 0:

            # Compute actor losse
            actor_loss = -self.critic.Q1(state, self.actor(state)).mean()

            # Optimize the actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # Update the frozen target models
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        else:
            actor_loss = None

        return critic_loss, actor_loss, info

    def save(self, filename):
        torch.save(self.critic.state_dict(), filename + "_critic")
        torch.save(self.critic_optimizer.state_dict(), filename + "_critic_optimizer")

        torch.save(self.actor.state_dict(), filename + "_actor")
        torch.save(self.actor_optimizer.state_dict(), filename + "_actor_optimizer")

    def load(self, filename):
        self.critic.load_state_dict(torch.load(filename + "_critic"))
        self.critic_optimizer.load_state_dict(torch.load(filename + "_critic_optimizer"))
        self.critic_target = copy.deepcopy(self.critic)

        self.actor.load_state_dict(torch.load(filename + "_actor"))
        self.actor_optimizer.load_state_dict(torch.load(filename + "_actor_optimizer"))
        self.actor_target = copy.deepcopy(self.actor)
