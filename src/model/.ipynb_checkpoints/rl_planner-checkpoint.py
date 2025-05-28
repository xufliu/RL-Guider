import torch
import numpy as np
import torch.nn as nn
import copy
import random
import torch.nn.functional as F
import torch.optim as optim
import collections
from torch.distributions import Categorical 
from torch.nn.utils import clip_grad_norm_

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity)

    def add(self, root_state, curr_state, action, reward_tuple, next_state, done):
        self.buffer.append((root_state, curr_state, action, reward_tuple, next_state, done))
        
    def sample(self, batch_size):
        transitions = random.sample(self.buffer, batch_size)
        root_state, curr_state, action, reward_tuple, next_state, done = zip(*transitions)
        return np.array(root_state), np.array(curr_state), action, reward_tuple, np.array(next_state), done
    
    def size(self):
        return len(self.buffer)

def hidden_init(layer):
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return (-lim, lim)

class Actor(nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, action_dim)
        self.softmax = nn.Softmax(dim=-1)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        action_probs = self.softmax(x)
        return action_probs
    
    def evaluate(self, x, epsilon=1e-6):
        action_probs = self.forward(x)
        dist = Categorical(action_probs)
        action = dist.sample().to(x.device)
        z = action_probs == 0.0
        z = z.float() * 1e-8
        log_action_probabilities = torch.log(action_probs + z)
        return action.detach().cpu(), action_probs, log_action_probabilities
    
    def get_action(self, x):
        action_probs = self.forward(x)
        dist = Categorical(action_probs)
        action = dist.sample().to(x.device)
        z = action_probs == 0.0
        z = z.float() * 1e-8
        log_action_probabilities = torch.log(action_probs + z)
        return action.detach().cpu(), action_probs, log_action_probabilities
    
    # def get_det_action(self, x):
    #     action_probs = self.forward(x)
    #     dist = Categorical(action_probs)
    #     action = dist.sample().to(x.device)
    #     return action.detach().cpu()
    
    def get_det_action(self, x, mask_id):
        action_probs = self.forward(x)
        mask = torch.zeros_like(action_probs)
        mask[mask_id] = 1
        action_probs = action_probs * mask
        action_probs /= (action_probs.sum()+1e-7)
        action = torch.argmax(action_probs)
        # dist = Categorical(action_probs)
        # action = dist.sample().to(x.device)
        return action.detach().cpu()
    
class Critic(nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim, seed=1):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, action_dim)
        self.reset_parameters()

    def reset_parameters(self):
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(*hidden_init(self.fc3))
        self.fc4.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x

class RL_Planner(nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim, actor_lr, critic_lr, alpha_lr, target_entropy, tau, gamma, device):
        super(RL_Planner, self).__init__()
        self.state_dim = state_dim
        self.hidden_sim = hidden_dim
        self.action_dim = action_dim
        self.device = device
        self.gamma = gamma
        self.tau = tau
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.target_entropy = target_entropy
        self.log_alpha = torch.tensor([0.0], requires_grad=True)
        self.alpha = self.log_alpha.exp().detach()
        self.alpha_optimizer = optim.Adam(params=[self.log_alpha], lr=alpha_lr) 

        # CQL params
        self.with_lagrange = False
        self.temp = 1.0
        self.cql_weight = 1.0
        self.clip_grad_param = 1
        self.target_action_gap = 0.0
        self.cql_log_alpha = torch.zeros(1, requires_grad=True)
        self.cql_alpha_optimizer = optim.Adam(params=[self.cql_log_alpha], lr=alpha_lr) 

        # Actor Network
        self.actor = Actor(state_dim, hidden_dim, action_dim).to(device)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)

        # Critic Network (w/ Target Network)
        self.critic_1 = Critic(state_dim, hidden_dim, action_dim, seed=1).to(device)
        self.critic_2 = Critic(state_dim, hidden_dim, action_dim, seed=2).to(device)

        assert self.critic_1.parameters() != self.critic_2.parameters()
        
        self.critic_1_target = Critic(state_dim, hidden_dim, action_dim, seed=3).to(device)
        self.critic_1_target.load_state_dict(self.critic_1.state_dict())

        self.critic_2_target = Critic(state_dim, hidden_dim, action_dim).to(device)
        self.critic_2_target.load_state_dict(self.critic_2.state_dict())

        self.critic_1_optimizer = optim.Adam(self.critic_1.parameters(), lr=critic_lr)
        self.critic_2_optimizer = optim.Adam(self.critic_2.parameters(), lr=critic_lr)
        self.softmax = nn.Softmax(dim=-1)

    # def get_action(self, state):
    #     state = torch.from_numpy(state).float().to(self.device)
        
    #     with torch.no_grad():
    #         action = self.actor.get_det_action(state)
    #     return action.numpy()
    
    def get_action(self, state, mask_id):
        state = torch.from_numpy(state).float().to(self.device)
        
        with torch.no_grad():
            action = self.actor.get_det_action(state, mask_id)
        return action.numpy()
    
    def calc_policy_loss(self, states, alpha):
        _, action_probs, log_pis = self.actor.evaluate(states)

        q1 = self.critic_1(states)   
        q2 = self.critic_2(states)
        min_Q = torch.min(q1,q2)
        actor_loss = (action_probs * (alpha.to(self.device) * log_pis - min_Q )).sum(1).mean()
        log_action_pi = torch.sum(log_pis * action_probs, dim=1)
        return actor_loss, log_action_pi
        
    def soft_update(self, net, target_net):
        for param_target, param in zip(target_net.parameters(), net.parameters()):
            param_target.data.copy_(param_target.data * (1.0 - self.tau) + param.data * self.tau)

    def learn(self, step, experiences, gamma, d=1):
        states, actions, rewards, next_states, dones = experiences
    
        current_alpha = copy.deepcopy(self.alpha)
        actor_loss, log_pis = self.calc_policy_loss(states, current_alpha)
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        # Compute alpha loss
        alpha_loss = - (self.log_alpha.exp() * (log_pis.cpu() + self.target_entropy).detach().cpu()).mean()
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()
        self.alpha = self.log_alpha.exp().detach()

        # ---------------------------- update critic ---------------------------- #
        # Get predicted next-state actions and Q values from target models
        with torch.no_grad():
            _, action_probs, log_pis = self.actor.evaluate(next_states)
            Q_target1_next = self.critic_1_target(next_states)
            Q_target2_next = self.critic_2_target(next_states)
            Q_target_next = action_probs * (torch.min(Q_target1_next, Q_target2_next) - self.alpha.to(self.device) * log_pis)

            # Compute Q targets for current states (y_i)
            Q_targets = rewards + (gamma * (1 - dones) * Q_target_next.sum(dim=1).unsqueeze(-1)) 


        # Compute critic loss
        q1 = self.critic_1(states)
        q2 = self.critic_2(states)
        
        q1_ = q1.gather(1, actions.long())
        q2_ = q2.gather(1, actions.long())
        
        critic1_loss = 0.5 * F.mse_loss(q1_, Q_targets)
        critic2_loss = 0.5 * F.mse_loss(q2_, Q_targets)
        
        cql1_scaled_loss = torch.logsumexp(q1, dim=1).mean() - q1.mean()
        cql2_scaled_loss = torch.logsumexp(q2, dim=1).mean() - q2.mean()
        
        cql_alpha_loss = torch.FloatTensor([0.0])
        cql_alpha = torch.FloatTensor([0.0])
        if self.with_lagrange:
            cql_alpha = torch.clamp(self.cql_log_alpha.exp(), min=0.0, max=1000000.0).to(self.device)
            cql1_scaled_loss = cql_alpha * (cql1_scaled_loss - self.target_action_gap)
            cql2_scaled_loss = cql_alpha * (cql2_scaled_loss - self.target_action_gap)

            self.cql_alpha_optimizer.zero_grad()
            cql_alpha_loss = (- cql1_scaled_loss - cql2_scaled_loss) * 0.5 
            cql_alpha_loss.backward(retain_graph=True)
            self.cql_alpha_optimizer.step()
        
        total_c1_loss = critic1_loss + cql1_scaled_loss
        total_c2_loss = critic2_loss + cql2_scaled_loss
        
        
        # Update critics
        # critic 1
        self.critic_1_optimizer.zero_grad()
        total_c1_loss.backward(retain_graph=True)
        clip_grad_norm_(self.critic_1.parameters(), self.clip_grad_param)
        self.critic_1_optimizer.step()
        # critic 2
        self.critic_2_optimizer.zero_grad()
        total_c2_loss.backward()
        clip_grad_norm_(self.critic_2.parameters(), self.clip_grad_param)
        self.critic_2_optimizer.step()

        # ----------------------- update target networks ----------------------- #
        self.soft_update(self.critic_1, self.critic_1_target)
        self.soft_update(self.critic_2, self.critic_2_target)
        
        return actor_loss.item(), alpha_loss.item(), critic1_loss.item(), critic2_loss.item(), cql1_scaled_loss.item(), cql2_scaled_loss.item(), current_alpha, cql_alpha_loss.item(), cql_alpha.item()


def compute_advantage(gamma, lmbda, td_delta):
    td_delta = td_delta.detach().cpu()
    advantage_list = []
    advantage = 0.0
    for delta in td_delta[:-1]:
        advantage = gamma * lmbda * advantage + delta
        advantage_list.append(advantage)
    advantage_list.reverse()
    return torch.tensor(advantage_list, dtype=torch.float)

class PolicyNet(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(PolicyNet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.fc4 = torch.nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        # print("x shape:", x.shape, "dtype:", x.dtype)
        return F.softmax(self.fc4(x), dim=-1)

class ValueNet(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim):
        super(ValueNet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.fc4 = torch.nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return self.fc4(x)


# class PPO(nn.Module):
#     def __init__(self, state_dim, hidden_dim, action_dim, actor_lr, critic_lr, lmbda, epochs, eps, gamma, device):
#         super(PPO, self).__init__()
#         self.actor = PolicyNet(state_dim=state_dim, hidden_dim=hidden_dim, action_dim=action_dim).to(device)
#         self.critic = ValueNet(state_dim=state_dim, hidden_dim=hidden_dim).to(device)
#         self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
#         self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)
#         self.gamma = gamma
#         self.lmbda = lmbda
#         self.epochs = epochs
#         self.eps = eps
#         self.device = device
    
#     def get_action(self, state, mask_id):
#         with torch.no_grad():
#             state = torch.from_numpy(state).float().to(self.device)
            
#             # state = torch.tensor([state], dtype=torch.float).to(self.device)
#             logits = self.actor(state)
#             # mask = torch.zeros_like(probs)
#             # mask = torch.ones_like(probs) * float('-inf')
#             # mask[mask_id] = 0
#             # masked_logits = logits + mask
#             # mask[mask_id] = 1
#             # mask = mask.to(probs.device)
#             # print("probs shape:", probs.shape, "dtype:", probs.dtype)
#             # print("mask shape:", mask.shape, "dtype:", mask.dtype)
#             # probs = probs * mask
#             # probs /= (probs.sum()+1e-7)
#             action_dist = torch.distributions.Categorical(logits=logits)
#             action = action_dist.sample()
#         return action.item()
        
    
#     def update(self, transition_dict):
#         states = torch.tensor(transition_dict['states'], dtype=torch.float).to(self.device)
#         actions = torch.tensor(transition_dict['actions']).view(-1, 1).to(self.device)
#         rewards = torch.tensor(transition_dict['rewards'], dtype=torch.float).view(-1, 1).to(self.device)
#         next_states = torch.tensor(transition_dict['next_states'], dtype=torch.float).to(self.device)
#         dones = torch.tensor(transition_dict['dones'], dtype=torch.float).view(-1, 1).to(self.device)
#         td_target = rewards + self.gamma * self.critic(next_states) * (1 - dones)
#         td_delta = td_target - self.critic(states)
#         advantage = compute_advantage(self.gamma, self.lmbda, td_delta.cpu()).to(self.device)
#         old_log_probs = torch.log(self.actor(states).gather(1, actions)).detach()

#         for _ in range(self.epochs):
#             log_probs = torch.log(self.actor(states).gather(1, actions))
#             ratio = torch.exp(log_probs - old_log_probs)
#             surr1 = ratio * advantage
#             surr2 = torch.clamp(ratio, 1 - self.eps, 1 + self.eps) * advantage
#             actor_loss = torch.mean(-torch.min(surr1, surr2))
#             critic_loss = torch.mean(F.mse_loss(self.critic(states), td_target.detach()))
#             self.actor_optimizer.zero_grad()
#             self.critic_optimizer.zero_grad()
#             actor_loss.backward()
#             critic_loss.backward()
#             self.actor_optimizer.step()
#             self.critic_optimizer.step()



class PPO(nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim, actor_lr, critic_lr, lmbda, epochs, eps, gamma, device):
        super(PPO, self).__init__()
        self.actor = PolicyNet(state_dim, hidden_dim, action_dim).to(device)
        self.critic = ValueNet(state_dim, hidden_dim).to(device)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_lr)
        self.gamma = gamma
        self.lmbda = lmbda
        self.epochs = epochs
        self.eps = eps
        self.device = device

    def get_action(self, state, mask_id=None):
        with torch.no_grad():
            state = torch.FloatTensor(state).to(self.device)
            logits = self.actor(state)
            
            # 可选：处理动作掩码（如某些动作不可用）
            if mask_id is not None:
                mask = torch.ones_like(logits) * -1e8
                mask[mask_id] = 0
                logits = logits + mask

            dist = Categorical(logits=logits)
            action = dist.sample()
            return action.item()

    def update(self, transition_dict):
        states = torch.FloatTensor(transition_dict['states']).to(self.device)
        actions = torch.LongTensor(transition_dict['actions']).view(-1, 1).to(self.device)
        rewards = torch.FloatTensor(transition_dict['rewards']).view(-1, 1).to(self.device)
        next_states = torch.FloatTensor(transition_dict['next_states']).to(self.device)
        dones = torch.FloatTensor(transition_dict['dones']).view(-1, 1).to(self.device)

        # 计算 TD 目标和优势函数
        with torch.no_grad():
            td_target = rewards + self.gamma * self.critic(next_states) * (1 - dones)
            td_delta = td_target - self.critic(states)
            advantage = compute_advantage(self.gamma, self.lmbda, td_delta.cpu()).to(self.device)
            old_logits = self.actor(states)
            old_dist = Categorical(logits=old_logits)
            old_log_probs = old_dist.log_prob(actions.squeeze()).view(-1, 1)

        # 多轮优化
        for _ in range(self.epochs):
            # 计算新的策略分布
            logits = self.actor(states)
            dist = Categorical(logits=logits)
            log_probs = dist.log_prob(actions.squeeze()).view(-1, 1)
            entropy = dist.entropy().mean()

            # 重要性采样比率
            ratio = torch.exp(log_probs - old_log_probs.detach())
            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1 - self.eps, 1 + self.eps) * advantage
            actor_loss = -torch.min(surr1, surr2).mean() - 0.01 * entropy  # 熵正则化

            # 价值函数损失
            critic_loss = F.mse_loss(self.critic(states), td_target.detach())

            # 梯度裁剪和更新
            self.actor_optimizer.zero_grad()
            self.critic_optimizer.zero_grad()
            actor_loss.backward()
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)  # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
            self.actor_optimizer.step()
            self.critic_optimizer.step()