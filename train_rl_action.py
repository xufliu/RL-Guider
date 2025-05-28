from transformers import RobertaTokenizerFast, RobertaForMaskedLM, DataCollatorWithPadding
from transformers import AutoModel, AutoTokenizer
import numpy as np
import pickle
import os
import sys
import ast
import time
import copy
from openai import OpenAI
import argparse
from tqdm import tqdm
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal, Categorical
from pathlib import Path
import matplotlib.pyplot as plt
from torch.nn.utils import clip_grad_norm_
import torch.optim.lr_scheduler as lr_scheduler

sys.path.append("src")
from model.rl_planner import ReplayBuffer, RL_Planner
from utils.rl_utils import get_all_general_action_list, get_general_action_list
from utils.tool import cal_logP, fstr, parse, is_valid_smiles, calculate_tanimoto_similarity, get_task_info, get_prop_function
from search.reward.reward_function import get_reward
from llm.prompt_template import system_prompt, get_generation_prompt_template

# client = OpenAI(api_key="sk-c587a4923103469eaf8224f4287ef9d4", base_url="https://api.deepseek.com")
# client = OpenAI(api_key="sk-zgqsblbwondpoxlactagcfrqidwmlbcikrpldomyqceqpsss", base_url="https://api.siliconflow.cn/v1")
client = OpenAI(api_key="sk-1ba964a199d747c89e02cabebf9c7e37", base_url="https://dashscope.aliyuncs.com/compatible-mode/v1")

prop_fns = get_prop_function()

def str_2_emb(smiles_list, tokenizer, model):

    inputs = tokenizer(smiles_list, return_tensors="pt")

    outputs = model(**inputs)

    embeddings = outputs.last_hidden_state
    smiles_embedding = embeddings[:, 0, :]
    return smiles_embedding

# def hidden_init(layer):
#     fan_in = layer.weight.data.size()[0]
#     lim = 1. / np.sqrt(fan_in)
#     return (-lim, lim)

# class Actor(nn.Module):
#     def __init__(self, state_dim, hidden_dim, action_dim):
#         super(Actor, self).__init__()
#         self.fc1 = nn.Linear(state_dim, hidden_dim)
#         self.fc2 = nn.Linear(hidden_dim, hidden_dim)
#         self.fc3 = nn.Linear(hidden_dim, hidden_dim)
#         self.fc4 = nn.Linear(hidden_dim, action_dim)
#         self.softmax = nn.Softmax(dim=-1)
    
#     def forward(self, x):
#         x = F.relu(self.fc1(x))
#         x = F.relu(self.fc2(x))
#         x = F.relu(self.fc3(x))
#         x = F.relu(self.fc4(x))
#         action_probs = self.softmax(x)
#         return action_probs
    
#     def evaluate(self, x, epsilon=1e-6):
#         action_probs = self.forward(x)
#         dist = Categorical(action_probs)
#         action = dist.sample().to(x.device)
#         z = action_probs == 0.0
#         z = z.float() * 1e-8
#         log_action_probabilities = torch.log(action_probs + z)
#         return action.detach().cpu(), action_probs, log_action_probabilities
    
#     def get_action(self, x):
#         action_probs = self.forward(x)
#         dist = Categorical(action_probs)
#         action = dist.sample().to(x.device)
#         z = action_probs == 0.0
#         z = z.float() * 1e-8
#         log_action_probabilities = torch.log(action_probs + z)
#         return action.detach().cpu(), action_probs, log_action_probabilities
    
#     # def get_det_action(self, x):
#     #     action_probs = self.forward(x)
#     #     dist = Categorical(action_probs)
#     #     action = dist.sample().to(x.device)
#     #     return action.detach().cpu()
    
#     def get_det_action(self, x, mask_id):
#         action_probs = self.forward(x)
#         mask = torch.zeros_like(action_probs)
#         mask[mask_id] = 1
#         action_probs = action_probs * mask
#         action_probs /= (action_probs.sum()+1e-7)
#         dist = Categorical(action_probs)
#         action = dist.sample().to(x.device)
#         return action.detach().cpu()
    
# class Critic(nn.Module):
#     def __init__(self, state_dim, hidden_dim, action_dim, seed=1):
#         super(Critic, self).__init__()
#         self.fc1 = nn.Linear(state_dim, hidden_dim)
#         self.fc2 = nn.Linear(hidden_dim, hidden_dim)
#         self.fc3 = nn.Linear(hidden_dim, hidden_dim)
#         self.fc4 = nn.Linear(hidden_dim, action_dim)
#         self.reset_parameters()

#     def reset_parameters(self):
#         self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
#         self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
#         self.fc3.weight.data.uniform_(*hidden_init(self.fc3))
#         self.fc4.weight.data.uniform_(-3e-3, 3e-3)

#     def forward(self, x):
#         x = F.relu(self.fc1(x))
#         x = F.relu(self.fc2(x))
#         x = F.relu(self.fc3(x))
#         x = self.fc4(x)
#         return x

# class RL_Planner(nn.Module):
#     def __init__(self, state_dim, hidden_dim, action_dim, actor_lr, critic_lr, alpha_lr, target_entropy, tau, gamma, device):
#         super(RL_Planner, self).__init__()
#         self.state_dim = state_dim
#         self.hidden_sim = hidden_dim
#         self.action_dim = action_dim
#         self.device = device
#         self.gamma = gamma
#         self.tau = tau
#         self.actor_lr = actor_lr
#         self.critic_lr = critic_lr
#         self.target_entropy = target_entropy
#         self.log_alpha = torch.tensor([0.0], requires_grad=True)
#         self.alpha = self.log_alpha.exp().detach()
#         self.alpha_optimizer = optim.Adam(params=[self.log_alpha], lr=alpha_lr) 

#         # CQL params
#         self.with_lagrange = False
#         self.temp = 1.0
#         self.cql_weight = 1.0
#         self.clip_grad_param = 1
#         self.target_action_gap = 0.0
#         self.cql_log_alpha = torch.zeros(1, requires_grad=True)
#         self.cql_alpha_optimizer = optim.Adam(params=[self.cql_log_alpha], lr=alpha_lr) 

#         # Actor Network
#         self.actor = Actor(state_dim, hidden_dim, action_dim).to(device)
#         self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)

#         # Critic Network (w/ Target Network)
#         self.critic_1 = Critic(state_dim, hidden_dim, action_dim, seed=1).to(device)
#         self.critic_2 = Critic(state_dim, hidden_dim, action_dim, seed=2).to(device)

#         assert self.critic_1.parameters() != self.critic_2.parameters()
        
#         self.critic_1_target = Critic(state_dim, hidden_dim, action_dim, seed=3).to(device)
#         self.critic_1_target.load_state_dict(self.critic_1.state_dict())

#         self.critic_2_target = Critic(state_dim, hidden_dim, action_dim).to(device)
#         self.critic_2_target.load_state_dict(self.critic_2.state_dict())

#         self.critic_1_optimizer = optim.Adam(self.critic_1.parameters(), lr=critic_lr)
#         self.critic_2_optimizer = optim.Adam(self.critic_2.parameters(), lr=critic_lr)
#         self.softmax = nn.Softmax(dim=-1)

#     # def get_action(self, state):
#     #     state = torch.from_numpy(state).float().to(self.device)
        
#     #     with torch.no_grad():
#     #         action = self.actor.get_det_action(state)
#     #     return action.numpy()
    
#     def get_action(self, state, mask_id):
#         state = torch.from_numpy(state).float().to(self.device)
        
#         with torch.no_grad():
#             action = self.actor.get_det_action(state, mask_id)
#         return action.numpy()
    
#     def calc_policy_loss(self, states, alpha):
#         _, action_probs, log_pis = self.actor.evaluate(states)

#         q1 = self.critic_1(states)   
#         q2 = self.critic_2(states)
#         min_Q = torch.min(q1,q2)
#         actor_loss = (action_probs * (alpha.to(self.device) * log_pis - min_Q )).sum(1).mean()
#         log_action_pi = torch.sum(log_pis * action_probs, dim=1)
#         return actor_loss, log_action_pi
        
#     def soft_update(self, net, target_net):
#         for param_target, param in zip(target_net.parameters(), net.parameters()):
#             param_target.data.copy_(param_target.data * (1.0 - self.tau) + param.data * self.tau)

#     def learn(self, step, experiences, gamma, d=1):
#         states, actions, rewards, next_states, dones = experiences
    
#         current_alpha = copy.deepcopy(self.alpha)
#         actor_loss, log_pis = self.calc_policy_loss(states, current_alpha)
#         self.actor_optimizer.zero_grad()
#         actor_loss.backward()
#         self.actor_optimizer.step()
        
#         # Compute alpha loss
#         alpha_loss = - (self.log_alpha.exp() * (log_pis.cpu() + self.target_entropy).detach().cpu()).mean()
#         self.alpha_optimizer.zero_grad()
#         alpha_loss.backward()
#         self.alpha_optimizer.step()
#         self.alpha = self.log_alpha.exp().detach()

#         # ---------------------------- update critic ---------------------------- #
#         # Get predicted next-state actions and Q values from target models
#         with torch.no_grad():
#             _, action_probs, log_pis = self.actor.evaluate(next_states)
#             Q_target1_next = self.critic_1_target(next_states)
#             Q_target2_next = self.critic_2_target(next_states)
#             Q_target_next = action_probs * (torch.min(Q_target1_next, Q_target2_next) - self.alpha.to(self.device) * log_pis)

#             # Compute Q targets for current states (y_i)
#             Q_targets = rewards + (gamma * (1 - dones) * Q_target_next.sum(dim=1).unsqueeze(-1)) 


#         # Compute critic loss
#         q1 = self.critic_1(states)
#         q2 = self.critic_2(states)
        
#         q1_ = q1.gather(1, actions.long())
#         q2_ = q2.gather(1, actions.long())
        
#         critic1_loss = 0.5 * F.mse_loss(q1_, Q_targets)
#         critic2_loss = 0.5 * F.mse_loss(q2_, Q_targets)
        
#         cql1_scaled_loss = torch.logsumexp(q1, dim=1).mean() - q1.mean()
#         cql2_scaled_loss = torch.logsumexp(q2, dim=1).mean() - q2.mean()
        
#         cql_alpha_loss = torch.FloatTensor([0.0])
#         cql_alpha = torch.FloatTensor([0.0])
#         if self.with_lagrange:
#             cql_alpha = torch.clamp(self.cql_log_alpha.exp(), min=0.0, max=1000000.0).to(self.device)
#             cql1_scaled_loss = cql_alpha * (cql1_scaled_loss - self.target_action_gap)
#             cql2_scaled_loss = cql_alpha * (cql2_scaled_loss - self.target_action_gap)

#             self.cql_alpha_optimizer.zero_grad()
#             cql_alpha_loss = (- cql1_scaled_loss - cql2_scaled_loss) * 0.5 
#             cql_alpha_loss.backward(retain_graph=True)
#             self.cql_alpha_optimizer.step()
        
#         total_c1_loss = critic1_loss + cql1_scaled_loss
#         total_c2_loss = critic2_loss + cql2_scaled_loss
        
        
#         # Update critics
#         # critic 1
#         self.critic_1_optimizer.zero_grad()
#         total_c1_loss.backward(retain_graph=True)
#         clip_grad_norm_(self.critic_1.parameters(), self.clip_grad_param)
#         self.critic_1_optimizer.step()
#         # critic 2
#         self.critic_2_optimizer.zero_grad()
#         total_c2_loss.backward()
#         clip_grad_norm_(self.critic_2.parameters(), self.clip_grad_param)
#         self.critic_2_optimizer.step()

#         # ----------------------- update target networks ----------------------- #
#         self.soft_update(self.critic_1, self.critic_1_target)
#         self.soft_update(self.critic_2, self.critic_2_target)
        
#         return actor_loss.item(), alpha_loss.item(), critic1_loss.item(), critic2_loss.item(), cql1_scaled_loss.item(), cql2_scaled_loss.item(), current_alpha, cql_alpha_loss.item(), cql_alpha.item()

# prompt_template = (
#     "Can you make molecule {root_mol} less soluble in water? "
# 	"The output molecule should be similar to the input molecule. "
#     "{suggestion} "
# 	"Give me five molecules in SMILES only and list them using bullet points. "
#     # "Do not show any hydrogen in SMILES strings! "
# 	"No explanation is needed. "
# )

# smiles_list, _ = zinc_process("250k_rndm_zinc_drugs_clean_3.csv", 30)
# print(smiles_list)

model_name = "seyonec/ChemBERTa-zinc-base-v1"
model = AutoModel.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

def get_random_action(action_list):
    action = random.choice(action_list)
    return action

def concat_state(b_rs, b_cs):
    batch_size = len(b_rs)
    b_s = [None] * batch_size
    for i in range(batch_size):
        b_s[i] = np.concatenate((np.array(b_rs[i]), np.array(b_cs[i])))
    return tuple(b_s)
    
def reward_function_concat(prop_name, b_r, opt_direction, threshold):
    batch_size = len(b_r)
    batch_reward = [None] * batch_size

    for i, b_r_list in enumerate(b_r):
        valid_val = b_r_list[0]
        root_prop = b_r_list[1]
        curr_prop = b_r_list[2]
        new_prop = b_r_list[3]
        curr_sim = b_r_list[4]
        root_sim = b_r_list[5]
        reward = 0
        # print(new_prop)
        reward = get_reward(prop_name=prop_name, root_prop=curr_prop, new_prop=new_prop, root_sim=curr_sim, valid_val=valid_val, opt_direction=opt_direction, threshold=threshold)
       
        batch_reward[i] = reward
    return tuple(batch_reward)

def get_hq_buffer(total_buffer, batch_size, hq_buffer_size):
    hq_buffer = ReplayBuffer(hq_buffer_size)
    # gather good samples:
    progress_bar = tqdm(total=hq_buffer_size, desc="Collecting HQ Buffer")
    while hq_buffer.size() < hq_buffer_size:
        b_rs, b_cs, b_a, b_r, b_ns, b_d = total_buffer.sample(batch_size)
        
        b_r_list = reward_function_concat(b_r)   
        
        for i in range(batch_size):
            r = b_r_list[i]
            if r >= 1:
                # if b_a[i] == 0:
                hq_buffer.add(b_rs[i], b_cs[i], b_a[i], b_r[i], b_ns[i], b_d[i])
                # else:
                #     hq_buffer.add(b_rs[i], b_cs[i], b_a[i], b_r[i], b_ns[i], b_d[i])
                progress_bar.update(1)
    progress_bar.close()
    return hq_buffer

class mol_env:
    def __init__(self, reasoning_instruction, prompt_template, max_depth, val_smiles_list, opt_direction, prop_name, threshold, task_objective):
        self.reasoning_instruction = reasoning_instruction
        self.prompt_template = prompt_template
        self.reward_fn = get_reward
        self.opt_direction = opt_direction
        self.prop_name = prop_name
        self.task_objective = task_objective
        self.threshold = threshold
        self.val_smiles_list = val_smiles_list
        self.max_depth = max_depth
        self.depth = 0

    def reset(self):
        self.depth = 0
        self.root_mol = random.choice(self.val_smiles_list)
        self.root_prop = {prop_nm: prop_fns[prop_nm](self.root_mol) for prop_nm in self.prop_name}
        self.curr_mol = self.root_mol
        self.curr_prop = {prop_nm: prop_fns[prop_nm](self.curr_mol) for prop_nm in self.prop_name}
        state_emb = str_2_emb([self.curr_mol], tokenizer, model).squeeze().detach()
        state_emb = np.array(state_emb)
        # state_emb = np.concatenate((state_emb, state_emb))
        return state_emb

    def step(self, action):
        self.depth += 1
        threshold_specific_prompt = ""
        for prop_nm in self.prop_name:
            threshold_specific_prompt += f"{self.opt_direction[prop_nm]} {prop_nm} by at least {self.threshold[prop_nm]}. "
        vals = {
            'root_mol': self.root_mol,
            'task_objective': self.task_objective,
            'threshold_specific_prompt': threshold_specific_prompt,
            'current_best_mol': self.curr_mol,
            'suggestion': action,
            "reasoning_instruction": self.reasoning_instruction,
        }
        prompt = fstr(self.prompt_template, vals)
        messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt},
            ]
        received = False
        while not received:
            try:
                response = client.chat.completions.create(
                    # model="deepseek-chat",
                    # model="deepseek-ai/DeepSeek-V3",
                    model='qwen-plus',
                    messages=messages,
                    max_tokens=1024,
                    temperature=0,
                    stream=False
                )
                received = True
            except:
                error = sys.exc_info()[0]
                print(f"Error: {error}")
                time.sleep(1)
        answer_txt = response.choices[0].message.content
        # answer = run_deepseek_prompts([prompt], [system_prompt])[0]
        # answer_txt = answer['answer']
        answer_list = parse(answer_txt)
        done = False
        reward = 0
        print(f"current mol: {self.curr_mol}; answer_list: {answer_list}")
        for answer_id, answer_mol in enumerate(answer_list):
            if (answer_mol == ""):
                continue

            if is_valid_smiles(answer_mol):
                valid_val = 1
                new_prop = {prop_nm: prop_fns[prop_nm](answer_mol) for prop_nm in self.prop_name}
                sim = calculate_tanimoto_similarity(self.curr_mol, answer_mol)
                # curr_sim = calculate_tanimoto_similarity(self.curr_mol, answer_mol)
                if sim == 1.0:
                    continue
                # print("Outside: ")
                # print("prop name: ", self.prop_name)
                # print("opt direction: ", self.opt_direction)
                # print("root prop: ", self.curr_prop)
                # print("root sim: ", sim)
                # print("new prop: ", new_prop)
                # print("threshold: ", self.threshold)
                reward = self.reward_fn(prop_name=self.prop_name, root_prop=self.curr_prop, new_prop=new_prop, root_sim=sim, valid_val=valid_val, opt_direction=self.opt_direction, threshold=self.threshold)
                ### not true
                if reward > 0:
                    done = True
                self.curr_mol = answer_mol
                self.curr_prop = new_prop
                break
        # root_emb = str_2_emb([self.curr_mol], tokenizer, model).squeeze().detach()
        # root_emb = np.array(root_emb)
        state_emb = str_2_emb([self.curr_mol], tokenizer, model).squeeze().detach()
        state_emb = np.array(state_emb)
        return state_emb, reward, done
    

def main(args):
    with open(args.validation_data_path, 'r') as file:
        lines = file.readlines()

    val_smiles_list = [line.strip() for line in lines]

    print("Task ID: ", args.task_id)
    print("Constraint: ", args.constraint)
    print("Replay Buffer Name: ", args.replay_buffer_name)

    prompt_template = get_generation_prompt_template(args.task_id, 'single')

    prop_name, opt_direction, task_objective, threshold = get_task_info(constraint=args.constraint, task_id=args.task_id)
    
    replay_buffer_name = args.replay_buffer_name
    constraint = args.constraint
    max_depth = args.val_max_depth

    env = mol_env(reasoning_instruction="No explanation is needed.", prompt_template=prompt_template, max_depth=max_depth, val_smiles_list=val_smiles_list, task_objective=task_objective,
                  prop_name=prop_name, opt_direction=opt_direction, threshold=threshold)

    with open('results/replay_buffer/' + replay_buffer_name+'.pkl', 'rb') as file:
        replay_buffer = pickle.load(file)

    # action_list = get_all_general_action_list()
    action_list = []
    with open('action_list.txt', 'r', encoding='utf-8') as file:
        lines = file.readlines()
        for line in lines:
            action_list.append(line.strip())
    
    state_dim = 768
    action_dim = len(action_list)
    print("action dim: ", action_dim)
    hidden_dim = 512

    actor_lr = args.learning_rate
    critic_lr = args.learning_rate
    alpha_lr = args.learning_rate

    gamma = 0.99
    tau = args.tau
    batch_size = 512

    num_epochs = 100
    num_val = 10
    num_trains_per_epoch = 10

    target_entropy = -1
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    agent = RL_Planner(state_dim=state_dim, 
                    hidden_dim=hidden_dim,
                    action_dim=action_dim,
                    actor_lr=actor_lr,
                    critic_lr=critic_lr,
                    alpha_lr=alpha_lr,
                    target_entropy=target_entropy,
                    tau=tau,
                    gamma=gamma,
                    device=device
                    )
    
    alpha_scheduler = lr_scheduler.CosineAnnealingLR(agent.alpha_optimizer, T_max=num_epochs, eta_min=1e-6)
    actor_scheduler = lr_scheduler.CosineAnnealingLR(agent.actor_optimizer, T_max=num_epochs, eta_min=1e-6)
    cql_alpha_scheduler = lr_scheduler.CosineAnnealingLR(agent.cql_alpha_optimizer, T_max=num_epochs, eta_min=1e-6)
    critic_1_scheduler = lr_scheduler.CosineAnnealingLR(agent.critic_1_optimizer, T_max=num_epochs, eta_min=1e-6)
    critic_2_scheduler = lr_scheduler.CosineAnnealingLR(agent.critic_2_optimizer, T_max=num_epochs, eta_min=1e-6)

    return_list = []
    best_reward = 0
    steps = 0
    for i in range(10):
        with tqdm(total=int(num_epochs / 10), desc='Iteration %d' % i) as pbar:
            for i_epoch in range(int(num_epochs / 10)):
                if i >= 0:
                    epoch_return = 0
                    for val_id in range(num_val):
                        print(f"start id validation of epoch {i_epoch}...")
                        max_reward = 0
                        id_state = env.reset()
                        mask_id = [i for i in range(len(action_list))]
                        # mask_id = [action_list.index(item) for item in get_general_action_list(env.curr_mol)]
                        done = False
                        steps = 0
                        while not done:
                            steps += 1
                            print(f"action space size for {env.curr_mol}: {len(mask_id)}")
                            action_idx = agent.get_action(id_state, mask_id)
                            action = action_list[action_idx]
                            next_state, reward, done = env.step(action)
                            mask_id.remove(action_idx)
                            print(f"val_id: {val_id}: step: {steps} with action: {action}; resulting reward: {reward}")
                            id_state = next_state
                            if reward > max_reward:
                                max_reward = reward
                            if env.depth >= env.max_depth:
                                break
                        epoch_return += max_reward
                    return_list.append(epoch_return / steps)

                    if (epoch_return / num_val) > best_reward:
                        best_reward = (epoch_return / num_val)

                        model_dict = {
                            'state_dict': agent.state_dict(),
                            'state_dim': state_dim,
                            'hidden_dim': hidden_dim,
                            'action_dim': action_dim,
                            'actor_lr': actor_lr,
                            'critic_lr': critic_lr,
                            'alpha_lr': alpha_lr,
                            'target_entropy': target_entropy,
                            'tau': tau,
                            'gamma': gamma,
                            'device': device
                        }
                        torch.save(model_dict, "results/rl_model_checkpoint/" + replay_buffer_name + "_" + constraint + "_" + str(args.task_id) +'_best_reward_action_2' + '.pth')

                for _ in range(num_trains_per_epoch):
                    b_rs, b_cs, b_a, b_r, b_ns, b_d = replay_buffer.sample(batch_size)
                    b_r = reward_function_concat(prop_name, b_r, opt_direction, threshold)
                    # b_s = concat_state(b_rs, b_cs)
                    b_s = b_cs
                    b_s = torch.from_numpy(np.stack(b_s, axis=0)).float().to(device)
                    
                    b_a = torch.tensor(list(b_a)).int().to(device).unsqueeze(1)
                    b_r = torch.tensor(list(b_r)).float().to(device).unsqueeze(1)
                    # b_ns = concat_state(b_rs, b_ns)
                    b_ns = b_ns
                    b_ns = torch.from_numpy(np.stack(b_ns, axis=0)).float().to(device)
                    b_d = torch.tensor(list(b_d), dtype=torch.bool).float().to(device).unsqueeze(1)
                    zero_indices = (b_r == 0).nonzero()
                    b_d.fill_(0)
                    for idx in zero_indices:
                        if idx[0] < b_d.size(0) - 1:
                            b_d[idx[0] + 1, idx[1]] = 1
                    experiences = (b_s, b_a, b_r, b_ns, b_d)
                    policy_loss, alpha_loss, bellmann_error1, bellmann_error2, cql1_loss, cql2_loss, current_alpha, lagrange_alpha_loss, lagrange_alpha = agent.learn(steps, experiences, gamma=0.99)
                    
                actor_scheduler.step()
                alpha_scheduler.step()
                cql_alpha_scheduler.step()
                critic_1_scheduler.step()
                critic_2_scheduler.step()
                    
                if (i_epoch + 1) % 10 == 0:
                    pbar.set_postfix({
                        'epoch':
                        '%d' % (num_epochs / 10 * i + i_epoch + 1),
                        'id_return':
                        '%.3f' % np.mean(return_list[-10:]),
                        # 'od_return':
                        # '%.3f' % np.mean(od_return_list[-10:]),
                    })
                pbar.update(1)

        model_dict = {
            'state_dict': agent.state_dict(),
            'state_dim': state_dim,
            'hidden_dim': hidden_dim,
            'action_dim': action_dim,
            'actor_lr': actor_lr,
            'critic_lr': critic_lr,
            'alpha_lr': alpha_lr,
            'target_entropy': target_entropy,
            'tau': tau,
            'gamma': gamma,
            'device': device
        }
        # torch.save(model_dict, "results/rl_model_checkpoint/" + replay_buffer_name+ "_" + constraint +  '_' + str(args.task_id) + '_epoch_' + str(i) + '.pth')

        episodes_list = list(range(len(return_list)))

        if i == 0:
            plt.plot(episodes_list, return_list, color='blue', label='Return')
            # plt.plot(episodes_list, od_return_list, color='orange', label='Od return')
        else:
            plt.plot(episodes_list, return_list, color='blue')
            # plt.plot(episodes_list, od_return_list, color='orange')

        plt.legend()

        plt.xlabel('Episodes')
        plt.ylabel('Returns')

        plt.title("CQL")
        plt.savefig("results/figure/" + replay_buffer_name+ "_" + constraint + "_" + str(args.task_id) + '_action_2.pdf')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Awesome Drug Edit")
    
    # basic param
    parser.add_argument("--task_id", type=int, default=101, help="optimization task")
    parser.add_argument("--learning_rate", type=float, default=3e-5, help="learning rate")
    parser.add_argument("--tau", type=float, default=0.02, help="tau")
    parser.add_argument("--validation_data_path", type=str, default="Data/validation_molecule.txt", help="optimization task")
    parser.add_argument("--replay_buffer_name", type=str, default='general_replay_buffer_mol_logP', help='Name of Replay Buffer')
    parser.add_argument("--constraint", type=str, default='strict', help="loose or strict")
    parser.add_argument("--val_max_depth", type=int, default=3, help="max depth of validation tree")
    args = parser.parse_args()
    
    main(args)