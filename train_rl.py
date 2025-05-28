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
from utils.rl_utils import get_smiles_all_general_action_list, get_smiles_general_action_list
from utils.tool import fstr, parse_molecule, is_valid_smiles, calculate_tanimoto_similarity, get_task_info, get_prop_function
from search.reward.reward_function import get_reward_mol_train
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

# model_name = "seyonec/ChemBERTa-zinc-base-v1"
model_name = "/root/smiles_embedding_model/"
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
    
def reward_function_concat(prop_name, b_r, opt_direction, threshold, reward_type, a, b, c):
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
        reward = get_reward_mol_train(prop_name=prop_name, root_prop=curr_prop, new_prop=new_prop, valid_val=valid_val, opt_direction=opt_direction, threshold=threshold, reward_type=reward_type, a=a, b=b, c=c)
       
        batch_reward[i] = reward
    return tuple(batch_reward)

class mol_env:
    def __init__(self, reasoning_instruction, prompt_template, max_depth, val_smiles_list, opt_direction, prop_name, threshold, task_objective, reward_type, a, b, c):
        self.reasoning_instruction = reasoning_instruction
        self.prompt_template = prompt_template
        self.reward_fn = get_reward_mol_train
        self.opt_direction = opt_direction
        self.prop_name = prop_name
        self.task_objective = task_objective
        self.threshold = threshold
        self.val_smiles_list = val_smiles_list
        self.max_depth = max_depth
        self.reward_type = reward_type
        self.a = a
        self.b = b
        self.c = c
        self.depth = 0

    def reset(self):
        self.depth = 0
        self.root_mol = random.choice(self.val_smiles_list)
        self.root_prop = {}
        for prop_nm in self.prop_name:
            if (prop_nm == "tanimoto_similarity" or prop_nm == "levenshtein_similarity"):
                self.root_prop[prop_nm] = prop_fns[prop_nm](self.root_mol, self.root_mol)
            else:
                self.root_prop[prop_nm] = prop_fns[prop_nm](self.root_mol)
        self.curr_mol = self.root_mol
        self.curr_prop = self.root_prop
        state_emb = str_2_emb([self.curr_mol], tokenizer, model).squeeze().detach()
        state_emb = np.array(state_emb)
        return state_emb

    def step(self, action):
        self.depth += 1
        vals = {
            'root_mol': self.root_mol,
            'task_objective': self.task_objective,
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
        answer_list = parse_molecule(answer_txt)
        done = False
        reward = 0
        print(f"current mol: {self.curr_mol}; answer_list: {answer_list}")
        for answer_id, answer_mol in enumerate(answer_list):
            if (answer_mol == ""):
                continue

            if is_valid_smiles(answer_mol):
                valid_val = 1
                new_prop = {}
                for prop_nm in self.prop_name:
                    if (prop_nm == "tanimoto_similarity" or prop_nm == "levenshtein_similarity"):
                        new_prop[prop_nm] = prop_fns[prop_nm](self.root_mol, answer_mol)
                    else:
                        new_prop[prop_nm] = prop_fns[prop_nm](answer_mol)
                
                reward = self.reward_fn(prop_name=self.prop_name, root_prop=self.curr_prop, new_prop=new_prop, valid_val=valid_val, opt_direction=self.opt_direction, threshold=self.threshold, reward_type=self.reward_type, a=self.a, b=self.b, c=self.c)
                ### not true
                if reward > 0:
                    done = True
                self.curr_mol = answer_mol
                self.curr_prop = new_prop
                break
        state_emb = str_2_emb([self.curr_mol], tokenizer, model).squeeze().detach()
        state_emb = np.array(state_emb)
        return state_emb, reward, done
    

def main(args):
    replay_buffer_name = args.replay_buffer_name
    constraint = args.constraint
    reward_type = args.reward_type
    a = args.a
    b = args.b
    c = args.c
    tau = args.tau
    max_depth = args.val_max_depth
    with open(args.validation_data_path, 'r') as file:
        lines = file.readlines()

    val_smiles_list = [line.strip() for line in lines]

    print("Task ID: ", args.task_id)
    print("Constraint: ", args.constraint)
    print("Replay Buffer Name: ", args.replay_buffer_name)
    
    checkpoint_dir = Path("results/rl_model_checkpoint")
    figure_dir = Path("results/figure")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)
    version = 1
    pth_save_dir = "results/rl_model_checkpoint/" + f"reward_type_{reward_type}_a{str(a)}_b{str(b)}_c{str(c)}_tau{str(tau)}" + replay_buffer_name + "_" + constraint + "_" + str(args.task_id) + "_best_reward" + f"_v{version}.pth"
    fig_save_dir = "results/figure/" + f"reward_type_{reward_type}_a{str(a)}_b{str(b)}_c{str(c)}_tau{str(tau)}" + replay_buffer_name+ "_" + constraint + "_" + str(args.task_id) + f"_v{version}.pdf"
    while True:
        if os.path.exists(pth_save_dir) and os.path.exists(fig_save_dir):
            version+=1
            pth_save_dir = "results/rl_model_checkpoint/" + f"reward_type_{reward_type}_a{str(a)}_b{str(b)}_c{str(c)}_tau{str(tau)}" + replay_buffer_name + "_" + constraint + "_" + str(args.task_id) + "_best_reward" + f"_v{version}.pth"
            fig_save_dir = "results/figure/" + f"reward_type_{reward_type}_a{str(a)}_b{str(b)}_c{str(c)}_tau{str(tau)}" + replay_buffer_name+ "_" + constraint + "_" + str(args.task_id) + f"_v{version}.pdf"
        else:
            break

    prompt_template = get_generation_prompt_template(args.task_id, "single", "deepseek")

    drug_type, prop_name, opt_direction, task_objective, threshold = get_task_info(constraint=args.constraint, task_id=args.task_id)
    
    print("Prop name: ", prop_name)
    
    env = mol_env(reasoning_instruction="No explanation is needed.", prompt_template=prompt_template, max_depth=max_depth, val_smiles_list=val_smiles_list, task_objective=task_objective,
                  prop_name=prop_name, opt_direction=opt_direction, threshold=threshold, reward_type=reward_type, a=a, b=b, c=c)

    with open('results/replay_buffer/' + replay_buffer_name+'.pkl', 'rb') as file:
        replay_buffer = pickle.load(file)

    action_list = get_smiles_all_general_action_list()
    
    state_dim = 768
    action_dim = len(action_list)
    print("action dim: ", action_dim)
    hidden_dim = 2048

    actor_lr = 3e-4
    critic_lr = 3e-4
    alpha_lr = 3e-4

    gamma = 0.99
    tau = args.tau
    batch_size = 128

    num_epochs = 10
    num_val = 1
    num_trains_per_epoch = 100

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
                        mask_id = [action_list.index(item) for item in get_smiles_general_action_list(env.curr_mol)]
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
                        
                        torch.save(model_dict, pth_save_dir)

                for _ in range(num_trains_per_epoch):
                    b_rs, b_cs, b_a, b_r, b_ns, b_d = replay_buffer.sample(batch_size)
                    b_r = reward_function_concat(prop_name, b_r, opt_direction, threshold, reward_type=args.reward_type, a=args.a, b=args.b, c=args.c)
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
        # if i == 0:
        #     torch.save(model_dict, "results/rl_model_checkpoint/" + replay_buffer_name+ "_" + constraint +  '_' + str(args.task_id) + '_epoch_' + str(i) + '.pth')

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
        plt.savefig(fig_save_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Awesome Drug Edit")
    
    # basic param
    parser.add_argument("--task_id", type=int, default=101, help="optimization task")
    parser.add_argument("--validation_data_path", type=str, default="Data/validation_molecule.txt", help="optimization task")
    parser.add_argument("--replay_buffer_name", type=str, default='general_replay_buffer_mol_logP', help='Name of Replay Buffer')
    parser.add_argument("--reward_type", type=str, help='type of reward (add or mul)')
    parser.add_argument("--a", type=float)
    parser.add_argument("--b", type=float)
    parser.add_argument("--c", type=float)
    parser.add_argument("--tau", type=float)
    
    parser.add_argument("--constraint", type=str, default='strict', help="loose or strict")
    parser.add_argument("--val_max_depth", type=int, default=3, help="max depth of validation tree")
    args = parser.parse_args()
    
    main(args)