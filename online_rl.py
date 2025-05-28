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
from matplotlib.backends.backend_pdf import PdfPages
from torch.nn.utils import clip_grad_norm_
import torch.optim.lr_scheduler as lr_scheduler

sys.path.append("src")
from model.rl_planner import PPO
from utils.rl_utils import get_smiles_all_general_action_list, get_smiles_general_action_list
from utils.tool import fstr, parse_molecule, get_task_info, get_prop_function, load_dataset, is_valid_smiles
from search.reward.reward_function import get_reward_mol
from llm.prompt_template import system_prompt, get_generation_prompt_template

client = OpenAI(api_key="sk-1ba964a199d747c89e02cabebf9c7e37", base_url="https://dashscope.aliyuncs.com/compatible-mode/v1")

prop_fns = get_prop_function()

def str_2_emb(smiles_list, tokenizer, model):

    inputs = tokenizer(smiles_list, return_tensors="pt")

    outputs = model(**inputs)

    embeddings = outputs.last_hidden_state
    smiles_embedding = embeddings[:, 0, :]
    return smiles_embedding
    
model_name = "/root/smiles_embedding_model/"
model = AutoModel.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

def if_done(prop_name, root_prop, curr_prop, threshold, opt_direction):
    done = True
    for prop_nm in prop_name:
        if "similarity" in prop_nm:
            continue
        direction = opt_direction[prop_nm]
        if direction == "increase":
            if (curr_prop[prop_nm] - root_prop[prop_nm]) < threshold[prop_nm]:
                done = False
        elif direction == "decrease":
            if (root_prop[prop_nm] - curr_prop[prop_nm]) < threshold[prop_nm]:
                done = False
    return done
            

class mol_env:
    def __init__(self, reasoning_instruction, prompt_template, val_drug_list, opt_direction, prop_name, threshold, task_objective):
        self.reasoning_instruction = reasoning_instruction
        self.prompt_template = prompt_template
        self.reward_fn = get_reward_mol
        self.opt_direction = opt_direction
        self.prop_name = prop_name
        self.task_objective = task_objective
        self.threshold = threshold
        self.val_drug_list = val_drug_list
        self.depth = 0

    def reset(self):
        self.depth = 0
        self.root_mol = random.choice(self.val_drug_list)
        self.curr_mol = self.root_mol
        self.root_prop = {}
        for prop_nm in self.prop_name:
            if ("similarity" in prop_nm):
                self.root_prop[prop_nm] = prop_fns[prop_nm](self.root_mol, self.root_mol)
            else:
                self.root_prop[prop_nm] = prop_fns[prop_nm](self.root_mol)
        self.curr_prop = self.root_prop
        state_emb = str_2_emb([self.curr_mol], tokenizer, model).squeeze().detach()
        state_emb = np.array(state_emb)
        # state_emb = np.concatenate((state_emb, state_emb))
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
        answer_list = parse_molecule(answer_txt)
        done = False
        reward = 0
        print(f"current mol: {self.curr_mol}; answer_list: {answer_list}")
        find_mol = False
        for answer_id, answer_mol in enumerate(answer_list):
            if (answer_mol == ""):
                continue

            if is_valid_smiles(answer_mol):
                find_mol = True
                valid_val = 1
                new_prop = {}
                for prop_nm in self.prop_name:
                    if ("similarity" in prop_nm):
                        new_prop[prop_nm] = prop_fns[prop_nm](self.root_mol, answer_mol)
                        if new_prop[prop_nm] == 1.0:
                            continue
                    else:
                        new_prop[prop_nm] = prop_fns[prop_nm](answer_mol)
                reward = self.reward_fn(prop_name=self.prop_name, root_prop=self.curr_prop, new_prop=new_prop, valid_val=valid_val, opt_direction=self.opt_direction, threshold=self.threshold)

                done = if_done(prop_name=self.prop_name, root_prop=self.curr_prop, curr_prop=new_prop, threshold=self.threshold, opt_direction=self.opt_direction)
                
                self.curr_mol = answer_mol
                self.curr_prop = new_prop
                break
        # root_emb = str_2_emb([self.curr_mol], tokenizer, model).squeeze().detach()
        # root_emb = np.array(root_emb)
        state_emb = str_2_emb([self.curr_mol], tokenizer, model).squeeze().detach()
        state_emb = np.array(state_emb)
        if (not find_mol) or (self.depth >= 5):
            reward = -1
            done = True
        return state_emb, reward, done

def main(args):
    constraint = args.constraint
    val_drug_list = load_dataset(args.task_id)

    log_dir = Path("results/rl_training_log/online")
    checkpoint_dir = Path("results/rl_model_checkpoint/online")
    figure_dir = Path("results/figure/online")
    log_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)
    
    version = 1
    log_save_dir = "results/rl_training_log/online/" + "Task" + str(args.task_id) + "_" + constraint + f"_v{version}.log"
    pth_save_dir = "results/rl_model_checkpoint/online/" + "Task" + str(args.task_id) + "_" + constraint + "_best_reward" + f"_v{version}.pth"
    fig_save_dir = "results/figure/online/" + "Task" + str(args.task_id) + "_" + constraint + f"_v{version}.pdf"
    while True:
        if os.path.exists(pth_save_dir) or os.path.exists(fig_save_dir) or os.path.exists(log_save_dir):
            version+=1
            log_save_dir = "results/rl_training_log/online/" + "Task" + str(args.task_id) + "_" + constraint + f"_v{version}.log"
            pth_save_dir = "results/rl_model_checkpoint/online/" + "Task" + str(args.task_id) + "_" + constraint + "_best_reward" + f"_v{version}.pth"
            fig_save_dir = "results/figure/online/" + "Task" + str(args.task_id) + "_" + constraint + f"_v{version}.pdf"
        else:
            break
    f = open(log_save_dir, 'w', encoding="utf-8")
    prompt_template = get_generation_prompt_template(args.task_id, "single", "deepseek")

    drug_type, prop_name, opt_direction, task_objective, threshold = get_task_info(constraint=args.constraint, task_id=args.task_id)

    env = mol_env(reasoning_instruction="No explanation is needed.", prompt_template=prompt_template, val_drug_list=val_drug_list, task_objective=task_objective, prop_name=prop_name, opt_direction=opt_direction, threshold=threshold)

    action_list = get_smiles_all_general_action_list()

    state_dim = 768
    action_dim = len(action_list)
    print("action dim: ", action_dim)
    hidden_dim = 2048

    actor_lr = 3e-4
    critic_lr = 3e-4
    num_episodes = 1000
    gamma = 0.98
    lmbda = 0.95
    epochs = 10
    eps = 0.2
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    agent = PPO(state_dim, hidden_dim, action_dim, actor_lr, critic_lr, lmbda, epochs, eps, gamma, device)

    return_list = []
    complete_list = []
    for i in range(100):
        
        with tqdm(total=int(num_episodes/100), desc='Iteration %d' % i) as pbar:
            for i_episode in range(int(num_episodes/100)):
                print(f">>Episode {num_episodes/100 * i + i_episode+1}", file=f)
                reward = 0
                transition_dict = {'states': [], 'actions': [], 'next_states': [], 'rewards': [], 'dones': []}
                state = env.reset()
                # mask_id = [action_list.index(item) for item in get_smiles_general_action_list(env.curr_mol)]
                mask_id = [action_list.index(item) for item in action_list]
                done = False
                complete = 0
                while not done:
                    action_idx = agent.get_action(state, mask_id)
                    action = action_list[action_idx]
                    print("*"*50, file=f)
                    print(f"    Input mol: {env.curr_mol}", file=f)
                    print(f"    Input prop: {env.curr_prop}", file=f)
                    print(f"    Action: {action}", file=f)
                    next_state, reward, done = env.step(action)
                    
                    print(f"    Output mol: {env.curr_mol}", file=f)
                    print(f"    Output prop: {env.curr_prop}", file=f)
                    print(f"    Reward: {reward}", file=f)
                    print(f"    Done: {done}", file=f)
                    print("*"*50, file=f)
                
                    # mask_id.remove(action_idx)
                    transition_dict['states'].append(state)
                    transition_dict['actions'].append(action_idx)
                    transition_dict['next_states'].append(next_state)
                    transition_dict['rewards'].append(reward)
                    transition_dict['dones'].append(done)
                    state = next_state
                    if reward == 2:
                        complete = 1
                    
                return_list.append(reward * (0.5**env.depth))
                complete_list.append(complete)
                agent.update(transition_dict)
                if (i_episode+1) % 100 == 0:
                    pbar.set_postfix({'episode': '%d' % (num_episodes/100 * i + i_episode+1), 'return': '%.3f' % np.mean(return_list[-100:])})
                pbar.update(1)
                
            episodes_list = list(range(len(return_list)))
            
            with PdfPages(fig_save_dir) as pdf:
    
                plt.figure(figsize=(10, 5))
                plt.plot(episodes_list, return_list, color='blue')
                plt.xlabel('Episodes')
                plt.ylabel('Returns')
                plt.title(f'PPO Returns on {args.task_id} {constraint}')
                pdf.savefig()
                plt.close()
                
                plt.figure(figsize=(10, 5))
    
                for i, (ep, val) in enumerate(zip(episodes_list, complete_list)):
                    if val == 1:  # True
                        plt.scatter(ep, val, color='red', marker='o', s=100, label='True' if i == 0 else "")
                    else:  # False
                        plt.scatter(ep, val, color='blue', marker='x', s=100, label='False' if i == 0 else "")
                
                plt.yticks([0, 1], ['False', 'True'])
                plt.xlabel('Episodes')
                plt.ylabel('Completion Status')
                plt.title(f'Task Completion Status on {args.task_id} {constraint}')
                pdf.savefig() 
                plt.close()
            
    
    model_dict = {
        'state_dict': agent.state_dict(),
        'state_dim': state_dim,
        'hidden_dim': hidden_dim,
        'action_dim': action_dim,
        'actor_lr': actor_lr,
        'critic_lr': critic_lr,
        'lmbda': lmbda,
        'epochs': epochs,
        'eps': eps,
        'gamma': gamma,
        'device': device
    }
    torch.save(model_dict, pth_save_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PPO")

    parser.add_argument("--task_id", type=int, default=101, help="optimization task")
    parser.add_argument("--constraint", type=str, default='strict', help="loose or strict")
    
    args = parser.parse_args()
    main(args)























