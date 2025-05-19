# train.py

import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import time
import os
from collections import deque
import random
from datetime import datetime 
import matplotlib.pyplot as plt
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'  # 允许重复加载OpenMP库

# 从其他模块导入
from env import ImageLunarLanderEnv # 环境包装器
from net import PG_Net     # 策略网络
from agent import PG_Agent # Agent (负责交互)
from config import CONFIG

print(f"Starting training on device: {CONFIG['device']}")
np.random.seed(CONFIG['seed'])
random.seed(CONFIG['seed'])
torch.manual_seed(CONFIG['seed'])
if CONFIG['device'] == "cuda":
    torch.cuda.manual_seed(CONFIG['seed'])

device = torch.device(CONFIG['device'])

# 1. 初始化环境
env = ImageLunarLanderEnv(env_id=CONFIG['env_id'],
                            num_frames=CONFIG['num_frames'],
                            img_size=CONFIG['img_size'])
state_shape = env.observation_space.shape # (C, H, W) -> (4, 256, 256)
num_actions = env.action_space.n # 4

# 2. 初始化网络
net = PG_Net(input_shape=state_shape, num_actions=num_actions).to(device)

# 3. 初始化优化器
optimizer = optim.Adam(net.parameters(), lr=CONFIG['learning_rate'])

# 4. 初始化 Agent 
agent = PG_Agent(state_shape=state_shape,
                    num_actions=num_actions,
                    device=device,
                    net=net,
                    if_eval=False)

# 5. 初始化训练状态变量
start_episode=1
history_loss=[]
history_G=[]


# 6. 加载检查点
if CONFIG["checkpoint_path"] and os.path.exists(CONFIG["checkpoint_path"]):
    print(f"Loading checkpoint from: {CONFIG['checkpoint_path']}")
    checkpoint = torch.load(CONFIG["checkpoint_path"], map_location=device)
    net.load_state_dict(checkpoint['net_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_episode = checkpoint.get('global_episode', 0)
    history_loss = checkpoint.get('history_loss', [])
    history_G = checkpoint.get('history_G', [])
    print(f"Resuming training from episode {start_episode}")

# 7.训练循环
for global_episode in range(start_episode, CONFIG['total_episodes']):
    # 执行一轮的采样
    state, _ = env.reset() # 重置状态 (NumPy:uint8)
    trajectory=[]
    done= False
    while(done==False):
        # 智能体根据初始状态选择动作
        action,log_prob = agent.select_action(state)

        # 与环境交互
        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        # 存储采样点
        #trajectory.append((state, action, reward, log_prob))
        trajectory.append((reward, log_prob))       

        # 更新状态
        state = next_state

    # 学习
    # 计算回报
    G=0
    G_=[]
    loss=0
    i=0
    for step in reversed(trajectory):
        i+=1
        #state, action, reward, log_prob=step
        reward, log_prob=step
        G=reward+G*CONFIG['gamma']
        G_.insert(0,G)
    G_ = torch.tensor(G_, dtype=torch.float32, device=device)    
    # 对回报序列进行标准化
    G_mean = G_.mean()
    G_std = G_.std()
    G_ = (G_ - G_mean) / (G_std + 1e-8) # epsilon
    # 计算损失
    log_probs_tensor = torch.stack([log_prob for _, log_prob in trajectory]) 
    loss = (-log_probs_tensor * G_).sum() 
    # 反向传播，更新策略网络
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print(f"global_episode: {global_episode},steps:{i},G:{G},loss:{loss}")
    #print(trajectory)

    # 记录训练过程变量
    history_loss.append(loss.item())
    history_G.append(G.item())

    # 定期保存模型
    if (global_episode + 1) % CONFIG['save_frequency'] == 0:
        save_dir = CONFIG['model_save_dir']
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        save_path = os.path.join(save_dir, f"PG_lunarlander_step_{global_episode+1}.pth")
        checkpoint_data = {
            'global_episode': global_episode + 1,
            'net_state_dict': net.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'history_loss':history_loss,
            'history_G':history_G
        }
        torch.save(checkpoint_data, save_path)
        print(f"\nCheckpoint saved to {save_path}")

def visualize(history_g, history_loss):
    history_g_np = np.array(history_g)   
    history_loss_np = np.array(history_loss) 
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(history_g_np, label='Episodic Return (G)', alpha=0.6)
    plt.xlabel('Episode')
    plt.ylabel('Return (G)')
    plt.title('Episodic Return over Training')
    plt.legend()
    plt.grid(True)
   
    plt.subplot(1, 2, 2)
    plt.plot(history_loss_np, label='Loss', color='green', alpha=0.6)
    plt.xlabel('Episode')
    plt.ylabel('Loss')
    plt.title('Loss over Training')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()

visualize(history_G,history_loss)
        
