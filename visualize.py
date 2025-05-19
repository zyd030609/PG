# visualize.py

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

device = torch.device(CONFIG['device'])

print(f"Loading checkpoint from: {CONFIG['checkpoint_path']}")
checkpoint = torch.load("D:\WORKS\py_works\RL_EXPERIMENT\mine\PG\models\PG_lunarlander_step_60000.pth", map_location=device)
start_episode = checkpoint.get('global_episode', 0)
history_loss = checkpoint.get('history_loss', [])
history_G = checkpoint.get('history_G', [])
print(f"Resuming training from episode {start_episode}")

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