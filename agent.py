# agent.py 

import torch
import torch.nn as nn
import torch.distributions as D
import numpy as np
import random
from typing import Tuple

from net import PG_Net

class PG_Agent:
    """
    PG 智能体，只负责动作选择。
    """

    def __init__(self,
                 state_shape,
                 num_actions: int,
                 device: torch.device,
                 net: nn.Module,
                 if_eval = False): 
        """
        初始化 PG_Agent。

        Args:
            state_shape (Tuple[int, int, int]): 输入状态的形状 (frames, height, width)。
            num_actions (int): 可执行动作的数量。
            device (torch.device): 计算设备 ('cpu' or 'cuda')。
            net (nn.Module): 用于决策的策略网络实例。
        """
        self.state_shape=state_shape
        self.num_actions = num_actions
        self.device = device
        self.net = net
        self.if_eval=if_eval

        if self.if_eval:
            self.net.eval()
        else:
            self.net.train()
        
        print(f"Agent initialized with {num_actions} actions on device {device}.")

    def select_action(self, state_np: np.ndarray) -> int:
        """
        将当前状态输入策略网络，根据输出概率随机选择一个动作。

        Args:
            state_np (np.ndarray): 当前环境状态 (预处理、堆叠后的 NumPy 数组, dtype=uint8)。

        Returns:
            int: 选择的动作索引。
        """

        state_tensor = torch.from_numpy(state_np).unsqueeze(0).to(self.device).float() / 255.0
        action_logits = self.net(state_tensor) # Shape: (1, num_actions)
        if not self.if_eval:
            action_distribution = D.Categorical(logits=action_logits)
            action_tensor = action_distribution.sample()
            log_prob = action_distribution.log_prob(action_tensor)
            return action_tensor.item(), log_prob.squeeze() # 返回动作标量和 log_prob (移除批次维度)
        else:
            action = action_logits.argmax(dim=1).item()
            return action