import numpy as np
import torch
import torch.nn as nn
import random
from collections import deque


class ReplayBuffer(object):
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
        self.memory_full = False

    def push(self, s, a, r, s_, done, best_a):
        s = np.expand_dims(s, 0)
        s_ = np.expand_dims(s_, 0)
        self.buffer.append((s, a, r, s_, done, best_a))
        if not self.memory_full and len(self.buffer) == self.buffer.maxlen:
            self.memory_full = True

    def sample(self, batch_size):
        s, a, r, s_, done, best_a = zip(*random.sample(self.buffer, batch_size))
        return np.concatenate(s), np.array(a), r, np.concatenate(s_), done, np.array(best_a)

    def __len__(self):
        return len(self.buffer)


class QNetwork(nn.Module):
    def __init__(self, s_dim, a_dim):
        super(QNetwork, self).__init__()

        self.feature = nn.Sequential(
            nn.Linear(s_dim, 128),
            nn.ReLU(),
        )

        self.advantage = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, a_dim)
        )

        self.value = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, s):
        x = self.feature(s)
        advantage = self.advantage(x)
        value = self.value(x)
        return value + advantage - advantage.mean()   # Q(s,a)


class D3QN(object):
    gamma = 0.99
    buffer_capacity = 300
    batch_size = 16
    tau = 0.005  # 软更新目标网的程度

    def __init__(self, s_dim, a_dim):
        self.s_dim = s_dim
        self.a_dim = a_dim
        self.current_net = QNetwork(s_dim, a_dim)
        self.target_net = QNetwork(s_dim, a_dim)
        self.optimizer = torch.optim.Adam(self.current_net.parameters(), lr=0.01)
        self.replay_buffer = ReplayBuffer(self.buffer_capacity)
        self.loss_func = nn.MSELoss()

    def choose_action(self, s, mask, epsilon):
        if sum(mask) == 0:
            return -1
        if random.random() < epsilon:
            mask[mask == 0] = float("-inf")
            s = torch.FloatTensor(s)
            q_values = self.current_net(s).data.numpy()
            action = np.argmax(q_values + mask)   # 取出最大值下标
            return action
        else:
            mask = np.array(mask, dtype=bool).tolist()
            actions = np.array([i for i in range(self.a_dim)])[mask]
            action = random.choice(actions)
        return action

    def learn(self):
        s, a, r, s_, done, best_a = self.replay_buffer.sample(self.batch_size)
        s = torch.FloatTensor(s)
        a = torch.LongTensor(a)
        r = torch.FloatTensor(r).unsqueeze(1)
        s_ = torch.FloatTensor(s_)
        done = torch.FloatTensor(done).unsqueeze(1)
        best_a = torch.LongTensor(best_a)

        q_values = self.current_net(s)
        q_value = torch.sum(q_values * a, dim=1).unsqueeze(1)   # (batch,1)

        next_q_values = self.target_net(s_)
        next_q_value = torch.sum(next_q_values * best_a, dim=1).unsqueeze(1)
        expected_q_value = r + (1-done) * self.gamma * next_q_value

        loss = self.loss_func(q_value, expected_q_value)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # 软更新
        for param, target_param in zip(self.current_net.parameters(), self.target_net.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def save(self):
        torch.save(self.current_net.state_dict(), "params/current_net.pth")
        torch.save(self.target_net.state_dict(), "params/target_net.pth")

    def load(self):
        self.current_net.load_state_dict(torch.load("params/current_net.pth"))
        self.target_net.load_state_dict(torch.load("params/target_net.pth"))