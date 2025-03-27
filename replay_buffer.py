import numpy as np
import random
from collections import deque

class ReplayBuffer:
    def __init__(self, size):
        self.buffer = deque(maxlen=size)
    
    def add(self, obs_t, action, reward, obs_tp1, done):
        self.buffer.append(((obs_t, action, reward, obs_tp1, done)))
    
    def sample(self, batch_size):
        samples = random.sample(self.buffer, batch_size)
        obs, act, rew, obs_tp1, done = [],[],[],[],[]
        for smp in samples: 
            obs.append(smp[0])
            act.append(smp[1]) 
            rew.append(smp[2]) 
            obs_tp1.append(smp[3]) 
            done.append(smp[4])
        return np.array(obs), np.array(act), np.array(rew), obs_tp1, np.array(done)
    
    def __len__(self):
        return len(self.buffer)