import torch 
import torch.nn as nn
import numpy as np
from torch.distributions.bernoulli import Bernoulli

device = "cpu"


class Model():
    def __init__(self, neur_cnt, astr_cnt, in_size, out_size, gamma=0.1, tau=0.01):
        self.n = neur_cnt
        self.m = astr_cnt
        self.k = in_size
        self.o = out_size
        
        self.gamma = gamma
        self.tau = tau

        self.C = nn.Parameter(torch.Tensor(self.n * self.n))
        self.D = nn.Parameter(torch.Tensor(self.n * self.n, self.m))
        self.F = nn.Parameter(torch.Tensor(self.m))
        self.H = nn.Parameter(torch.Tensor(self.m, self.n * self.n))

        self.W_in_1 = nn.Parameter(torch.Tensor(self.n, self.k))
        self.W_in_2 = nn.Parameter(torch.Tensor(self.n, self.k))


        with torch.no_grad():
            self.C.normal_(std = 1. / np.sqrt(self.n * self.n))
            self.D.normal_(std = 1. / np.sqrt(self.n * self.n))
            self.F.normal_(std = 1. / np.sqrt(self.m))
            self.H.normal_(std = 1. / np.sqrt(self.m))
            self.W_in_1.uniform_(-1. / np.sqrt(self.k), 1. / np.sqrt(self.k))
            self.W_in_2.uniform_(-1. / np.sqrt(self.k), 1. / np.sqrt(self.k))
            
        
        x = torch.zeros(self.n)
        W = torch.zeros(self.n * self.n)
        z = torch.zeros(self.m)



        def phi(x):
            return torch.sigmoid(x)

        def Phi(x):
            xp = phi(x).flatten()

        def psi(x):
            return torch.tanh(z)


        def forward(self, state, I):
            if state is None:
                x = torch.zeros(self.n)
                W = torch.zeros(self.n, self.n)
                z = torch.zeros(self.m)
            else:
                x, W, z = state

            x = (1 - gamma) * x + gamma * self.W @ phi(x) + self.W_in_1 @ I
            W = (1. - gamma) * W + gamma * ((torch.diag(self.C) @ Phi(x)) + self.D @ psi(z)).reshape(self.n, self.n)
            z = (1. - gamma * tau) * z + gamma * tau * (torch.diag(self.F) @ psi(z) + self.H @ Phi(x) + self.W_in_2 @ I)



def SBMAB(action_cnt, samples_cnt):  # standard Bernoulli multi-armed bandit  
    Mu = torch.rand((action_cnt))
    return Bernoulli(probs=Mu), samples_cnt



class NetModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = Model(n=128, m=64, k=1, o=3)
        self.affine_transformation = nn.Linear(128, 3)

    def forward(self, x, state):
        x, state = self.net(x, state)
        y = nn.affine_transformation(x)
        return y, state


class RL():
    def __init__(self, in_size, out_size):
        self.model = NetModule()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)


        
    def train(self, arm_samples):
        self.model.train()
        regret_rec = np.array([])
        expected_reward = 0
        cumulative_regret = 0



        state = None
        
        avg_rewards = 0
        L = 0
        
        for t in range(arm_samples.size()):

            L += (avg_rewards - reward) * 



            L.backward()
            self.optimizer.zero_grad()
            

        return
    
    def (self, T):
        
        state = torch.ones(1).unsqueeze




A = SBMAB(action_cnt=3, samples_cnt=10)