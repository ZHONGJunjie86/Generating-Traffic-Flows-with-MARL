from utils import cross_loss_curve, GAMA_connect,reset,send_to_GAMA
from CV_input import generate_img
import os
from itertools import count
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import MultivariateNormal
import numpy as np
import pandas as pd
import warnings

warnings.filterwarnings("ignore")

device = torch.device("cuda" if torch.cuda.is_available() else"cpu")  

state_size = 6
action_size = 1 
torch.set_default_tensor_type(torch.DoubleTensor)

class Memory:
    def __init__(self,h_cv,h_n):
        self.h_state_cv_a = h_cv
        self.h_state_n_a = h_n 
    
    def set_hidden(self,h_cv,h_n):
        self.h_state_cv_a = h_cv
        self.h_state_n_a = h_n
"""  # LSTM
class Actor(nn.Module):
    def __init__(self, state_size, action_size):
        super(Actor, self).__init__()
        self.conv1 = nn.Conv2d(3,8, kernel_size=8, stride=4, padding=0) # 500*500*3 -> 124*124*8
        self.maxp1 = nn.MaxPool2d(4, stride = 2, padding=0) # 124*124*8 -> 61*61*8
        self.conv2 = nn.Conv2d(8, 16, kernel_size=4, stride=1, padding=0) # 61*61*8 -> 58*58*16 
        self.maxp2 = nn.MaxPool2d(2, stride=2, padding=0) # 58*58*16  -> 29*29*16 = 13456
        self.linear_CNN = nn.Linear(13456, 256)   # *3
        self.lstm_CNN = nn.LSTM(256,85,batch_first=True)
        
        #
        self.state_size = state_size
        self.action_size = action_size
        self.linear1 = nn.Linear(self.state_size, 128)
        self.linear2 = nn.Linear(128,128)
        self.lstm3 = nn.LSTM(128,85,batch_first=True)
        
        #self.LSTM_layer_3 = nn.LSTM(511,128,1, batch_first=True)
        self.linear3 = nn.Linear(510,128)
        self.linear4 = nn.Linear(128,32)
        self.mu = nn.Linear(32,self.action_size)  #256 linear2
        self.sigma = nn.Linear(32,self.action_size)

    def forward(self, state,tensor_cv,h_state_cv_a=(torch.zeros(1,1,85).to(device),
                            torch.zeros(1,1,85).to(device)),h_state_n_a=(torch.zeros(1,3,85).to(device),
                            torch.zeros(1,3,85).to(device))):
        # CV
        x = F.relu(self.maxp1(self.conv1(tensor_cv)))
        x = F.relu(self.maxp2(self.conv2(x)))#.reshape(3,1,13456)
        x = x.view(x.size(0), -1) #[3, 16, 29, 29]
        x = F.relu(self.linear_CNN(x))#.reshape(3,1,256)
        x,h_state_cv = self.lstm_CNN(x.unsqueeze(0),h_state_cv_a)    #.unsqueeze(0)
        x = F.relu(x).reshape(1,255)  #torch.tanh
        
        # num
        output_1 = F.relu(self.linear1(state))
        output_2 = F.relu(self.linear2(output_1))
        output_2,h_state_n_a = self.lstm3(output_2,h_state_n_a)
        output_2 = F.relu(output_2) .squeeze().reshape(1,255) 
        # LSTM
        output_2 = torch.cat((x,output_2),1) 
        output_3 = F.relu(self.linear3(output_2))
        #
        output_4 = F.relu(self.linear4(output_3))#.view(-1,c))) #
        mu = torch.tanh(self.mu(output_4))   #有正有负 sigmoid 0-1
        sigma = F.relu(self.sigma(output_4)) + 0.001 
        mu = torch.diag_embed(mu).to(device)
        sigma = torch.diag_embed(sigma).to(device)  # change to 2D
        dist = MultivariateNormal(mu,sigma)  #N(μ，σ^2)
        entropy = dist.entropy().mean()
        action = dist.sample()
        action_logprob = dist.log_prob(action)     
        return action,(h_state_cv_a[0].data,h_state_cv_a[1].data),(h_state_n_a[0].data,h_state_n_a[1].data)
"""
class Actor(nn.Module):
    def __init__(self, state_size, action_size):
        super(Actor, self).__init__()
        self.conv1 = nn.Conv2d(3,8, kernel_size=8, stride=4, padding=0) # 500*500*3 -> 124*124*8
        self.maxp1 = nn.MaxPool2d(4, stride = 2, padding=0) # 124*124*8 -> 61*61*8
        self.conv2 = nn.Conv2d(8, 16, kernel_size=4, stride=1, padding=0) # 61*61*8 -> 58*58*16 
        self.maxp2 = nn.MaxPool2d(2, stride=2, padding=0) # 58*58*16  -> 29*29*16 = 13456
        self.linear_CNN_1 = nn.Linear(13456, 256)
        self.linear_CNN_2 = nn.Linear(768,256)
        
        #
        self.state_size = state_size
        self.action_size = action_size
        self.linear1 = nn.Linear(self.state_size, 128)
        self.linear2 = nn.Linear(128, 85)
        
        self.linear3 = nn.Linear(511,128)
        self.linear4 = nn.Linear(128,32)
        self.mu = nn.Linear(32,self.action_size)  #256 linear2
        self.sigma = nn.Linear(32,self.action_size)
        self.hidden_cell = (torch.zeros(1,1,64).to(device),
                            torch.zeros(1,1,64).to(device))

    def forward(self, state,tensor_cv):
        # CV
        x = F.relu(self.maxp1(self.conv1(tensor_cv)))
        x = F.relu(self.maxp2(self.conv2(x)))
        x = x.view(x.size(0), -1) #展開
        x = F.relu(self.linear_CNN_1(x)).reshape(1,768)
        x = F.relu(self.linear_CNN_2(x)).reshape(1,256)
        # num
        output_1 = F.relu(self.linear1(state))
        output_2 = F.relu(self.linear2(output_1)).reshape(1,255)
        # merge
        output_2 = torch.cat((x,output_2),1) 
        output_3 = F.relu(self.linear3(output_2) )
        #
        output_4 =F.relu(self.linear4(output_3)) #F.relu(self.linear4(output_3.view(-1,c))) #
        mu = torch.tanh(self.mu(output_4))   #有正有负 sigmoid 0-1
        sigma = F.relu(self.sigma(output_4)) + 0.001 
        mu = torch.diag_embed(mu).to(device)
        sigma = torch.diag_embed(sigma).to(device)  # change to 2D
        dist = MultivariateNormal(mu,sigma)  #N(μ，σ^2)
        entropy = dist.entropy().mean()
        action = dist.sample()
        action_logprob = dist.log_prob(action)     
        action = torch.clamp(action.detach(), -0.8, 0.6)
        return action,action_logprob,entropy

def main():
    ################ load ###################
    actor_path = os.path.abspath(os.curdir)+'/Generate_Traffic_Flow_MAS_RL/weight/AC_TD2_actor.pkl'
    if os.path.exists(actor_path):
        actor =  Actor(state_size, action_size).to(device)
        actor.load_state_dict(torch.load(actor_path))
        print('Actor Model loaded')
    else:
        actor = Actor(state_size, action_size).to(device)
    print("Waiting for GAMA...")
    ################### initialization ########################
    reset()

    Using_LSTM = False
    test = "GAMA"
    N_agent = 20
    list_hidden = []

    count = 0
    ##################  start  #########################
    state = GAMA_connect(test)
    print("Connected")
    while True:
        if  Using_LSTM == False:
            state = [torch.DoubleTensor(elem).reshape(1,state_size).to(device) for elem in state]  
            state = torch.stack(state).to(device).detach()
            tensor_cv = generate_img() 
            tensor_cv = [torch.from_numpy(np.transpose(elem, (2, 0, 1))).double().to(device)/255 for elem in tensor_cv]  
            tensor_cv = torch.stack(tensor_cv).to(device).detach()

            action,h_state_cv_a,h_state_n_a = actor(state,tensor_cv)
            
            send_to_GAMA([[1,float(action.cpu().numpy()*10)]])

        else:
            if len(list_hidden) < N_agent:
                state = [torch.DoubleTensor(elem).reshape(1,state_size).to(device) for elem in state]  
                state = torch.stack(state).to(device).detach()
                tensor_cv = generate_img() 
                tensor_cv = [torch.from_numpy(np.transpose(elem, (2, 0, 1))).double().to(device)/255 for elem in tensor_cv]  
                tensor_cv = torch.stack(tensor_cv).to(device).detach()

                action,h_state_cv_a,h_state_n_a = actor(state,tensor_cv)
                
                send_to_GAMA([[1,float(action.cpu().numpy()*10)]])
                list_hidden.append(Memory(h_state_cv_a,h_state_n_a)) 
                count += 1

            else:
                state = [torch.DoubleTensor(elem).reshape(1,state_size).to(device) for elem in state] 
                state = torch.stack(state).to(device).detach()
                tensor_cv = generate_img() 
                tensor_cv = [torch.from_numpy(np.transpose(elem, (2, 0, 1))).double().to(device)/255 for elem in tensor_cv]  
                tensor_cv = torch.stack(tensor_cv).to(device).detach()

                action,h_state_cv_a,h_state_n_a = actor(state,tensor_cv,
                                        list_hidden[count%N_agent].h_state_cv_a,list_hidden[count%N_agent].h_state_n_a)
                
                send_to_GAMA([[1,float(action.cpu().numpy()*10)]])
                list_hidden[count%N_agent].set_hidden(h_state_cv_a,h_state_n_a)
                count += 1

        state = GAMA_connect(test)

    return None 

if __name__ == '__main__':
    main()

