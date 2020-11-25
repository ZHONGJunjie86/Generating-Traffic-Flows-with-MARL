import matplotlib.pyplot as plt
import numpy as np
import time,random
import os 

from_GAMA_1 = os.getcwd()+'\\GAMA_python\\Generate_Traffic_Flow_MAS_RL\\GAMA_R\\GAMA_intersection_data_1.csv'
from_GAMA_2 = os.getcwd()+'\\GAMA_python\\Generate_Traffic_Flow_MAS_RL\\GAMA_R\\GAMA_intersection_data_2.csv'
from_GAMA_3 = os.getcwd()+'\\GAMA_python\\Generate_Traffic_Flow_MAS_RL\\GAMA_R\\GAMA_intersection_data_3.csv'
from_python_1 = os.getcwd()+'\\GAMA_python\\Generate_Traffic_Flow_MAS_RL\\GAMA_R\\python_AC_1.csv'
from_python_2 = os.getcwd()+'\\GAMA_python\\Generate_Traffic_Flow_MAS_RL\\GAMA_R\\python_AC_2.csv'

D_A_T = os.getcwd()+'/Generate_Traffic_Flow_MAS_RL/GAMA_R/D_A_T.csv'

def reset():
    f=open(from_GAMA_1, "r+")
    f.truncate()
    f=open(from_GAMA_2, "r+")
    f.truncate()
    f=open(from_GAMA_3, "r+")
    f.truncate()
    f=open(from_python_1, "r+")
    f.truncate()
    f=open(from_python_2, "r+")
    f.truncate()
    f4=open(D_A_T, "r+")
    f4.truncate()
    return_ = [0]
    np.savetxt(from_python_1,return_,delimiter=',')
    np.savetxt(from_python_2,return_,delimiter=',')

def cross_loss_curve(critic_loss,total_rewards,save_curve_pic,save_critic_loss,save_reward):
    critic_loss = np.hstack((np.loadtxt(save_critic_loss, delimiter=","),critic_loss))
    reward = np.hstack((np.loadtxt(save_reward, delimiter=",") ,total_rewards))
    plt.plot(np.array(critic_loss), c='b', label='critic_loss',linewidth=0.5)
    plt.plot(np.array(reward), c='r', label='total_rewards',linewidth=0.5)
    plt.legend(loc='best')
    #plt.ylim(-15,15)
    plt.ylim(-0.25,0.1)
    plt.ylabel('critic_loss') 
    plt.xlabel('training steps')
    plt.grid()
    plt.savefig(save_curve_pic)
    plt.close()
    np.savetxt(save_critic_loss,critic_loss,delimiter=',')
    np.savetxt(save_reward,reward,delimiter=',')

def send_to_GAMA(to_GAMA):
    error = True
    while error == True:
        try:
            np.savetxt(from_python_1,to_GAMA,delimiter=',')
            np.savetxt(from_python_2,to_GAMA,delimiter=',')
            error = False
        except(IndexError,FileNotFoundError,ValueError,OSError,PermissionError):  
            error = True 

#[real_speed/10, target_speed/10, elapsed_time_ratio, distance_left/100,distance_front_car/10,distance_behind_car/10,reward,done,over]
def GAMA_connect(test=0):
    error = True
    while error == True:
        try:
            state_1 = np.loadtxt(from_GAMA_1, delimiter=",")
            state_2 = np.loadtxt(from_GAMA_2, delimiter=",")
            state_3 = np.loadtxt(from_GAMA_3, delimiter=",")
            A_T= np.loadtxt(D_A_T, delimiter=",")
            time_pass = state_1[2];time_pass = state_2[2];time_pass = state_3[2];test = A_T[1]
            error = False
        except (IndexError,FileNotFoundError,ValueError,OSError):
            error = True

    error = True
    while error == True:
        try:
            f1=open(from_GAMA_1, "r+")
            f1.truncate()
            f2=open(from_GAMA_2, "r+")
            f2.truncate()
            f3=open(from_GAMA_3, "r+")
            f3.truncate()
            f4=open(D_A_T, "r+")
            f4.truncate()
            error = False

        except (IndexError,FileNotFoundError,ValueError,OSError):
            time.sleep(0.003)
            error = True

   
    if  A_T[0] == 0: 
        time_pass = state_1[2]
        reward = state_1[6]
        done = state_1[7]  
        over = state_1[8] 
        state = np.delete(state_1, [2,3,5,6,7,8], axis = 0) #4,5,  # 3!!!!
        return 0,state,reward,done,time_pass,over,
    elif  A_T[0] == 1: 
        state_1 = np.delete(state_1, [2,3,5,6,7,8], axis = 0) #4,5,
        state_2 = np.delete(state_2, [2,3,5,6,7,8], axis = 0)
        state_3 = np.delete(state_3, [2,3,5,6,7,8], axis = 0)
        state = [state_1,state_2,state_3]
        return 1,state,0,0,0,0#,reward,done,time_pass,over,