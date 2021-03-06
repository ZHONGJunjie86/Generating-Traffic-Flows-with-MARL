import matplotlib.pyplot as plt
import numpy as np
import time,random
import os 
#/home/cdl/gama_workspace/GAMA_python/ Generate_Traffic_Flow_MAS_RL/GAMA_R/GAMA_intersection_data_1.csv
from_GAMA_0 = os.getcwd()+'/Generate_Traffic_Flow_MAS_RL/GAMA_R/GAMA_intersection_data_0.csv'
from_GAMA_0_0 = os.getcwd()+'/Generate_Traffic_Flow_MAS_RL/GAMA_R/GAMA_intersection_data_0_0.csv'
from_GAMA_1 = os.getcwd()+'/Generate_Traffic_Flow_MAS_RL/GAMA_R/GAMA_intersection_data_1.csv'
from_GAMA_2 = os.getcwd()+'/Generate_Traffic_Flow_MAS_RL/GAMA_R/GAMA_intersection_data_2.csv'
from_GAMA_3 = os.getcwd()+'/Generate_Traffic_Flow_MAS_RL/GAMA_R/GAMA_intersection_data_3.csv'

from_python_1 = os.getcwd()+'/Generate_Traffic_Flow_MAS_RL/GAMA_R/python_AC_1.csv'
from_python_2 = os.getcwd()+'/Generate_Traffic_Flow_MAS_RL/GAMA_R/python_AC_2.csv'

D_A_T = os.getcwd()+'/Generate_Traffic_Flow_MAS_RL/GAMA_R/D_A_T.csv'

save_curve_pic_speed = os.path.abspath(os.curdir)+'/Generate_Traffic_Flow_MAS_RL/result/Average_speed_curve.png'

def reset():
    f=open(from_GAMA_0, "r+")
    f.truncate()
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

def cross_loss_curve(critic_loss,total_rewards,save_curve_pic,save_critic_loss,save_reward,average_speed,save_speed,average_speed_NPC,save_NPC_speed):
    critic_loss = np.hstack((np.loadtxt(save_critic_loss, delimiter=","),critic_loss))
    reward = np.hstack((np.loadtxt(save_reward, delimiter=",") ,total_rewards))
    average_speeds = np.hstack((np.loadtxt(save_speed, delimiter=",") ,average_speed))
    NPC_speeds = np.hstack((np.loadtxt(save_NPC_speed, delimiter=",") ,average_speed_NPC))
    plt.plot(np.array(critic_loss), c='b', label='critic_loss',linewidth=0.5)
    plt.plot(np.array(reward), c='r', label='total_rewards',linewidth=0.5)
    plt.legend(loc='best')
    #plt.ylim(-15,15)
    plt.ylim(-0.23,0.18)
    plt.ylabel('critic_loss') #average_speed/100 m/s
    plt.xlabel('Training Episode')
    plt.grid()
    plt.savefig(save_curve_pic)
    plt.close()

   #
    plt.plot(np.array(average_speeds), c='g', label='training average_speeds',linewidth=0.5) #/100
    plt.plot(np.array(NPC_speeds), c='b', label='50 RL agent average speeds',linewidth=0.5)
    plt.legend(loc='best')
    plt.ylabel('average_speed m/s') 
    plt.xlabel('Training Episode')
    plt.grid()
    plt.savefig(save_curve_pic_speed)
    plt.close()

    np.savetxt(save_critic_loss,critic_loss,delimiter=',')
    np.savetxt(save_reward,reward,delimiter=',')
    np.savetxt(save_speed,average_speeds,delimiter=',')
    np.savetxt(save_NPC_speed,NPC_speeds,delimiter=',')

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
            state_0 = np.loadtxt(from_GAMA_0, delimiter=",")
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
            f1=open(from_GAMA_0, "r+")
            f1.truncate()
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
            error = True
   
    if  A_T[0] == 0: 
        time_pass = state_1[2]
        reward = state_1[6]
        done = state_1[7]  
        over = state_1[8] 
        average_speed_NPC =state_1[9]
        state = np.delete(state_1, [2,3,5,6,7,8,9], axis = 0) #4,5,  # 3!!!!
        state = np.array([state[0], state[0], state[0], state[1],state[1],state[1] ,state[2],state[2], state[2] ])
        return 0,state,reward,done,time_pass,over,average_speed_NPC

    elif  A_T[0] == 1: 
        try:
            state_0 = np.delete(state_0, [2,3,5,6,7,8,9], axis = 0) 
            time_pass = state_0[2]
        except (IndexError,FileNotFoundError,ValueError,OSError):
            state_0 = np.loadtxt(from_GAMA_0_0, delimiter=",")
            state_0 = np.delete(state_0, [2,3,5,6,7,8,9], axis = 0) 

            state_1 = np.delete(state_1, [2,3,5,6,7,8,9], axis = 0) #4,5,
            state_2 = np.delete(state_2, [2,3,5,6,7,8,9], axis = 0)
            state_3 = np.delete(state_3, [2,3,5,6,7,8,9], axis = 0)

        #print(state_0 ,state_1)

        state_0 = np.array([state_0[0], state_0[0], state_0[0], state_0[1],state_0[1],state_0[1] ,state_0[2],state_0[2], state_0[2] ])
        state_1 = np.array([state_1[0], state_1[0], state_1[0], state_1[1],state_1[1],state_1[1] ,state_1[2],state_1[2], state_1[2] ])
        state_2 = np.array([state_2[0], state_2[0], state_2[0], state_2[1],state_2[1],state_2[1] ,state_2[2],state_2[2], state_2[2] ])
        state_3 = np.array([state_3[0], state_3[0], state_3[0], state_3[1],state_3[1],state_3[1] ,state_3[2],state_3[2], state_3[2] ])
        
        state = [state_0,state_1,state_2,state_3]
        


        return 1,state,0,0,0,0,0#,reward,done,time_pass,over,