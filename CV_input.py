import csv
import matplotlib.pyplot as plt
import numpy as np
import os 
import cv2
import io
from PIL import Image  

file_head = os.getcwd()+'/Generate_Traffic_Flow_MAS_RL/GAMA_img/'
save_img = os.getcwd()+'/Generate_Traffic_Flow_MAS_RL/GAMA_img/save_agents.png'

def generate_img(): 
    list_img = []
    location_route = file_head+'route.csv'
    for i in range(3):
        location_NPC_front = file_head + 'save_NPC_front_'+str(i+1)+'.csv'
        location_NPC_behind = file_head + 'save_NPC_behind_'+str(i+1)+'.csv'
        location_NPC_closest_10 = file_head + 'save_NPC_'+str(i+1)+'.csv'
        location_self =file_head +'save_self_'+str(i+1)+'.csv'
        error = True
        while error == True:
            try:
                NPC_front = []
                NPC_behind = []
                NPC_closest_10 = []
                Route = []
                SELF = []
                with open(location_NPC_front)as f:
                    f_csv = csv.reader(f)
                    for i in f_csv:
                        NPC_front = i
                with open(location_NPC_behind)as f:
                    f_csv = csv.reader(f)
                    for i in f_csv:
                        NPC_behind = i
                with open(location_NPC_closest_10)as f:
                    f_csv = csv.reader(f)
                    for i in f_csv:
                        NPC_closest_10 = i
                with open(location_route)as f:
                    f_csv = csv.reader(f)
                    for i in f_csv:
                        Route = i
                with open(location_self)as f:
                    f_csv = csv.reader(f)
                    for i in f_csv:
                        SELF = i

                NPC_front_X = []
                NPC_front_Y = []
                NPC_behind_X = []
                NPC_behind_Y = []
                NPC_closest_10_X = []
                NPC_closest_10_Y = []
                Route_X = []
                Route_Y = []
                count = 1
                for i in NPC_front:
                    if count == 1 :
                        NPC_front_X.append(float(i[1:]))
                    elif(count == 2):
                        NPC_front_Y.append(float(i))
                    else:
                        count = 0
                    count += 1

                for i in NPC_behind:
                    if count == 1 :
                        NPC_behind_X.append(float(i[1:]))
                    elif(count == 2):
                        NPC_behind_Y.append(float(i))
                    else:
                        count = 0
                    count += 1
                for i in NPC_closest_10:
                    if count == 1 :
                        NPC_closest_10_X.append(float(i[1:]))
                    elif(count == 2):
                        NPC_closest_10_Y.append(float(i))
                    else:
                        count = 0
                    count += 1
                for i in Route:
                    if count == 1 :
                        try:
                            Route_X.append(float(i[1:]))
                        except(ValueError):
                            Route_X.append(float(i[2:]))
                    elif(count == 2):
                        Route_Y.append(float(i))
                    else:
                            count = 0
                    count += 1
                plt.figure(figsize=(5, 5)) #500*500
                plt.axis('off') 
                plt.xlim(float(SELF[0][1:])-100,float(SELF[0][1:])+100)
                plt.ylim(float(SELF[1])-100, float(SELF[1])+100)
                plt.scatter(Route_X[0],Route_Y[0],color = 'g',marker = 'h') #start
                plt.scatter(Route_X[len(Route_X)-1],Route_Y[len(Route_Y)-1],color = 'purple',marker = 'h') #end
                plt.plot(Route_X,Route_Y,color = 'grey',alpha=0.7)   #route
                plt.scatter(NPC_behind_X, NPC_behind_Y,color = 'b',marker = 'o',s=12) #NPC_behind
                plt.scatter(NPC_front_X, NPC_front_Y,color = 'c',marker = '>',s=12) #NPC_front
                plt.scatter(NPC_closest_10_X, NPC_closest_10_Y,color = 'm',marker = 'P',s=12) #NPC_10
                plt.scatter(float(SELF[0][1:]), float(SELF[1]),color = 'r',marker= 'D',s=12)
                error = False

            except(IndexError,FileNotFoundError,ValueError,OSError,PermissionError):
                error = True
        
        buffer =io.BytesIO()
        plt.savefig( buffer,dpi=100) #save_img
        buffer.seek(0) 
        img = np.asarray(bytearray(buffer.read()), dtype=np.uint8)
        img_cv = cv2.imdecode(img,1)  #0-grey
        #img_cv = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
        #cv2.imwrite(save_img,img_cv)
        list_img.append(img_cv) 
        
        """
        buffer =io.BytesIO()
        plt.savefig( buffer,dpi=100) #save_img
        plt.close()
        buffer.seek(0) 
        img = Image.open(buffer)
        img.save(save_img)

        #Image.fromarray(np.array(imgdata.buf)).save('save_img')
        img_cv = cv2.imread(save_img)   # save_  cv2.imread()------np.array, (H x W xC), [0, 255], BGR
        img_cv = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
        list_img.append(img_cv) 
        """
        buffer.close()

    return list_img

def generate_img_train(): 
    location_NPC_front = os.getcwd()+'/Generate_Traffic_Flow_MAS_RL/GAMA_img/train/save_NPC_front.csv'
    location_NPC_behind = os.getcwd()+'/Generate_Traffic_Flow_MAS_RL/GAMA_img/train/save_NPC_behind.csv'
    location_NPC_closest_10 = os.getcwd()+'/Generate_Traffic_Flow_MAS_RL/GAMA_img/train/save_NPC_closest_10.csv'
    location_route = os.getcwd()+'/Generate_Traffic_Flow_MAS_RL/GAMA_img/train/route.csv'
    location_self =os.getcwd()+"/Generate_Traffic_Flow_MAS_RL/GAMA_img/train/save_self.csv"
    save_img = os.getcwd()+'/Generate_Traffic_Flow_MAS_RL/GAMA_img/train/save_agents.png'
    error = True
    while error == True:
        try:
            NPC_front = []
            NPC_behind = []
            NPC_closest_10 = []
            Route = []
            SELF = []
            with open(location_NPC_front)as f:
                f_csv = csv.reader(f)
                for i in f_csv:
                    NPC_front = i
            with open(location_NPC_behind)as f:
                f_csv = csv.reader(f)
                for i in f_csv:
                    NPC_behind = i
            with open(location_NPC_closest_10)as f:
                f_csv = csv.reader(f)
                for i in f_csv:
                    NPC_closest_10 = i
            with open(location_route)as f:
                f_csv = csv.reader(f)
                for i in f_csv:
                    Route = i
            with open(location_self)as f:
                f_csv = csv.reader(f)
                for i in f_csv:
                    SELF = i

            NPC_front_X = []
            NPC_front_Y = []
            NPC_behind_X = []
            NPC_behind_Y = []
            NPC_closest_10_X = []
            NPC_closest_10_Y = []
            Route_X = []
            Route_Y = []
            count = 1
            for i in NPC_front:
                if count == 1 :
                    NPC_front_X.append(float(i[1:]))
                elif(count == 2):
                    NPC_front_Y.append(float(i))
                else:
                    count = 0
                count += 1

            for i in NPC_behind:
                if count == 1 :
                    NPC_behind_X.append(float(i[1:]))
                elif(count == 2):
                    NPC_behind_Y.append(float(i))
                else:
                    count = 0
                count += 1
            for i in NPC_closest_10:
                if count == 1 :
                    NPC_closest_10_X.append(float(i[1:]))
                elif(count == 2):
                    NPC_closest_10_Y.append(float(i))
                else:
                    count = 0
                count += 1
            for i in Route:
                if count == 1 :
                    try:
                        Route_X.append(float(i[1:]))
                    except(ValueError):
                        Route_X.append(float(i[2:]))
                elif(count == 2):
                    Route_Y.append(float(i))
                else:
                        count = 0
                count += 1
            plt.figure(figsize=(5, 5)) #500*500
            plt.axis('off') 
            plt.xlim(float(SELF[0][1:])-100,float(SELF[0][1:])+100)
            plt.ylim(float(SELF[1])-100, float(SELF[1])+100)
            plt.scatter(Route_X[0],Route_Y[0],color = 'g',marker = 'h') #start
            plt.scatter(Route_X[len(Route_X)-1],Route_Y[len(Route_Y)-1],color = 'purple',marker = 'h') #end
            plt.plot(Route_X,Route_Y,color = 'grey',alpha=0.7)   #route
            plt.scatter(NPC_behind_X, NPC_behind_Y,color = 'b',marker = 'o',s=12) #NPC_behind
            plt.scatter(NPC_front_X, NPC_front_Y,color = 'c',marker = '>',s=12) #NPC_front
            plt.scatter(NPC_closest_10_X, NPC_closest_10_Y,color = 'm',marker = 'P',s=12) #NPC_10
            plt.scatter(float(SELF[0][1:]), float(SELF[1]),color = 'r',marker= 'D',s=12)
            error = False
        except(IndexError,FileNotFoundError,ValueError,OSError,PermissionError):
            error = True
    
    buffer =io.BytesIO()
    plt.savefig( buffer,dpi=100) #save_img
    plt.close()
    buffer.seek(0) 
    img = Image.open(buffer)
    img.save(save_img)
    
    img_cv = cv2.imread(save_img)   # cv2.imread()------np.array, (H x W xC), [0, 255], BGR
    img_cv = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
    buffer.close()
    
    return img_cv