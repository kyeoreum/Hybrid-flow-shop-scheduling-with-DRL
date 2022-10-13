# Simulator
import os
from re import A
from unittest import result
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import random
from collections import deque
import copy
import tensorflow as tf
import tensorflow.keras as keras
import numpy as np
from tensorflow.keras.layers import Input, Dense, LSTM
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.optimizers import Adam

model = keras.models.load_model("C:/Users/user/Desktop/GP_DQN.h5")
target_model = keras.models.load_model("C:/Users/user/Desktop/GP_DQN.h5")


SEED = 42
os.environ['PYTHONHASHSEED'] = str(SEED)
os.environ['TF_DETERMINISTIC_OPS'] = '1'
tf.random.set_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

job = [30,50,70]
stage = [5, 10]
alternative_machine = [3,5,7]
action_list = []

# rescheduling point - get system states
def get_feature(stage_queue, machine_state2):
    
    features = []

    #0 total tardiness
    Td1 = 0
    Td2 = 0
    
    machine_state = copy.deepcopy(machine_state2)
    for m in range(0, len(machine_state)):
        if machine_state[m] == 9999:
            machine_state[m] = 0

    
    for q in stage_queue:
        st = len(completion_time[q])
        tmp = machine_state[0]
        for m in range(0, len(machine_state)):
            if tmp >= machine_state[m]:
                tmp = machine_state[m]
       
        index = machine_state.index(tmp)
        if st < stage_num:
            machine_state[index] += re_job_temp[q][3][st]
        if job_temp[q][-1][0] - (machine_state[index] + job_flow[q]) < 0:
            Td1 += 1
    if Td1 > 1:
        Td1 /= len(stage_queue)
        
    features.append(Td1)

    #1 Estimated tardiness
    Et1 = 0
    Et2 = 0
    machine_state = copy.deepcopy(machine_state2)
    for m in range(0, len(machine_state)):
        if machine_state[m] == 9999:
            machine_state[m] = 0
    
    for q in stage_queue:
        st = len(completion_time[q])
        tmp = machine_state[0]
        for m in range(0, len(machine_state)):
            if tmp >= machine_state[m]:
                tmp = machine_state[m]
        index = machine_state.index(tmp)
        Et1 = machine_state[index] + job_flow[q]
        if st < stage_num:
            machine_state[index] += job_temp[q][3][st]
        re_pt = 0
        for t in range(st, stage_num):
            re_pt += re_job_temp[q][3][t]
        Et1 += re_pt
        if job_temp[q][-1][0] - Et1 < 0:
            Et2 += 1
    if Et2 > 1:
        Et2 /= len(stage_queue)
    
    features.append(Et2)


    #2 QTL 여유 job

    for m in range(0, len(machine_state)):
        if machine_state[m] == 9999:
            machine_state[m] = 0

    cnt = 0
    for q in stage_queue:
        st = len(completion_time[q])
        tmp = machine_state[0]
        for m in range(0, len(machine_state)):
            if tmp >= machine_state[m]:
                tmp = machine_state[m]
       
        index = machine_state.index(tmp)
  
        if  job_temp[q][1][0] - machine_state[index] < 0 or job_temp[q][2][0] - machine_state[index] < 0:
            cnt += 1
        if st < stage_num:
            machine_state[index] += re_job_temp[q][3][st]
        
    if cnt > 1:
        cnt /= len(stage_queue)

    features.append(cnt)

        
    #3 rework job
    R1 = 0
    R2 = 0
    for q in stage_queue:
        if job_temp[q][-1][1] != 0:
            R1 += 1
    if R1 > 0:
            R1 = R1 / len(stage_queue)
    features.append(R1)


    #4 number of job
    features.append(len(stage_queue) / job_num)

    #5 mean remaining time
    mrt = 0
    tmp = 0
    for q in stage_queue:
        mrt = job_flow[q] / job_temp[q][-1][0]
        if mrt > 1:
            tmp += 1
    if tmp > 1:
        tmp /= len(stage_queue)
    features.append(tmp)
    
    #6 number of empty machine
    cnt = 0
    for m in machine_state:
        if m == 9999:
            cnt += 1
    if cnt > 1:
        cnt /= len(machine_state)
    features.append(cnt)
    

    return features


def fit_eval(i,sq, rule_num, job_temp, stage_queue, job_flow, completion_time, re_job_temp): 

    save_rule_result = []
    if len(sq) > 1:
        for o in range(0, len(sq)): 
            try:           
                save_rule_result.append(eval(rule_num))
            except ZeroDivisionError:
                save_rule_result.append(0)

    for op in range(0, len(save_rule_result)): 
        for op2 in range(0, len(save_rule_result)):     
            if save_rule_result[op] < save_rule_result[op2]:                
                sq[op], sq[op2] = sq[op2], sq[op]
                save_rule_result[op], save_rule_result[op2] = save_rule_result[op2], save_rule_result[op]




# episode (all instainces)

for jn in job:
    for sn in stage:
        for al in alternative_machine:

            
                
            for num in range(0, 10):
                action_list = []
                f = open("C:/Users/user/source/repos/my_thesis/TESTData2/TESTData_"+str(jn)+"_"+str(sn)+"_"+str(al)+"_"+str(num)+".txt", 'r')                            
                action_list = []

                ##### Data Preprocessing #####

                buff = f.readline()
                temp = []
                while buff != '':
                    temp.append(buff)
                    buff = f.readline()

                job_num = int(temp[0])
                stage_num = int(temp[1])
                machine_num = int(temp[2])
                del temp[0:5]

                batch_size = job_num * stage_num

                job_temp = []
                i = 0
                while i < len(temp):
                    job_temp.append(temp[i:i+6])
                    i += 6


                for i in range(0, len(job_temp)):
                    for j in range(0, len(job_temp[i])):
                        job_temp[i][j] = job_temp[i][j].split(' ')
                        for k in range(0, len(job_temp[i][j])):
                            job_temp[i][j][k] = job_temp[i][j][k].rstrip('\n')
                            if job_temp[i][j][k] == '':
                                del job_temp[i][j][k]
                            else:
                                job_temp[i][j][k] = int(job_temp[i][j][k])
                        if len(job_temp[i][j]) == 0:
                            del job_temp[i][j]
                        
                ### count rework ###
                for i in range(0, len(job_temp)):
                    job_temp[i][-1].append(0)    
                            
                re_job_temp = copy.deepcopy(job_temp)        

                #print(job_temp)
                ##### system state settings #####
                machine_state = []
                for i in range(0, stage_num):
                    line = []
                    for j in range(0, machine_num):
                        line.append(9999)
                    machine_state.append(line)


                job_flow = []
                for i in range(0, job_num):
                    job_flow.append(0)
                #print(job_flow)
                        

                job_fin = []
                for i in range(0, job_num):
                    
                    for j in range(0, stage_num):
                        line = []
                    job_fin.append(0)
                #print(job_fin)

                stage_queue = []
                for i in range(0, stage_num):
                    for j in range(0, 1):
                        line = []
                    stage_queue.append(line)

                start_time = []
                for i in range(0, job_num):
                    line = []
                    start_time.append(line)
                #print(start_time)

                completion_time = []
                for i in range(0, job_num):
                    line = []
                    completion_time.append(line)
                #print(completion_time)

                machine_chk = []
                for i in range(0, job_num):
                    line = []
                    machine_chk.append(line)

                for i in range(0, 1):
                    for j in range(0, job_num):
                        stage_queue[i].append(j)
    

                ####### action : dispatching rule #######
                def edd(i):
                    
                    for n in range(0, len(stage_queue[i])-1):
                        for m in range(1, len(stage_queue[i])):
                            if job_temp[stage_queue[i][n]][-1][0] > job_temp[stage_queue[i][m]][-1][0]:
                                stage_queue[i][n], stage_queue[i][m] = stage_queue[i][m], stage_queue[i][n]
                def sum(k,a):
                    processing = 0
                    for c in range(a,stage_num):
                        processing += job_temp[k][3][a]
                    return processing

                def mst(i):
                    for n in range(0, len(stage_queue[i])-1):
                        for m in range(1, len(stage_queue[i])):
                            if job_temp[stage_queue[i][n]][-1][0] - job_flow[stage_queue[i][n]] - sum(stage_queue[i][n],i+1) > job_temp[stage_queue[i][m]][-1][0] - job_flow[stage_queue[i][m]] - sum(stage_queue[i][m],i+1):
                                stage_queue[i][n], stage_queue[i][m] = stage_queue[i][m], stage_queue[i][n]

                def mdd(i):
                    for n in range(0, len(stage_queue[i])-1):
                        for m in range(1, len(stage_queue[i])):
                            if max(job_temp[stage_queue[i][n]][-1][0], job_flow[stage_queue[i][n]] + sum(stage_queue[i][n], i+1)) > max(job_temp[stage_queue[i][m]][-1][0], job_flow[stage_queue[i][m]] + sum(stage_queue[i][m], i+1)):
                                stage_queue[i][n], stage_queue[i][m] = stage_queue[i][m], stage_queue[i][n]

                def mdda(i):
                    for n in range(0, len(stage_queue[i])-1):
                        for m in range(1, len(stage_queue[i])):
                            if max(job_temp[stage_queue[i][n]][-1][0] - sum(stage_queue[i][n], i+1) + job_flow[stage_queue[i][n]], job_flow[stage_queue[i][n]] + job_temp[stage_queue[i][n]][3][i]) > max(job_temp[stage_queue[i][m]][-1][0] - sum(stage_queue[i][m], i+1) + job_flow[stage_queue[i][m]], job_flow[stage_queue[i][m]] + job_temp[stage_queue[i][m]][3][i]):
                                stage_queue[i][n], stage_queue[i][m] = stage_queue[i][m], stage_queue[i][n]


                ####### Simulator start #######
                rework_station_queue = []
                rework_machine = 9999
                check = 0
        
                time = -1

                while check != job_num:
                    
                    check = 0
                    for i in range(0, job_num):
                        if job_fin[i] == 1:
                            check += 1

                    i = 0
                    time +=1 

            
                    
                    while i < stage_num:
                        j = 0               

                        while j < machine_num:
                            # ith stage, jth machine is full : processing
                            if machine_state[i][j] != 9999:
                                if job_temp[machine_state[i][j]][3][i] != 0:
                                    job_temp[machine_state[i][j]][3][i] -= 1
                                    job_flow[machine_state[i][j]] += 1

                                    

                                elif job_temp[machine_state[i][j]][3][i] == 0 and i+1 == stage_num:
                                    job_flow[machine_state[i][j]] += 1
                                    job_fin[machine_state[i][j]] = 1
                                    completion_time[machine_state[i][j]].append(time)
                                    machine_state[i][j] = 9999
                                    
    
                                elif job_temp[machine_state[i][j]][3][i] == 0 and i+1 < stage_num:
                                                                    
                                    ### event : rescheduling point ###
                                    stage_queue[i+1].append(machine_state[i][j])
                                    completion_time[machine_state[i][j]].append(time)
                                    machine_state[i][j] = 9999

                                    state = get_feature(stage_queue[i+1], machine_state[i+1])
                                    state = np.array(state)
                                    state = state.reshape(-1, 7*1)  
                                    predict = model.predict(state)   
                                    action = np.argmax(predict) 

                                    if action == 0:
                                        fit_eval(i+1,stage_queue[i+1], "(((re_job_temp[sq[o]][3][i]*job_temp[sq[o]][-1][0])/(max(job_flow[sq[o]],job_temp[sq[o]][1][0]/re_job_temp[sq[o]][1][0])))*((job_temp[sq[o]][2][0]/re_job_temp[sq[o]][2][0]-(job_temp[sq[o]][-1][0] - job_flow[sq[o]])/re_job_temp[sq[o]][3][i])+(max(job_temp[sq[o]][-1][0],(job_temp[sq[o]][-1][0] - job_flow[sq[o]])/re_job_temp[sq[o]][3][i]))))", job_temp, stage_queue, job_flow, completion_time, re_job_temp)    
                                    elif action == 1:
                                        fit_eval(i+1,stage_queue[i+1], "((re_job_temp[sq[o]][3][i]+re_job_temp[sq[o]][3][i])-((1/re_job_temp[sq[o]][3][i])-(1*job_temp[sq[o]][-1][0] - job_flow[sq[o]])))", job_temp, stage_queue, job_flow, completion_time, re_job_temp) 
                                    elif action == 2:
                                        fit_eval(i+1,stage_queue[i+1], "(max((max((1-job_temp[sq[o]][-1][0]),(max(re_job_temp[sq[o]][3][i]/job_temp[sq[o]][-1][0],job_temp[sq[o]][1][0]/re_job_temp[sq[o]][1][0])))),((max(re_job_temp[sq[o]][3][i]/job_temp[sq[o]][-1][0],re_job_temp[sq[o]][3][i]))+(job_temp[sq[o]][-1][0] - job_flow[sq[o]]-(job_temp[sq[o]][-1][0] - job_flow[sq[o]])/re_job_temp[sq[o]][3][i]))))", job_temp, stage_queue, job_flow, completion_time, re_job_temp) 
                                    elif action == 3:
                                        fit_eval(i+1,stage_queue[i+1], "(((min(re_job_temp[sq[o]][3][i],(job_temp[sq[o]][-1][0] - job_flow[sq[o]])/re_job_temp[sq[o]][3][i]))+(re_job_temp[sq[o]][3][i]/job_temp[sq[o]][-1][0]*job_temp[sq[o]][-1][0] - job_flow[sq[o]]))+(job_temp[sq[o]][-1][0]*1))", job_temp, stage_queue, job_flow, completion_time, re_job_temp) 

                                
                                    
                            # ith stage, j th machine is empty : pop 
                            elif machine_state[i][j] == 9999 and len(stage_queue[i]) != 0:               
                            
                                machine_state[i][j] = stage_queue[i].pop(0)
                                job_flow[machine_state[i][j]] += 1
                                machine_chk[machine_state[i][j]].append(j)
                                start_time[machine_state[i][j]].append(time)
                                
                                state = get_feature(stage_queue[i], machine_state[i])
                                state = np.array(state)
                                state = state.reshape(-1, 7*1) 
                                predict = model.predict(state)   
                                action = np.argmax(predict) 

                                if action == 0:
                                    fit_eval(i,stage_queue[i], "(((re_job_temp[sq[o]][3][i]*job_temp[sq[o]][-1][0])/(max(job_flow[sq[o]],job_temp[sq[o]][1][0]/re_job_temp[sq[o]][1][0])))*((job_temp[sq[o]][2][0]/re_job_temp[sq[o]][2][0]-(job_temp[sq[o]][-1][0] - job_flow[sq[o]])/re_job_temp[sq[o]][3][i])+(max(job_temp[sq[o]][-1][0],(job_temp[sq[o]][-1][0] - job_flow[sq[o]])/re_job_temp[sq[o]][3][i]))))", job_temp, stage_queue, job_flow, completion_time, re_job_temp)    
                                elif action == 1:
                                    fit_eval(i,stage_queue[i], "((re_job_temp[sq[o]][3][i]+re_job_temp[sq[o]][3][i])-((1/re_job_temp[sq[o]][3][i])-(1*job_temp[sq[o]][-1][0] - job_flow[sq[o]])))", job_temp, stage_queue, job_flow, completion_time, re_job_temp) 
                                elif action == 2:
                                    fit_eval(i,stage_queue[i], "(max((max((1-job_temp[sq[o]][-1][0]),(max(re_job_temp[sq[o]][3][i]/job_temp[sq[o]][-1][0],job_temp[sq[o]][1][0]/re_job_temp[sq[o]][1][0])))),((max(re_job_temp[sq[o]][3][i]/job_temp[sq[o]][-1][0],re_job_temp[sq[o]][3][i]))+(job_temp[sq[o]][-1][0] - job_flow[sq[o]]-(job_temp[sq[o]][-1][0] - job_flow[sq[o]])/re_job_temp[sq[o]][3][i]))))", job_temp, stage_queue, job_flow, completion_time, re_job_temp) 
                                elif action == 3:
                                    fit_eval(i,stage_queue[i], "(((min(re_job_temp[sq[o]][3][i],(job_temp[sq[o]][-1][0] - job_flow[sq[o]])/re_job_temp[sq[o]][3][i]))+(re_job_temp[sq[o]][3][i]/job_temp[sq[o]][-1][0]*job_temp[sq[o]][-1][0] - job_flow[sq[o]]))+(job_temp[sq[o]][-1][0]*1))", job_temp, stage_queue, job_flow, completion_time, re_job_temp) 
                                
                    
                            j += 1

                        for queue in stage_queue[i]:                
                            job_flow[queue] += 1
                        

                            # queue time limit over x : rework check
                            if job_temp[queue][-1][-1] == 0:    

                                # queue time limit over 1 : go to rework setup station       
                                if job_temp[queue][1][0] == 0:
                                    job_temp[queue][-1][-1] = 1
                        
                                    queue_time_over = stage_queue[i].index(queue)                                       
                                    goto_rework = stage_queue[i].pop(queue_time_over)                       
                                    rework_station_queue.append(goto_rework)
                                    
                                # queue time limit over 2 : go to rework setup station                  
                                elif job_temp[queue][2][0] == 0:
                                    job_temp[queue][-1][-1] = 2      
                                    queue_time_over = stage_queue[i].index(queue)
                                    goto_rework = stage_queue[i].pop(queue_time_over)
                                    rework_station_queue.append(goto_rework)                                   

                                # rework x -> queue time limit check
                                if i >= job_temp[queue][1][1] and i < job_temp[queue][1][2]:
                                    job_temp[queue][1][0] -= 1
                                                    
                                elif i >= job_temp[queue][2][1] and i < job_temp[queue][2][2]:
                                    job_temp[queue][2][0] -= 1
                            
                        ##### rework process #####     
                        # rework machine is empty
                        if rework_machine == 9999 and len(rework_station_queue) != 0:
                            
                            rework_machine = rework_station_queue.pop(0)

                            state = get_feature(stage_queue[i], machine_state[i])
                            state = np.array(state)
                            state = state.reshape(-1, 7*1)  
                            predict = model.predict(state)   
                            action = np.argmax(predict) 
                            action_list.append(action)
                            
                            if action == 0:
                                fit_eval(i,stage_queue[i], "(((re_job_temp[sq[o]][3][i]*job_temp[sq[o]][-1][0])/(max(job_flow[sq[o]],job_temp[sq[o]][1][0]/re_job_temp[sq[o]][1][0])))*((job_temp[sq[o]][2][0]/re_job_temp[sq[o]][2][0]-(job_temp[sq[o]][-1][0] - job_flow[sq[o]])/re_job_temp[sq[o]][3][i])+(max(job_temp[sq[o]][-1][0],(job_temp[sq[o]][-1][0] - job_flow[sq[o]])/re_job_temp[sq[o]][3][i]))))", job_temp, stage_queue, job_flow, completion_time, re_job_temp)    
                            elif action == 1:
                                fit_eval(i,stage_queue[i], "((re_job_temp[sq[o]][3][i]+re_job_temp[sq[o]][3][i])-((1/re_job_temp[sq[o]][3][i])-(1*job_temp[sq[o]][-1][0] - job_flow[sq[o]])))", job_temp, stage_queue, job_flow, completion_time, re_job_temp) 
                            elif action == 2:
                                fit_eval(i,stage_queue[i], "(max((max((1-job_temp[sq[o]][-1][0]),(max(re_job_temp[sq[o]][3][i]/job_temp[sq[o]][-1][0],job_temp[sq[o]][1][0]/re_job_temp[sq[o]][1][0])))),((max(re_job_temp[sq[o]][3][i]/job_temp[sq[o]][-1][0],re_job_temp[sq[o]][3][i]))+(job_temp[sq[o]][-1][0] - job_flow[sq[o]]-(job_temp[sq[o]][-1][0] - job_flow[sq[o]])/re_job_temp[sq[o]][3][i]))))", job_temp, stage_queue, job_flow, completion_time, re_job_temp) 
                            elif action == 3:
                                fit_eval(i,stage_queue[i], "(((min(re_job_temp[sq[o]][3][i],(job_temp[sq[o]][-1][0] - job_flow[sq[o]])/re_job_temp[sq[o]][3][i]))+(re_job_temp[sq[o]][3][i]/job_temp[sq[o]][-1][0]*job_temp[sq[o]][-1][0] - job_flow[sq[o]]))+(job_temp[sq[o]][-1][0]*1))", job_temp, stage_queue, job_flow, completion_time, re_job_temp) 
                            
                            job_flow[rework_machine] += 1
                            machine_chk[rework_machine].append("RS")
                            start_time[rework_machine].append(time)

    
                        # rework machine is full
                        elif rework_machine != 9999:
                            
                            #fin _ rework processing
                            if job_temp[rework_machine][0][0] == 0:   
                            
                                # return to the original stage 1
                                if job_temp[rework_machine][-1][-1] == 1:     
                                    
                                    for k in range(job_temp[rework_machine][1][1], job_temp[rework_machine][1][2]):
                                        job_temp[rework_machine][3][k] = re_job_temp[rework_machine][3][k]
                                    stage_queue[job_temp[rework_machine][1][1]].append(rework_machine)   
                                    job_flow[rework_machine] += 1                
                                    completion_time[rework_machine].append(time)
                                    
                                    state = get_feature(stage_queue[job_temp[rework_machine][1][1]], machine_state[job_temp[rework_machine][1][1]])
                                    state = np.array(state)
                                    state = state.reshape(-1, 7*1) 
                                    predict = model.predict(state)   
                                    action = np.argmax(predict) 
                                    action_list.append(action)

                                    if action == 0:
                                        fit_eval(job_temp[rework_machine][1][1],stage_queue[job_temp[rework_machine][1][1]], "(((re_job_temp[sq[o]][3][i]*job_temp[sq[o]][-1][0])/(max(job_flow[sq[o]],job_temp[sq[o]][1][0]/re_job_temp[sq[o]][1][0])))*((job_temp[sq[o]][2][0]/re_job_temp[sq[o]][2][0]-(job_temp[sq[o]][-1][0] - job_flow[sq[o]])/re_job_temp[sq[o]][3][i])+(max(job_temp[sq[o]][-1][0],(job_temp[sq[o]][-1][0] - job_flow[sq[o]])/re_job_temp[sq[o]][3][i]))))", job_temp, stage_queue, job_flow, completion_time, re_job_temp)    
                                    elif action == 1:
                                        fit_eval(job_temp[rework_machine][1][1],stage_queue[job_temp[rework_machine][1][1]], "((re_job_temp[sq[o]][3][i]+re_job_temp[sq[o]][3][i])-((1/re_job_temp[sq[o]][3][i])-(1*job_temp[sq[o]][-1][0] - job_flow[sq[o]])))", job_temp, stage_queue, job_flow, completion_time, re_job_temp) 
                                    elif action == 2:
                                        fit_eval(job_temp[rework_machine][1][1],stage_queue[job_temp[rework_machine][1][1]], "(max((max((1-job_temp[sq[o]][-1][0]),(max(re_job_temp[sq[o]][3][i]/job_temp[sq[o]][-1][0],job_temp[sq[o]][1][0]/re_job_temp[sq[o]][1][0])))),((max(re_job_temp[sq[o]][3][i]/job_temp[sq[o]][-1][0],re_job_temp[sq[o]][3][i]))+(job_temp[sq[o]][-1][0] - job_flow[sq[o]]-(job_temp[sq[o]][-1][0] - job_flow[sq[o]])/re_job_temp[sq[o]][3][i]))))", job_temp, stage_queue, job_flow, completion_time, re_job_temp) 
                                    elif action == 3:
                                        fit_eval(job_temp[rework_machine][1][1],stage_queue[job_temp[rework_machine][1][1]], "(((min(re_job_temp[sq[o]][3][i],(job_temp[sq[o]][-1][0] - job_flow[sq[o]])/re_job_temp[sq[o]][3][i]))+(re_job_temp[sq[o]][3][i]/job_temp[sq[o]][-1][0]*job_temp[sq[o]][-1][0] - job_flow[sq[o]]))+(job_temp[sq[o]][-1][0]*1))", job_temp, stage_queue, job_flow, completion_time, re_job_temp) 

                                    rework_machine = 9999
                                
                                # return to the original stage 2
                                elif job_temp[rework_machine][-1][-1] == 2:
                            
                                    for k in range(job_temp[rework_machine][2][1], job_temp[rework_machine][2][2]):
                                        job_temp[rework_machine][3][k] = re_job_temp[rework_machine][3][k]
                                    

                                    ### event : rescheduling point
                                    stage_queue[job_temp[rework_machine][2][1]].append(rework_machine)   
                                    job_flow[rework_machine] += 1                 
                                    completion_time[rework_machine].append(time)

                                    state = get_feature(stage_queue[job_temp[rework_machine][2][1]],machine_state[job_temp[rework_machine][2][1]]) 
                                    state = np.array(state)
                                    state = state.reshape(-1, 7*1)
                                    predict = model.predict(state)   
                                    action = np.argmax(predict) 
                                    action_list.append(action)

                                    if action == 0:
                                        fit_eval(job_temp[rework_machine][2][1],stage_queue[job_temp[rework_machine][2][1]], "(((re_job_temp[sq[o]][3][i]*job_temp[sq[o]][-1][0])/(max(job_flow[sq[o]],job_temp[sq[o]][1][0]/re_job_temp[sq[o]][1][0])))*((job_temp[sq[o]][2][0]/re_job_temp[sq[o]][2][0]-(job_temp[sq[o]][-1][0] - job_flow[sq[o]])/re_job_temp[sq[o]][3][i])+(max(job_temp[sq[o]][-1][0],(job_temp[sq[o]][-1][0] - job_flow[sq[o]])/re_job_temp[sq[o]][3][i]))))", job_temp, stage_queue, job_flow, completion_time, re_job_temp)    
                                    elif action == 1:
                                        fit_eval(job_temp[rework_machine][2][1],stage_queue[job_temp[rework_machine][2][1]], "((re_job_temp[sq[o]][3][i]+re_job_temp[sq[o]][3][i])-((1/re_job_temp[sq[o]][3][i])-(1*job_temp[sq[o]][-1][0] - job_flow[sq[o]])))", job_temp, stage_queue, job_flow, completion_time, re_job_temp) 
                                    elif action == 2:
                                        fit_eval(job_temp[rework_machine][2][1],stage_queue[job_temp[rework_machine][2][1]], "(max((max((1-job_temp[sq[o]][-1][0]),(max(re_job_temp[sq[o]][3][i]/job_temp[sq[o]][-1][0],job_temp[sq[o]][1][0]/re_job_temp[sq[o]][1][0])))),((max(re_job_temp[sq[o]][3][i]/job_temp[sq[o]][-1][0],re_job_temp[sq[o]][3][i]))+(job_temp[sq[o]][-1][0] - job_flow[sq[o]]-(job_temp[sq[o]][-1][0] - job_flow[sq[o]])/re_job_temp[sq[o]][3][i]))))", job_temp, stage_queue, job_flow, completion_time, re_job_temp) 
                                    elif action == 3:
                                        fit_eval(job_temp[rework_machine][2][1],stage_queue[job_temp[rework_machine][2][1]], "(((min(re_job_temp[sq[o]][3][i],(job_temp[sq[o]][-1][0] - job_flow[sq[o]])/re_job_temp[sq[o]][3][i]))+(re_job_temp[sq[o]][3][i]/job_temp[sq[o]][-1][0]*job_temp[sq[o]][-1][0] - job_flow[sq[o]]))+(job_temp[sq[o]][-1][0]*1))", job_temp, stage_queue, job_flow, completion_time, re_job_temp) 

                                    rework_machine = 9999

                            # rework processing for queue time limit 1
                            elif job_temp[rework_machine][-1][-1] == 1:                               
                                job_temp[rework_machine][0][0] -= 1
                                job_flow[rework_machine] += 1 
                        
                            # rework processing for queue time limit 2
                            elif job_temp[rework_machine][-1][-1] == 2:
                                job_temp[rework_machine][0][0] -= 1
                                job_flow[rework_machine] += 1 

                            for re in rework_station_queue:
                                job_flow[re] += 1
                                
                        i += 1
                                

                #### Tardiness : C - d ####    
                tardiness = []          
                total_tardiness = 0
                for i in range(0, job_num):
                    tardiness.append(max(0, job_flow[i] - job_temp[i][-1][0]))
                    total_tardiness += max(0, job_flow[i] - job_temp[i][-1][0])
                if num < 9:
                    print(total_tardiness, end = ' ')
                else:
                    print(total_tardiness)
            
                    print(action_list)
        
                
                #print("makespan : ", job_flow)
                #print("machine : ", machine_chk)
                #print("start time : ", start_time)
                #print("completion time : ", completion_time)
        print("\n")
