# Simulator
import os
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


SEED = 42
os.environ['PYTHONHASHSEED'] = str(SEED)
os.environ['TF_DETERMINISTIC_OPS'] = '1'
tf.random.set_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

job = [50,70]
stage = [5, 10]
alternative_machine = [3,5,7]
action_list = []


# rescheduling point - get system states
def simulation(time_2, action, job_num_2, stage_num_2, job_fin_2, machine_num_2, machine_state_2, machine_chk_2, start_time_2, re_job_temp_2_2,stage_queue_2, job_temp_2, job_flow_2, completion_time_2, rework_station_queue_2, rework_machine_2):
    
 
    results = []
    def edd(i):                    
        for n in range(0, len(stage_queue_2[i])-1):
            for m in range(1, len(stage_queue_2[i])):
                if job_temp_2[stage_queue_2[i][n]][-1][0] > job_temp_2[stage_queue_2[i][m]][-1][0]:
                    stage_queue_2[i][n], stage_queue_2[i][m] = stage_queue_2[i][m], stage_queue_2[i][n]
    def sum(k,a):
        processing = 0
        for c in range(a,stage_num_2):
            processing += job_temp_2[k][3][a]
        return processing

    def mst(i):
        for n in range(0, len(stage_queue_2[i])-1):
            for m in range(1, len(stage_queue_2[i])):
                if job_temp_2[stage_queue_2[i][n]][-1][0] - job_flow_2[stage_queue_2[i][n]] - sum(stage_queue_2[i][n],i+1) > job_temp_2[stage_queue_2[i][m]][-1][0] - job_flow_2[stage_queue_2[i][m]] - sum(stage_queue_2[i][m],i+1):
                    stage_queue_2[i][n], stage_queue_2[i][m] = stage_queue_2[i][m], stage_queue_2[i][n]

    def mdd(i):
        for n in range(0, len(stage_queue_2[i])-1):
            for m in range(1, len(stage_queue_2[i])):
                if max(job_temp_2[stage_queue_2[i][n]][-1][0], job_flow_2[stage_queue_2[i][n]] + sum(stage_queue_2[i][n], i+1)) > max(job_temp_2[stage_queue_2[i][m]][-1][0], job_flow_2[stage_queue_2[i][m]] + sum(stage_queue_2[i][m], i+1)):
                    stage_queue_2[i][n], stage_queue_2[i][m] = stage_queue_2[i][m], stage_queue_2[i][n]

    def mdda(i):
        for n in range(0, len(stage_queue_2[i])-1):
            for m in range(1, len(stage_queue_2[i])):
                if max(job_temp_2[stage_queue_2[i][n]][-1][0] - sum(stage_queue_2[i][n], i+1) + job_flow_2[stage_queue_2[i][n]], job_flow_2[stage_queue_2[i][n]] + job_temp_2[stage_queue_2[i][n]][3][i]) > max(job_temp_2[stage_queue_2[i][m]][-1][0] - sum(stage_queue_2[i][m], i+1) + job_flow_2[stage_queue_2[i][m]], job_flow_2[stage_queue_2[i][m]] + job_temp_2[stage_queue_2[i][m]][3][i]):
                    stage_queue_2[i][n], stage_queue_2[i][m] = stage_queue_2[i][m], stage_queue_2[i][n]


    check_3 = 0
 
    while check_3 != job_num_2:
   
        check_3 = 0
        for i in range(0, job_num_2):
            if job_fin_2[i] == 1:
                check_3 += 1
  

        i = 0
        time_2 += 1  
        while i < stage_num_2:
            j = 0        
  
            while j < machine_num_2:
                # ith stage, jth machine is full : processing
                if machine_state_2[i][j] != 9999:
                    if job_temp_2[machine_state_2[i][j]][3][i] != 0:
                        job_temp_2[machine_state_2[i][j]][3][i] -= 1
                        job_flow_2[machine_state_2[i][j]] += 1
                        if action == 0:
                            edd(i)
                        elif action == 1:
                            mst(i)
                        elif action == 2:
                            mdd(i)
                        elif action == 3:
                            mdda(i)
                        

                    elif job_temp_2[machine_state_2[i][j]][3][i] == 0 and i+1 == stage_num_2:
                        job_flow_2[machine_state_2[i][j]] += 1
                        job_fin_2[machine_state_2[i][j]] = 1
                        completion_time_2[machine_state_2[i][j]].append(time_2)
                        machine_state_2[i][j] = 9999
                        

                    elif job_temp_2[machine_state_2[i][j]][3][i] == 0 and i+1 < stage_num_2:
                                                        
                        ### event : rescheduling point ###
                        stage_queue_2[i+1].append(machine_state_2[i][j])
                        if action == 0:
                            edd(i+1)
                        elif action == 1:
                            mst(i+1)
                        elif action == 2:
                            mdd(i+1)
                        elif action == 3:
                            mdda(i+1)
                        completion_time_2[machine_state_2[i][j]].append(time_2)
                        machine_state_2[i][j] = 9999
                        
                        
                # ith stage, j th machine is empty : pop 
                elif machine_state_2[i][j] == 9999 and len(stage_queue_2[i]) != 0:               
                    
                    machine_state_2[i][j] = stage_queue_2[i].pop(0)
                    job_flow_2[machine_state_2[i][j]] += 1
                    machine_chk_2[machine_state_2[i][j]].append(j)
                    start_time_2[machine_state_2[i][j]].append(time_2)
                    if action == 0:
                        edd(i)
                    elif action == 1:
                        mst(i)
                    elif action == 2:
                        mdd(i)
                    elif action == 3:
                        mdda(i)
                    
                j += 1

            for queue in stage_queue_2[i]:                
                job_flow_2[queue] += 1
                            

                # queue time limit over x : rework check
                if job_temp_2[queue][-1][-1] == 0:    

                    # queue time limit over 1 : go to rework setup station       
                    if job_temp_2[queue][1][0] == 0:
                        job_temp_2[queue][-1][-1] = 1
            
                        queue_time_over = stage_queue_2[i].index(queue)                                       
                        goto_rework = stage_queue_2[i].pop(queue_time_over)                       
                        rework_station_queue_2.append(goto_rework)
                        
                    # queue time limit over 2 : go to rework setup station                  
                    elif job_temp_2[queue][2][0] == 0:
                        job_temp_2[queue][-1][-1] = 2      
                        queue_time_over = stage_queue_2[i].index(queue)
                        goto_rework = stage_queue_2[i].pop(queue_time_over)
                        rework_station_queue_2.append(goto_rework)                                   

                    # rework x -> queue time limit check
                    if i >= job_temp_2[queue][1][1] and i < job_temp_2[queue][1][2]:
                        job_temp_2[queue][1][0] -= 1
                                        
                    elif i >= job_temp_2[queue][2][1] and i < job_temp_2[queue][2][2]:
                        job_temp_2[queue][2][0] -= 1
                
            ##### rework process #####     
            # rework machine is empty
            if rework_machine_2 == 9999 and len(rework_station_queue_2) != 0:
            
                if action == 0:
                    edd(i)
                elif action == 1:
                    mst(i)
                elif action == 2:
                    mdd(i)
                elif action == 3:
                    mdda(i)
                rework_machine_2 = rework_station_queue_2.pop(0)
                job_flow_2[rework_machine_2] += 1
                machine_chk_2[rework_machine_2].append("RS")
                start_time_2[rework_machine_2].append(time_2)

            # rework machine is full
            elif rework_machine_2 != 9999:
                
                #fin _ rework processing
                if job_temp_2[rework_machine_2][0][0] == 0:   
                
                    # return to the original stage 1
                    if job_temp_2[rework_machine_2][-1][-1] == 1:     
                        
                        for k in range(job_temp_2[rework_machine_2][1][1], job_temp_2[rework_machine_2][1][2]):
                            job_temp_2[rework_machine_2][3][k] = re_job_temp_2_2[rework_machine_2][3][k]
                        stage_queue_2[job_temp_2[rework_machine_2][1][1]].append(rework_machine_2)   
                        job_flow_2[rework_machine_2] += 1                
                        completion_time_2[rework_machine_2].append(time_2)
                        if action == 0:
                            edd(job_temp_2[rework_machine_2][1][1])
                        elif action == 1:
                            mst(job_temp_2[rework_machine_2][1][1])
                        elif action == 2:
                            mdd(job_temp_2[rework_machine_2][1][1])
                        elif action == 3:
                            mdda(job_temp_2[rework_machine_2][1][1])
                        
                        rework_machine_2 = 9999
                        
                    # return to the original stage 2
                    elif job_temp_2[rework_machine_2][-1][-1] == 2:
                
                        for k in range(job_temp_2[rework_machine_2][2][1], job_temp_2[rework_machine_2][2][2]):
                            job_temp_2[rework_machine_2][3][k] = re_job_temp_2_2[rework_machine_2][3][k]
            
                        stage_queue_2[job_temp_2[rework_machine_2][2][1]].append(rework_machine_2)   
                        job_flow_2[rework_machine_2] += 1                 
                        completion_time_2[rework_machine_2].append(time_2)

                        if action == 0:
                            edd(job_temp_2[rework_machine_2][2][1])
                        elif action == 1:
                            mst(job_temp_2[rework_machine_2][2][1])
                        elif action == 2:
                            mdd(job_temp_2[rework_machine_2][2][1])
                        elif action == 3:
                            mdda(job_temp_2[rework_machine_2][2][1])
                        
                        rework_machine_2 = 9999
                # rework processing for queue time limit 1
                elif job_temp_2[rework_machine_2][-1][-1] == 1:                               
                    job_temp_2[rework_machine_2][0][0] -= 1
                    job_flow_2[rework_machine_2] += 1 
            
                # rework processing for queue time limit 2
                elif job_temp_2[rework_machine_2][-1][-1] == 2:
                    job_temp_2[rework_machine_2][0][0] -= 1
                    job_flow_2[rework_machine_2] += 1 

                for re in rework_station_queue_2:
                    job_flow_2[re] += 1
                    
            i += 1
    tardiness = []          
    total_tardiness = 0
    for i in range(0, job_num_2):
        tardiness.append(max(0, job_flow_2[i] - job_temp_2[i][-1][0]))
        total_tardiness += max(0, job_flow_2[i] - job_temp_2[i][-1][0])
  
    
    results.append(total_tardiness)
  
    return total_tardiness

    
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

                    results = []
                    rework_machine_2 = copy.deepcopy(rework_machine)
                    time_2 = copy.deepcopy(time)
                    job_num_2 = copy.deepcopy(job_num)
                    stage_num_2 = copy.deepcopy(stage_num)
                    job_fin_2 = copy.deepcopy(job_fin)
                    machine_num_2 = copy.deepcopy(machine_num)
                    machine_state_2 = copy.deepcopy(machine_state)
                    machine_chk_2 = copy.deepcopy(machine_chk)
                    start_time_2 = copy.deepcopy(start_time)
                    re_job_temp_2_2 = copy.deepcopy(re_job_temp)
                    stage_queue_2 = copy.deepcopy(stage_queue)
                    job_temp_2 = copy.deepcopy(job_temp)
                    job_flow_2 = copy.deepcopy(job_flow)
                    completion_time_2 = copy.deepcopy(completion_time)
                    rework_station_queue_2 = copy.deepcopy(rework_station_queue)
                    
                    results.append(simulation(time_2, 0,job_num_2, stage_num_2, job_fin_2, machine_num_2, machine_state_2, machine_chk_2, start_time_2, re_job_temp_2_2,stage_queue_2, job_temp_2, job_flow_2, completion_time_2, rework_station_queue_2,rework_machine_2))
                    rework_machine_2 = copy.deepcopy(rework_machine)
                    time_2 = copy.deepcopy(time)
                    job_num_2 = copy.deepcopy(job_num)
                    stage_num_2 = copy.deepcopy(stage_num)
                    job_fin_2 = copy.deepcopy(job_fin)
                    machine_num_2 = copy.deepcopy(machine_num)
                    machine_state_2 = copy.deepcopy(machine_state)
                    machine_chk_2 = copy.deepcopy(machine_chk)
                    start_time_2 = copy.deepcopy(start_time)
                    re_job_temp_2_2 = copy.deepcopy(re_job_temp)
                    stage_queue_2 = copy.deepcopy(stage_queue)
                    job_temp_2 = copy.deepcopy(job_temp)
                    job_flow_2 = copy.deepcopy(job_flow)
                    completion_time_2 = copy.deepcopy(completion_time)
                    rework_station_queue_2 = copy.deepcopy(rework_station_queue)
                    results.append(simulation(time_2, 1,job_num_2, stage_num_2, job_fin_2, machine_num_2, machine_state_2, machine_chk_2, start_time_2, re_job_temp_2_2,stage_queue_2, job_temp_2, job_flow_2, completion_time_2, rework_station_queue_2,rework_machine_2))
                    
                    time_2 = copy.deepcopy(time)
                    rework_machine_2 = copy.deepcopy(rework_machine)
                    job_num_2 = copy.deepcopy(job_num)
                    stage_num_2 = copy.deepcopy(stage_num)
                    job_fin_2 = copy.deepcopy(job_fin)
                    machine_num_2 = copy.deepcopy(machine_num)
                    machine_state_2 = copy.deepcopy(machine_state)
                    machine_chk_2 = copy.deepcopy(machine_chk)
                    start_time_2 = copy.deepcopy(start_time)
                    re_job_temp_2_2 = copy.deepcopy(re_job_temp)
                    stage_queue_2 = copy.deepcopy(stage_queue)
                    job_temp_2 = copy.deepcopy(job_temp)
                    job_flow_2 = copy.deepcopy(job_flow)
                    completion_time_2 = copy.deepcopy(completion_time)
                    rework_station_queue_2 = copy.deepcopy(rework_station_queue)
                    results.append(simulation(time_2, 2,job_num_2, stage_num_2, job_fin_2, machine_num_2, machine_state_2, machine_chk_2, start_time_2, re_job_temp_2_2,stage_queue_2, job_temp_2, job_flow_2, completion_time_2, rework_station_queue_2,rework_machine_2))
                    
                    rework_machine_2 = copy.deepcopy(rework_machine)
                    time_2 = copy.deepcopy(time)
                    job_num_2 = copy.deepcopy(job_num)
                    stage_num_2 = copy.deepcopy(stage_num)
                    job_fin_2 = copy.deepcopy(job_fin)
                    machine_num_2 = copy.deepcopy(machine_num)
                    machine_state_2 = copy.deepcopy(machine_state)
                    machine_chk_2 = copy.deepcopy(machine_chk)
                    start_time_2 = copy.deepcopy(start_time)
                    re_job_temp_2_2 = copy.deepcopy(re_job_temp)
                    stage_queue_2 = copy.deepcopy(stage_queue)
                    job_temp_2 = copy.deepcopy(job_temp)
                    job_flow_2 = copy.deepcopy(job_flow)
                    completion_time_2 = copy.deepcopy(completion_time)
                    rework_station_queue_2 = copy.deepcopy(rework_station_queue)
                    results.append(simulation(time_2, 3,job_num_2, stage_num_2, job_fin_2, machine_num_2, machine_state_2, machine_chk_2, start_time_2, re_job_temp_2_2,stage_queue_2, job_temp_2, job_flow_2, completion_time_2, rework_station_queue_2, rework_machine_2))
                    action = np.argmin(results)
                   
                     
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

                                    if action == 0:
                                        edd(i+1)
                                    elif action == 1:
                                        mst(i+1)
                                    elif action == 2:
                                        mdd(i+1)
                                    elif action == 3:
                                        mdda(i+1)
                                    
                            # ith stage, j th machine is empty : pop 
                            elif machine_state[i][j] == 9999 and len(stage_queue[i]) != 0:               
                               
                                machine_state[i][j] = stage_queue[i].pop(0)
                                job_flow[machine_state[i][j]] += 1
                                machine_chk[machine_state[i][j]].append(j)
                                start_time[machine_state[i][j]].append(time)

                                if action == 0:
                                    edd(i)
                                elif action == 1:
                                    mst(i)
                                elif action == 2:
                                    mdd(i)
                                elif action == 3:
                                    mdda(i)
                     
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

                            
                            if action == 0:
                                edd(i)
                            elif action == 1:
                                mst(i)
                            elif action == 2:
                                mdd(i)
                            elif action == 3:
                                mdda(i)
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
                                    

                                    if action == 0:
                                        edd(job_temp[rework_machine][1][1])
                                    elif action == 1:
                                        mst(job_temp[rework_machine][1][1])
                                    elif action == 2:
                                        mdd(job_temp[rework_machine][1][1])
                                    elif action == 3:
                                        mdda(job_temp[rework_machine][1][1])
                                    rework_machine = 9999
                                   
                                # return to the original stage 2
                                elif job_temp[rework_machine][-1][-1] == 2:
                            
                                    for k in range(job_temp[rework_machine][2][1], job_temp[rework_machine][2][2]):
                                        job_temp[rework_machine][3][k] = re_job_temp[rework_machine][3][k]
                                    

                                     ### event : rescheduling point
                                    stage_queue[job_temp[rework_machine][2][1]].append(rework_machine)   
                                    job_flow[rework_machine] += 1                 
                                    completion_time[rework_machine].append(time)
   
                                    if action == 0:
                                        edd(job_temp[rework_machine][2][1])
                                    elif action == 1:
                                        mst(job_temp[rework_machine][2][1])
                                    elif action == 2:
                                        mdd(job_temp[rework_machine][2][1])
                                    elif action == 3:
                                        mdda(job_temp[rework_machine][2][1])
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
                print("total_tardiness : ", total_tardiness)
                print(action_list)
          
           
                
                #print("makespan : ", job_flow)
                #print("machine : ", machine_chk)
                #print("start time : ", start_time)
                #print("completion time : ", completion_time)
