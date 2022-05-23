# Simulator
# Run - GPU 인지 확인할 것!

from audioop import avg
from logging.handlers import QueueListener
import os
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
import plaidml.keras
plaidml.keras.install_backend()

# 다시 !!!! state feature ,reward
# 만약 reward 기준을 edd로 하면..?

job = [30,50]
stage = [5, 10]
alternative_machine = [3,5,7]
action_list = []
stage_queue = []

# Parameter settings
gamma = 0.95
epsilon = 0.9
epsilon_min = 0.01
epsilon_decay = 0.995
learning_rate = 0.005

# number of rules (DRs)
action_size = 4

memory = deque(maxlen=2000)


# rescheduling point - get system states
def get_feature(stage_queue):

    features = []
    
    # Average completion rate of operations
    cro = 0
    for n in range(0, job_num):
        cro += len(completion_time[n])
    cro /= (stage_num * job_num)
    features.append(cro)

    # Average completion rate of jobs
    crj = 0
    temp = 0
    for n in range(0, job_num):
        temp += len(completion_time[n])
        temp /= stage_num
        crj += temp
        temp = 0
    crj /= job_num
    features.append(crj)


    # 3. 전체 대비 해당 Queue tardiness (Actual)
    Td1 = 0
    Td2 = 0
    temp = 0

    for n in range(0, job_num):
        if job_flow[n] > job_temp[n][-1][0]:
            Td1 += abs(job_flow[n] - job_temp[n][-1][0])
    
    for k in stage_queue:
        if job_flow[k] > job_temp[k][-1][0]:
            Td2 += abs(job_flow[k] - job_temp[k][-1][0])
    
    if Td1 == 0:
        Td1 = 0
    else:
        Td1 = Td2 / Td1
    features.append(Td1)

    # 4. 전체 대비 Queue tardiness (Estimated)
    At1 = 0
    left1 = 0
    At2 = 0
    left2 = 0
    temp = 0

    for n in range(0, job_num):
        temp = (stage_num * job_num) - len(completion_time[n])
        for x in range(0, job_num):
            At1 = job_flow[x] - job_temp[x][-1][0]
            for y in range(temp, stage_num):
                left1 += job_temp[x][3][y]
            At1 += left1
    
    for i in stage_queue:
        temp = (stage_num * job_num) - len(completion_time[i])
        for j in stage_queue:
            At2 = job_flow[j] - job_temp[j][-1][0]
            for k in range(temp, stage_num):
                left2 += job_temp[j][3][k]
            At2 += left2

    if At1 == 0:
        At1 = 0
    else:
        At1 = At2 / At1
    features.append(At1)


    # 5.전체 대비 Critical ratio 개수
    Cr1 = 0
    Cr2 = 0
    temp =0

    for n in range(0, job_num):
        temp = len(completion_time[n])
        for k in range(0, job_num):
            if temp+1 < stage_num:
                if (job_temp[k][-1][0] - job_flow[k]) / re_job_temp[k][3][temp+1] < 1:
                    Cr1 += 1
            elif temp+1 == stage_num:
                if (job_temp[k][-1][0] - job_flow[k]) / re_job_temp[k][3][temp] < 1:
                    Cr1 += 1

    for x in stage_queue:
        temp = len(completion_time[x])
        for y in range(0, job_num):
            if temp+1 < stage_num:
                if (job_temp[y][-1][0] - job_flow[y]) / re_job_temp[y][3][temp+1] < 1:
                    Cr2 += 1
            elif temp+1 == stage_num:
                if (job_temp[y][-1][0] - job_flow[y]) / re_job_temp[y][3][temp] < 1:
                    Cr2 += 1

    if Cr1 == 0:
        Cr1 = 0
    else:
        Cr1 = Cr2 / Cr1
    features.append(Cr1)

  

    # 6. 전체 대비 Queue 여유 개수
    rm1 = 0
    rm2 = 0
    temp = 0
    for n in range(0, job_num):
        if (job_temp[n][1][0] / re_job_temp[n][1][0]) < 0.5:
            rm1 += 1
    
    for i in stage_queue:
        if(job_temp[i][1][0] / re_job_temp[i][1][0]) < 0.5:
            rm2 += 1
    
    if rm1 == 0:
        rm1 = 0
    else:
        rm1 = rm2 / rm1    
    features.append(rm1)

    return features

# model (relu : hidden layer)
def create_model():
    model = Sequential()
    model.add(LSTM(128, return_sequences=True, activation = 'softmax', input_shape=(6,1)))
    model.add(Dense(64, activation = 'softmax'))
    model.add(Dense(32, activation = 'softmax'))
    model.add(Dense(4))
    model.compile(loss = 'mse', optimizer = Adam (lr= learning_rate))
    return model


# select the action (using epsilon)
def act(cur_state):
    cur_state = np.array(cur_state)
    cur_state = np.reshape(cur_state, [1,6,1])
    
    # epsilon보다 작으면 random action
    if np.random.rand() <= epsilon:
        return random.randrange(action_size)
    act_values = model.predict(cur_state)
    #print(act_values)
    return np.argmax(act_values[0][0]) 

# save in memory 
def remember(state, action, reward, next_state, done):
    memory.append((state, action, reward, next_state, done))


# replay memory (for training)
def replay(batch_size):
    minibatch = random.sample(memory, batch_size)

    for state, action, reward, next_state, done in minibatch:
        state = np.array(state)
        next_state = np.array(next_state)

        state = np.reshape(state, [1,6,1])
        next_state = np.reshape(next_state, [1,6,1])

        target = reward
        if not done:
            target = (reward + gamma * np.amax(model.predict(next_state)[0]))
        target_f = model.predict(state)
        #print(action)  
        target_f[0][0][action] = target
   
        model.fit(state, target_f, epochs=1, verbose = 0)

    global epsilon
    if epsilon > epsilon_min:
        epsilon *= epsilon_decay

    

# create model
model = create_model()
target_model = create_model()
reward = 0
Qtl1 = 0
Qtl2 = 0

# episode (all instainces)
for jn in job:
    for sn in stage:
        for al in alternative_machine:
            for num in range(0, 10):
                memory = []
                f = open("Data/TData_"+str(jn)+"_"+str(sn)+"_"+str(al)+"_"+str(num)+".txt", 'r')                            
                
                done = False
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

                batch_size = 100

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
                    for j in range(0, 10):
                        stage_queue[i].append(j)
                cut = 10

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
                rand_pois = np.random.exponential(10, job_num-10)
                rand_pois = sorted(rand_pois)
                rand_pois = np.round(rand_pois, 0) 
                #print(rand_pois)
                time = -1

                while check != job_num:
                    check = 0
                    for i in range(0, job_num):
                        if job_fin[i] == 1:
                            check += 1
                    
                    
                    time += 1
                    for t in rand_pois:
                        if time == t:
                            stage_queue[0].append(cut)
                            cut += 1
                    
                    i = 0
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
                                    stage_queue[i+1].append(machine_state[i][j])
                                    completion_time[machine_state[i][j]].append(time)
                                    machine_state[i][j] = 9999
                                    
                                    ### event : rescheduling point ###
                                    if i+1 < stage_num:
                                        state = get_feature(stage_queue[i+1])                                     
                                    else:
                                        state = get_feature(stage_queue[i])
                                    
                                    action = act(state)
               
                               
                                    if action == 0:
                                        edd(i)   
                                    elif action == 1:
                                        mst(i)
                                    elif action == 2:
                                        mdd(i)
                                    elif action == 3:
                                        mdda(i)

                                    # action_list : Check the selected rules
                                    action_list.append(action)

                                    # after apply the action : check the new system state
                                    new_state = get_feature(stage_queue[i+1])
                     
                         
                                    
                                # reward 
                                    # 실제 tardiness 감소
                                    if state[2] > new_state[2]:
                                        reward += 10
                                    elif state[3] > new_state[3]:
                                        reward += 5
                                    else:
                                        reward -= 100

                                    count_len = 0
                                    for c in range(0, job_num):
                                        count_len +=  len(completion_time[i]) 
                                    if count_len == job_num * stage_num:
                                        done = True
                                    
                                    #save the system states
                                    remember(state, action, reward, new_state, done)
                                
                                    
                                    if len(memory) > batch_size:
                                        replay(batch_size)


                            # ith stage, j th machine is empty : pop 
                            elif machine_state[i][j] == 9999 and len(stage_queue[i]) != 0:               
                                machine_state[i][j] = stage_queue[i].pop(0)
                                job_flow[machine_state[i][j]] += 1
                                machine_chk[machine_state[i][j]].append(j)
                                start_time[machine_state[i][j]].append(time)    
                                
                                ### event : rescheduling point
                                state = get_feature(stage_queue[i])
                    
                                      
                                action = act(state)
                              
                    
                                if action == 0:
                                    edd(i)   
                                elif action == 1:
                                    mst(i)
                                elif action == 2:
                                    mdd(i)
                                elif action == 3:
                                    mdda(i)

                                # action_list : Check the selected rules
                                action_list.append(action)

                                # after apply the action : check the new system state
                                new_state = get_feature(stage_queue[i])
                     
            
                        
                                # reward 
                                # 실제 tardiness 감소
                                if state[2] > new_state[2]:
                                        reward += 10
                                elif state[3] > new_state[3]:
                                    reward += 5
                                else:
                                    reward -= 100
                            
                                count_len = 0
                                for c in range(0, job_num):
                                    count_len +=  len(completion_time[i]) 
                                if count_len == job_num * stage_num:
                                    done = True

                                #save the system states
                                remember(state, action, reward, new_state, done)
                             
                                
                                if len(memory) > batch_size:
                                    replay(batch_size)
                                
                                #model.summary()
                                
                            j += 1

                        for queue in stage_queue[i]:                
                            job_flow[queue] += 1
                        

                            # queue time limit over x : rework check
                            if job_temp[queue][-1][-1] == 0:    
                                # rework x -> queue time limit check
                                if i >= job_temp[queue][1][1] and i < job_temp[queue][1][2]:
                                    job_temp[queue][1][0] -= 1
                                                    
                                elif i >= job_temp[queue][2][1] and i < job_temp[queue][2][2]:
                                    job_temp[queue][2][0] -= 1

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

                                
                            
                        ##### rework process #####     
                        # rework machine is empty
                        if rework_machine == 9999 and len(rework_station_queue) != 0:
                        
                            rework_machine = rework_station_queue.pop(0)
                            job_flow[rework_machine] += 1
                            machine_chk[rework_machine].append("RS")
                            start_time[rework_machine].append(time)

                             ### event : rescheduling point
                            state = get_feature(rework_station_queue)
                            action = act(state)
                       
                            if action == 0:
                                edd(i)   
                            elif action == 1:
                                mst(i)
                            elif action == 2:
                                mdd(i)
                            elif action == 3:
                                mdda(i)
                            action_list.append(action)

                            new_state = get_feature(rework_station_queue)
                

                             # reward 
                            # 실제 tardiness 감소
                            if state[2] > new_state[2]:
                                reward += 10
                            elif state[3] > new_state[3]:
                                reward += 5
                            else:
                                reward -= 100
                    
                            count_len = 0
                            for c in range(0, job_num):
                                count_len +=  len(completion_time[i]) 
                            if count_len == job_num * stage_num:
                                done = True
                            
                            remember(state, action, reward, new_state, done)
                            
                            if len(memory) > batch_size:
                                replay(batch_size)
                            

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
                                    rework_machine = 9999

                                     ### event : rescheduling point
                                    state = get_feature(stage_queue[i])
                                    action = act(state)
                   
                                    if action == 0:
                                        edd(i)   
                                    elif action == 1:
                                        mst(i)
                                    elif action == 2:
                                        mdd(i)
                                    elif action == 3:
                                        mdda(i)
                                    action_list.append(action)

                                    new_state = get_feature(stage_queue[i])

                                # reward 
                                    # 실제 tardiness 감소
                                    if state[2] > new_state[2]:
                                        reward += 10
                                    elif state[3] > new_state[3]:
                                        reward += 5
                                    else:
                                        reward -= 100
                                    
                                    count_len = 0
                                    for c in range(0, job_num):
                                        count_len +=  len(completion_time[i]) 
                                    if count_len == job_num * stage_num:
                                        done = True
                             
                                    remember(state, action, reward, new_state, done)
                                    
                                    if len(memory) > batch_size:
                                        replay(batch_size)


                                # return to the original stage 2
                                elif job_temp[rework_machine][-1][-1] == 2:
                            
                                    for k in range(job_temp[rework_machine][2][1], job_temp[rework_machine][2][2]):
                                        job_temp[rework_machine][3][k] = re_job_temp[rework_machine][3][k]
                                    stage_queue[job_temp[rework_machine][2][1]].append(rework_machine)   
                                    job_flow[rework_machine] += 1                 
                                    completion_time[rework_machine].append(time)
                                    rework_machine = 9999

                                     ### event : rescheduling point
                                    state = get_feature(stage_queue[i])                         
                                    action = act(state)

                                    if action == 0:
                                        edd(i)   
                                    elif action == 1:
                                        mst(i)
                                    elif action == 2:
                                        mdd(i)
                                    elif action == 3:
                                        mdda(i)
                                    action_list.append(action)

                                    new_state = get_feature(stage_queue[i])
                         
                                # reward 
                                    # 실제 tardiness 감소
                                    if state[2] > new_state[2]:
                                        reward += 10
                                    elif state[3] > new_state[3]:
                                        reward += 5
                                    else:
                                        reward -= 100

                                    count_len = 0
                                    for i in range(0, job_num):
                                        count_len +=  len(completion_time[i]) 
                                    if count_len == job_num * stage_num:
                                        done = True
                                    
                                    remember(state, action, reward, new_state, done)
                                    
                                    if len(memory) > batch_size:
                                        replay(batch_size)

            
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
                
                #print("makespan : ", job_flow)
                #print("machine : ", machine_chk)
                #print("start time : ", start_time)
                #print("completion time : ", completion_time)

model.save("practice5.h5")
