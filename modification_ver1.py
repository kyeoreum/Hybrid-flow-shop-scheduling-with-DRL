# Simulator
# Run - GPU 인지 확인할 것!

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



job = [30, 50]
stage = [5, 10]
alternative_machine = [3,5,7]
action_list = []

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
def get_feature():
    features = []
    # Average completion rate of operations (CRO)
    cro = 0
    for n in range(0, job_num):
        cro += len(completion_time[n])
    cro /= (stage_num * job_num)
    features.append(cro)    

    # Average completion rate of jobs (CRJ)
    crj = 0
    temp = 0
    for n in range(0, job_num):
        temp += len(completion_time[n])
        temp /= stage_num
        crj += temp
        temp = 0
    crj /= job_num
    features.append(crj)
    
    # Actual tardiness rate At
    at = 0
    left = 0
    temp = 0
    for n in range(0, job_num):
        temp += len(completion_time[n])
    if temp < stage_num * job_num:
        left = (stage_num * job_num) - temp
    else:
        left = 1
    for n in range(0, job_num):
        if job_flow[n] > job_temp[n][-1][0]:
            at += 1
    if at == 0:
        at = 0
    else:
        at /= left
    features.append(at)

    # Estimated tardiness rate ET
    et = 0
    left = 0
    T_left = 0
    T_cur = 0
    temp = 0

    # 전체 job - operation의 남은 개수
    for n in range(0, job_num):
        temp += len(completion_time[n])
    if temp < stage_num * job_num:
        left +=  (stage_num * job_num) - temp
    if left == 0:
        left = 1
   
    
    for i in range(0, job_num):   
        
        for j in range(temp+1, stage_num):
            T_left += job_temp[i][3][j]
        if job_flow[i] + T_left > job_temp[i][-1][0]:
            T_cur += 1
    if T_cur == 0:
        et = 0
    else:
        et = T_cur / left
        

    features.append(et)        

    # rework rate RR
    rr = 0
    for i in range(0, job_num):
        if job_temp[i][-1][-1] == 1:
            rr += 1
    rr /= job_num
    features.append(rr)
    
    # CR 
    cr = 0
    temp = 0
    temp2 = 0
    left = 0
    for n in range(0, job_num):
        temp = len(completion_time[n])
        for j in range(temp+1, stage_num):
            temp2 = (job_temp[i][-1][0] - job_flow[i])/re_job_temp[i][3][j]
            if temp2 < 1:   # late
                cr += 1
    cr /= job_num
    features.append(cr)

    # QTL 임박률 평균
    avg_qtl = []
    sum = 0
    for n in range(0, job_num):
        avg_qtl.append((re_job_temp[n][1][0] - job_temp[n][1][0])/re_job_temp[n][1][0])
    for k in range(0, len(avg_qtl)):
        sum += avg_qtl[k]
    qtl_avg = sum / len(avg_qtl)
    features.append(qtl_avg)         

    return features

# model (relu : hidden layer)
def create_model():

    model = Sequential()
    model.add(LSTM(128, batch_input_shape = (None,7,1), return_sequences=True, activation = 'tanh'))
    model.add(Dense(64, activation = 'tanh'))
    model.add(Dense(32, activation = 'tanh'))
    model.add(Dense(4))
    model.compile(loss = 'mse', optimizer = Adam (lr= learning_rate))
    return model


# select the action (using epsilon)
def act(cur_state):
    cur_state = np.array(cur_state)
    cur_state = cur_state.reshape(1,7,1)
    
    # epsilon보다 작으면 random action
    if np.random.rand() <= epsilon:
        return random.randrange(action_size)
    else:      
        act_values = model.predict(cur_state)
        return np.argmax(act_values[0][0])


# save in memory 
def remember(state, action, reward, next_state):
    memory.append((state, action, reward, next_state))


# replay memory (for training)
def replay(batch_size):
    minibatch = random.sample(memory, batch_size)
    for state, action, reward, next_state in minibatch:
        state = state.reshape(1,7,1)
        next_state = next_state.reshape(1,7,1)
        target = reward
        target = (reward + gamma *
                          np.amax(model.predict(next_state)[0]))
        
        
        target_f = model.predict(state)    
        target_f[0][0][action] = target
        
     
        model.fit(state, target_f, epochs=1, verbose=0)
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
           
                f = open("Data/TData_"+str(jn)+"_"+str(sn)+"_"+str(al)+"_"+str(num)+".txt", 'r')                            
                


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

                batch_size = 50

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
                                    state = get_feature()
                                    state = np.array(state)
                                    state = state.reshape(7,1)         
                                                                    
                                    action = act(state)
                                    action = np.array(action)

                                    print(job_temp)
                                    print(get_feature())

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
                                    new_state = get_feature()
                                    new_state = np.array(new_state)
                                    new_state = new_state.reshape(7,1)
                                    
                                # reward 
                                    # 실제 tardiness 감소
                                    if state[2] > new_state[2]:
                                        reward += 10
                                        if state[5] > new_state[5]:
                                            if state[3] > new_state[3]:
                                                reward += 20
                                           
                                        elif state[5] == new_state[5]:
                                            reward += 10
                                                                                             
                                    elif state[2] == new_state[2]:
                                        if state[3] > new_state[3]:
                                            
                                            reward += 5
                                        elif state[3] == new_state[3]:
                                            reward += 1
                                        elif state[3] < new_state[3]:
                                            reward -= 1
                                    elif state[2] < new_state[2]:
                                        reward -= 8
                                
                                    
                                    
                                    #save the system states
                                    remember(state, action, reward, new_state)
                                
                                    
                                    if len(memory) > batch_size:
                                        replay(batch_size)


                            # ith stage, j th machine is empty : pop 
                            elif machine_state[i][j] == 9999 and len(stage_queue[i]) != 0:               
                                machine_state[i][j] = stage_queue[i].pop(0)
                                job_flow[machine_state[i][j]] += 1
                                machine_chk[machine_state[i][j]].append(j)
                                start_time[machine_state[i][j]].append(time)    
                                
                                ### event : rescheduling point
                                state = get_feature()
                                state = np.array(state)
                                state = state.reshape(7,1)         
                                                                
                                action = act(state)
                                action = np.array(action)

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
                                new_state = get_feature()
                                new_state = np.array(new_state)
                                new_state = new_state.reshape(7,1)
                                # reward 
                                # 실제 tardiness 감소
                                if state[2] > new_state[2]:
                                    reward += 10
                                    if state[5] > new_state[5]:
                                        if state[3] > new_state[3]:
                                            reward += 20
                                        
                                    elif state[5] == new_state[5]:
                                        reward += 10
                                                                                            
                                elif state[2] == new_state[2]:
                                    if state[3] > new_state[3]:
                                        
                                        reward += 5
                                    elif state[3] == new_state[3]:
                                        reward += 1
                                    elif state[3] < new_state[3]:
                                        reward -= 1
                                elif state[2] < new_state[2]:
                                    reward -= 8
                                
                                #save the system states
                                remember(state, action, reward, new_state)
                             
                                
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
                            state = get_feature()
                            state = np.array(state)
                            state = state.reshape(7,1)
                                                    
                            action = act(state)
                            action = np.array(action)
                        

                            if action == 0:
                                edd(i)   
                            elif action == 1:
                                mst(i)
                            elif action == 2:
                                mdd(i)
                            elif action == 3:
                                mdda(i)
                            action_list.append(action)

                            new_state = get_feature()
                            new_state = np.array(new_state)
                            new_state = new_state.reshape(7,1)
    
                                                            # reward 
                            # 실제 tardiness 감소
                            if state[2] > new_state[2]:
                                reward += 10
                                if state[5] > new_state[5]:
                                    if state[3] > new_state[3]:
                                        reward += 20
                                    
                                elif state[5] == new_state[5]:
                                    reward += 10
                                                                                        
                            elif state[2] == new_state[2]:
                                if state[3] > new_state[3]:
                                    
                                    reward += 5
                                elif state[3] == new_state[3]:
                                    reward += 1
                                elif state[3] < new_state[3]:
                                    reward -= 1
                            elif state[2] < new_state[2]:
                                reward -= 8
                            remember(state, action, reward, new_state)
                        
                            
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
                                    state = get_feature()
                                    state = np.array(state)
                                    state = state.reshape(7,1)
                                                                    
                                    action = act(state)
                                    action = np.array(action)
                                

                                    if action == 0:
                                        edd(i)   
                                    elif action == 1:
                                        mst(i)
                                    elif action == 2:
                                        mdd(i)
                                    elif action == 3:
                                        mdda(i)
                                    action_list.append(action)

                                    new_state = get_feature()
                                    new_state = np.array(new_state)
                                    new_state = new_state.reshape(7,1)

                                # reward 
                                    # 실제 tardiness 감소
                                    if state[2] > new_state[2]:
                                        reward += 10
                                        if state[5] > new_state[5]:
                                            if state[3] > new_state[3]:
                                                reward += 20
                                           
                                        elif state[5] == new_state[5]:
                                            reward += 10
                                                                                             
                                    elif state[2] == new_state[2]:
                                        if state[3] > new_state[3]:
                                            
                                            reward += 5
                                        elif state[3] == new_state[3]:
                                            reward += 1
                                        elif state[3] < new_state[3]:
                                            reward -= 1
                                    elif state[2] < new_state[2]:
                                        reward -= 8
                                    remember(state, action, reward, new_state)
                                
                                    
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
                                    state = get_feature()
                                    state = np.array(state)
                                    state = state.reshape(7,1)

                                    action = act(state)
                                    action = np.array(action)
                                

                                    if action == 0:
                                        edd(i)   
                                    elif action == 1:
                                        mst(i)
                                    elif action == 2:
                                        mdd(i)
                                    elif action == 3:
                                        mdda(i)
                                    action_list.append(action)

                                    new_state = get_feature()
                                    new_state = np.array(new_state)
                                    new_state = new_state.reshape(7,1)

                                # reward 
                                    # 실제 tardiness 감소
                                    if state[2] > new_state[2]:
                                        reward += 10
                                        if state[5] > new_state[5]:
                                            if state[3] > new_state[3]:
                                                reward += 20
                                           
                                        elif state[5] == new_state[5]:
                                            reward += 10
                                                                                             
                                    elif state[2] == new_state[2]:
                                        if state[3] > new_state[3]:
                                            
                                            reward += 5
                                        elif state[3] == new_state[3]:
                                            reward += 1
                                        elif state[3] < new_state[3]:
                                            reward -= 1
                                    elif state[2] < new_state[2]:
                                        reward -= 8
                                    remember(state, action, reward, new_state)
                                
                                    
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

model.save("practice.h5")