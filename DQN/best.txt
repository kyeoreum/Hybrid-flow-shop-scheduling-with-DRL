import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import random
import time
from collections import deque
import copy
import tensorflow as tf
import tensorflow.keras as keras
import numpy as np
from tensorflow.keras.layers import Input, Dense, LSTM, Flatten
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.optimizers import Adam
import plaidml.keras
plaidml.keras.install_backend()

start = time.time()
SEED = 42
os.environ['PYTHONHASHSEED'] = str(SEED)
os.environ['TF_DETERMINISTIC_OPS'] = '1'
tf.random.set_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

# 다시 !!!! state feature ,reward
# 만약 reward 기준을 edd로 하면..?

job = [30, 50]
stage = [5, 10]
alternative_machine = [3, 5]

action_list = []
stage_queue = []

# Parameter settings
gamma = 0.85
epsilon = 1.0
epsilon_min = 0.01
epsilon_decay = 0.999
learning_rate = 0.001
tau = 1e-3

# number of rules (DRs)
action_size = 4

memory = deque(maxlen=1500)

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


# model (relu : hidden layer)
def create_model():
    model = Sequential()

    model.add(Dense(64, input_dim=7, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(4, activation = 'linear'))
    model.compile(loss="mean_squared_error", optimizer=Adam(lr=learning_rate)) #optimizer Adam
  
    return model

# select the action (using epsilon)
def act(cur_state):
    cur_state = np.array(cur_state)
    cur_state = cur_state.reshape(-1, 7*1)

    # epsilon보다 작으면 random action
    if np.random.rand() <= epsilon:
        return random.randrange(action_size)
    
    else:        
        act_values = model.predict(cur_state)
        print(act_values, np.argmax(act_values[0]))

        return np.argmax(act_values[0]) 

# save in memory 
def remember(state, action, reward, next_state, done):
    memory.append((state, action, reward, next_state, done))
 

def replay(batch_size):

    global memory
    global epsilon

    minibatch = random.sample(memory, batch_size)
    cnt = 0

    for state, action, reward, next_state, done in minibatch:
       
        state = np.array(state)
        state = state.reshape(-1, 7*1)

        next_state = np.array(next_state)
        next_state = next_state.reshape(-1, 7*1)

        if not done:
            k = np.argmax(model.predict(next_state)[0])
            target = (reward + gamma*target_model.predict(next_state)[0][k])

        else:
            target = reward
        target_f = model.predict(state)
        target_f[0][action] = target

        model.fit(state, target_f, epochs = 1, verbose = 0)

    if epsilon > epsilon_min:
        epsilon *= epsilon_decay

def target_train():
    weights = model.get_weights()
    target_weights = target_model.get_weights()
    for i in range(len(target_weights)):
        target_weights[i] = weights[i]* tau + target_weights[i]*(1-tau)
    target_model.set_weights(target_weights)


# create model
model = create_model()
target_model = model
reward = 0
Qtl1 = 0
Qtl2 = 0
e_num = 0
best_num = 0
batch_size = 60
# episode (all instainces)
for jn in job:
    for sn in stage:
        for al in alternative_machine:
            for num in range(0, 5):
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
                    for n in range(0, len(stage_queue[i])):
                        for m in range(0, len(stage_queue[i])):
                            if job_temp[stage_queue[i][n]][-1][0] >= job_temp[stage_queue[i][m]][-1][0]:
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
                
                
                training_period = 0
                while check != job_num:
                    training_period += 1
                    if training_period % 5 == 0 and len(memory) > batch_size:  
                        print(len(memory))                         
                        replay(batch_size)
                
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
                                    state = get_feature(stage_queue[i+1], machine_state[i+1])
                                    action = act(state)

                                   
                                    if action == 0:
                                        edd(i+1)   
                                    elif action == 1:
                                        mst(i+1)
                                    elif action == 2:
                                        mdd(i+1)
                                    elif action == 3:
                                        mdda(i+1)

                                    # action_list : Check the selected rules
                                    action_list.append(action)

                                    # after apply the action : check the new system state
                                  
                                    new_state = get_feature(stage_queue[i+1], machine_state[i+1])
                                    completion_time[machine_state[i][j]].append(time)
                                    machine_state[i][j] = 9999
                                    e_num += 1
                                   

                                # reward 
                                    reward = 0
                                    # 실제 tardiness 감소
                                    if state[0] > new_state[0]: 
                                        reward += 2
                                    if state[1] > new_state[1]:
                                        reward += 1
                                    if state[2] > new_state[2]:
                                        reward += 1
                                    if state[2] < new_state[2]:
                                        reward -= 1
                                    
                                    if state[0] < new_state[0]:
                                        reward -= 2
                                    if state[1] < new_state[1]:
                                        reward -= 1
                                    

                                    count_len = 0
                                    for c in range(0, job_num):
                                        count_len +=  len(completion_time[i]) 
                                    if count_len == job_num * stage_num:
                                        done = True
                                         
                                    
                                    #save the system states
                                    remember(state, action, reward, new_state, done)
                    

                            # ith stage, j th machine is empty : pop 
                            elif machine_state[i][j] == 9999 and len(stage_queue[i]) != 0:                                              
                                ### event : rescheduling point
                                state = get_feature(stage_queue[i], machine_state[i])   
                                 
                                action = act(state)
                              
                                if action == 0:
                                    edd(i)   
                                elif action == 1:
                                    mst(i)
                                elif action == 2:
                                    mdd(i)
                                elif action == 3:
                                    mdda(i)
                                new_state = get_feature(stage_queue[i], machine_state[i])

                                machine_state[i][j] = stage_queue[i].pop(0)
                                job_flow[machine_state[i][j]] += 1
                                machine_chk[machine_state[i][j]].append(j)
                                start_time[machine_state[i][j]].append(time) 

                                # action_list : Check the selected rules
                                action_list.append(action)

                                # after apply the action : check the new system state
                                
                                e_num += 1
                                
                                # reward 
                                reward = 0
                                # 실제 tardiness 감소
                                if state[0] > new_state[0]: 
                                    reward += 2
                                if state[1] > new_state[1]:
                                    reward += 1
                                if state[2] < new_state[2]:
                                    reward += 1
                                if state[2] > new_state[2]:
                                    reward -= 1
                                
                                if state[0] < new_state[0]:
                                    reward -= 2
                                if state[1] < new_state[1]:
                                    reward -= 1
                                
                              
                                count_len = 0
                                for c in range(0, job_num):
                                    count_len +=  len(completion_time[i]) 
                                if count_len == job_num * stage_num:
                                    done = True
                                     

                                #save the system states
                                remember(state, action, reward, new_state, done)
                             
                                
                       
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
                        
                             ### event : rescheduling point
                            state = get_feature(stage_queue[i], machine_state[i])
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

                            new_state = get_feature(stage_queue[i], machine_state[i])
                            e_num += 1
                            
                            rework_machine = rework_station_queue.pop(0)
                            job_flow[rework_machine] += 1
                            machine_chk[rework_machine].append("RS")
                            start_time[rework_machine].append(time)

                             # reward 
                            # 실제 tardiness 감소
                            reward = 0
                            if state[0] > new_state[0]: 
                                reward += 2
                            if state[1] > new_state[1]:
                                reward += 1
                            if state[2] < new_state[2]:
                                reward += 1
                            if state[2] > new_state[2]:
                                reward -= 1
                            
                            if state[0] < new_state[0]:
                                reward -= 2
                            if state[1] < new_state[1]:
                                reward -= 1
                           
                    
                            count_len = 0
                            for c in range(0, job_num):
                                count_len +=  len(completion_time[i]) 
                            if count_len == job_num * stage_num:
                                done = True
                                                            
                            remember(state, action, reward, new_state, done)
                            
                    
                        # rework machine is full
                        elif rework_machine != 9999:
                            

                            
                            #fin _ rework processing
                            if job_temp[rework_machine][0][0] == 0:   
                            
                                # return to the original stage 1
                                if job_temp[rework_machine][-1][-1] == 1:     
                                    
                                    for k in range(job_temp[rework_machine][1][1], job_temp[rework_machine][1][2]):
                                        job_temp[rework_machine][3][k] = re_job_temp[rework_machine][3][k]
                                  
                                    
                                    state = get_feature(stage_queue[job_temp[rework_machine][1][1]], machine_state[job_temp[rework_machine][1][1]])
                                    action = act(state)
                   
                                    if action == 0:
                                        edd(job_temp[rework_machine][1][1])   
                                    elif action == 1:
                                        mst(job_temp[rework_machine][1][1])
                                    elif action == 2:
                                        mdd(job_temp[rework_machine][1][1])
                                    elif action == 3:
                                        mdda(job_temp[rework_machine][1][1])
                                    action_list.append(action)

                                    new_state = get_feature(stage_queue[job_temp[rework_machine][1][1]], machine_state[job_temp[rework_machine][1][1]])
                                    e_num += 1
                                    stage_queue[job_temp[rework_machine][1][1]].append(rework_machine)   
                                    job_flow[rework_machine] += 1                
                                    completion_time[rework_machine].append(time)

                                    rework_machine = 9999

                                     ### event : rescheduling point

                                # reward 
                                    # 실제 tardiness 감소
                                    reward = 0
                                    if state[0] > new_state[0]: 
                                        reward += 2
                                    if state[1] > new_state[1]:
                                        reward += 1
                                    if state[2] < new_state[2]:
                                        reward += 1
                                    if state[2] > new_state[2]:
                                        reward -= 1
                                    
                                    if state[0] < new_state[0]:
                                        reward -= 2
                                    if state[1] < new_state[1]:
                                        reward -= 1
                                    
                                    
                                    count_len = 0
                                    for c in range(0, job_num):
                                        count_len +=  len(completion_time[i]) 
                                    if count_len == job_num * stage_num:
                                        done = True
                                         
                                    remember(state, action, reward, new_state, done)

                                # return to the original stage 2
                                elif job_temp[rework_machine][-1][-1] == 2:
                            
                                    for k in range(job_temp[rework_machine][2][1], job_temp[rework_machine][2][2]):
                                        job_temp[rework_machine][3][k] = re_job_temp[rework_machine][3][k]
                                    
                                     ### event : rescheduling point
                                    state = get_feature(stage_queue[job_temp[rework_machine][2][1]],machine_state[job_temp[rework_machine][2][1]]) 
                                                          
                                    action = act(state)

                                    if action == 0:
                                        edd(job_temp[rework_machine][2][1])   
                                    elif action == 1:
                                        mst(job_temp[rework_machine][2][1])
                                    elif action == 2:
                                        mdd(job_temp[rework_machine][2][1])
                                    elif action == 3:
                                        mdda(job_temp[rework_machine][2][1])
                                    action_list.append(action)

                                    new_state = get_feature(stage_queue[job_temp[rework_machine][2][1]], machine_state[job_temp[rework_machine][2][1]]) 
                                    e_num += 1
                                    stage_queue[job_temp[rework_machine][2][1]].append(rework_machine)   
                                    job_flow[rework_machine] += 1                 
                                    completion_time[rework_machine].append(time)
                                    

                                    rework_machine = 9999
                 
                                    # 실제 tardiness 감소
                                    reward = 0
                                    if state[0] > new_state[0]: 
                                        reward += 2
                                    if state[1] > new_state[1]:
                                        reward += 1
                                    if state[2] > new_state[2]:
                                        reward += 1
                                    if state[2] < new_state[2]:
                                        reward -= 1
                                    
                                    if state[0] < new_state[0]:
                                        reward -= 2
                                    if state[1] < new_state[1]:
                                        reward -= 1
                                   

                                    count_len = 0
                                    for i in range(0, job_num):
                                        count_len +=  len(completion_time[i]) 
                                    if count_len == job_num * stage_num:
                                        done = True
                                         
                                    
                                    remember(state, action, reward, new_state, done)
                                    

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
                    print(total_tardiness)
             
              
                
                
                #print("makespan : ", job_flow)
                #print("machine : ", machine_chk)
                #print("start time : ", start_time)
                #print("completion time : ", completion_time)

model.save("practice8.h5")
end = time.time()
print(e_num)
print(best_num)