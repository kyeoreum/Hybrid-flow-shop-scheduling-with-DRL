# Simulator
import copy
import time

job = [30, 50, 70]
stage = [5, 10]
alternative_machine = [3,5,7]

f = open("C:/Users/user/source/repos/my_thesis/Data/TData_30_5_3_0.txt", 'r')                            

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

stage_queue = []
for i in range(0, stage_num):
    for j in range(0, 1):
        line = []
    stage_queue.append(line)

for i in range(0, 1):
    for j in range(0, job_num):
        stage_queue[i].append(j)


####### Simulator start #######
rework_station_queue = []
rework_machine = 9999
check = 0
while check != job_num:
    check = 0
    for i in range(0, job_num):
        if job_fin[i] == 1:
            check += 1
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
                    machine_state[i][j] = 9999
                   
                
                elif job_temp[machine_state[i][j]][3][i] == 0 and i+1 < stage_num:
                    stage_queue[i+1].append(machine_state[i][j])
                    machine_state[i][j] = 9999


            # ith stage, j th machine is empty : pop 
            elif machine_state[i][j] == 9999 and len(stage_queue[i]) != 0:               
                machine_state[i][j] = stage_queue[i].pop(0)
                job_flow[machine_state[i][j]] += 1

            j += 1

        for queue in stage_queue[i]:                
            job_flow[queue] += 1
           
            # queue time limit over 
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
            job_flow[rework_machine] += 1
            

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
                    rework_machine = 9999
                   
              
                # return to the original stage 2
                elif job_temp[rework_machine][-1][-1] == 2:
              
                    for k in range(job_temp[rework_machine][2][1], job_temp[rework_machine][2][2]):
                        job_temp[rework_machine][3][k] = re_job_temp[rework_machine][3][k]
                    stage_queue[job_temp[rework_machine][2][1]].append(rework_machine)   
                    job_flow[rework_machine] += 1                 
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
for i in range(0, job_num):
    tardiness.append(max(0, job_flow[i] - job_temp[i][-1][0]))
print(tardiness)

