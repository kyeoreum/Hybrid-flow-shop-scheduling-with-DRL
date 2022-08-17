from audioop import avg
from unittest import result
import numpy as np
import random
import copy
import time


start_time = time.time()


population = 25
generation = 30
function = ['+', '-', '*', '/', 'max', 'min']
# Terminal set 추가할 것!!!
#Terminal = ['job_temp[sq[o]][-1][0]', '1', 're_job_temp[sq[o]][3][i]','job_flow[sq[o]]', '(job_temp[sq[o]][-1][0] - job_flow[sq[o]])/re_job_temp[sq[o]][3][i]'] 
Terminal = ['1', 're_job_temp[sq[o]][3][i]', 'job_flow[sq[o]]', 'job_temp[sq[o]][-1][0]', 'job_temp[sq[o]][-1][0] - job_flow[sq[o]]','job_temp[sq[o]][2][0]', 'job_temp[sq[o]][1][0]', 're_job_temp[sq[o]][3][i]/job_temp[sq[o]][-1][0]'] 

Frequency = []
for i in range(0, len(Terminal)):
    Frequency.append(0)
max_depth = 4
min_depth = 2


# ramped half and half : Tree generation 
# 1. max tree
GP_Tree = []
for p in range(0, population):
    tree1 = []
    len1 = 1
    for i in range(0, max_depth):
        sub_tree = []
        for j in range(0, len1):
            
            if i < max_depth - 1: # function
                x = random.choice(function)
                sub_tree.append(x)
            else:
                x = random.choice(Terminal)
                sub_tree.append(x)
        tree1.append(sub_tree)
        len1 *= 2
    GP_Tree.append(tree1)


# 2. random tree
for p in range(0, population):
    tree2 = []
    len2 = 1
    for i in range(0, max_depth):
        sub_tree = []
        if i < min_depth:
            for j in range(0, len2):
                x = random.choice(function)
                sub_tree.append(x)
            len2 *= 2
        else:
            for z in tree2[-1]:
                if z in Terminal:
                    for k in range(0, 2):
                        pass
                else:
                    y = random.randint(0, 1)
                                        
                    if y == 1 and i < max_depth-1:
                        for k in range(0, 2):
                            x = random.choice(function)
                            sub_tree.append(x)
                    else:
                        for k in range(0,2):
                            x = random.choice(Terminal)
                            sub_tree.append(x)
        tree2.append(sub_tree)
    GP_Tree.append(tree2)
    

def flat_list(sentence):  
    rules = []
    for i in sentence:
        if type(i) == type(list()):
            rules += (flat_list(i))
        else:
            rules.append(i)
    return rules


def make_rule(tree):
    sentence = copy.deepcopy(tree)
   
    md = len(sentence)-1
    while md > 1:
        x= []
        
        if len(sentence[md]) == 0:
            del sentence[md]                
            md -= 1
              
        for i_x in range(0, len(sentence[md-1])):
            
            if sentence[md-1][i_x] in function:
              
                a = sentence[md][0]
                b = sentence[md-1][i_x]
                c = sentence[md][1]

                if (b == '-' or b == 'min' or b == 'max') and a == c:
                    while a == c:
                        c = random.choice(Terminal)
                        sentence[md][1] = c

                x.append("(")
                x.append(sentence[md].pop(0))   
                x.append(sentence[md-1].pop(i_x))
                x.append(sentence[md].pop(0))
                x.append(")")         
                   
                # max min 처리
                for chk in range(0, len(x)):     
                    if x[chk] == "max" or x[chk] == "min":               
                        x[chk], x[chk-1] = x[chk-1], x[chk]
                        x.insert(chk, "(")
                        x.insert(chk+2, ",")
                        x.insert(chk+4, ")")
                sentence[md-1].insert(i_x, x)
                x = []
                break
    del sentence[-1]
   
    r_rule = "".join(flat_list(sentence))
    return r_rule

def re_generation():
    tree2 = []
    len2 = 1
    for i in range(0, max_depth):
        sub_tree = []
        if i < min_depth:
            for j in range(0, len2):
                x = random.choice(function)
                sub_tree.append(x)
            len2 *= 2
        else:
            for z in tree2[-1]:
                if z in Terminal:
                    for k in range(0, 2):
                        pass
                else:
                    y = random.randint(0, 1)
                                        
                    if y == 1 and i < max_depth-1:
                        for k in range(0, 2):
                            x = random.choice(function)
                            sub_tree.append(x)
                    else:
                        for k in range(0,2):
                            x = random.choice(Terminal)
                            sub_tree.append(x)
        tree2.append(sub_tree)
    return tree2

re_g = []
for i in range(0, len(GP_Tree)):
    re_g.append(0)

GP_rules = []

for gp in range(0, len(GP_Tree)):
    GP_rules.append(make_rule(GP_Tree[gp]))
    

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


################# simulator _ fitness evaluation #################
def simulator(GP_rules):
    
    job = [30]
    stage = [5]
    alternative_machine = [3]
    cnt = 0
    avg_tard = []
    for rule_num in GP_rules:
    
        tt = 0
        for jn in job:
            for sn in stage:
                for al in alternative_machine:
                    for num in range(0,5):
                        
                        f = open("C:/Users/user/source/repos/my_thesis/TESTData2/TESTData_"+str(jn)+"_"+str(sn)+"_"+str(al)+"_"+str(num)+".txt", 'r')                            
                        
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
                            for j in range(0, jn):
                                stage_queue[i].append(j)
                        cut = 10

                        def edd(i):
                            for n in range(0, len(stage_queue[i])-1):
                                for m in range(1, len(stage_queue[i])):
                                    if job_temp[stage_queue[i][n]][-1][0] > job_temp[stage_queue[i][m]][-1][0]:
                                        stage_queue[i][n], stage_queue[i][m] = stage_queue[i][m], stage_queue[i][n]

                    
                        ####### Simulator start #######
                        rework_station_queue = []
                        rework_machine = 9999
                        check = 0
                   
                        #print(rand_pois)
                        time = -1

                        while check != job_num:
                            check = 0
                            for i in range(0, job_num):
                                if job_fin[i] == 1:
                                    check += 1
                            
                            time += 1
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



                                            fit_eval(i,stage_queue[i], rule_num, job_temp, stage_queue, job_flow, completion_time, re_job_temp) 
                               # queue 안의 job들 순서대로 sequencing


                                    # ith stage, j th machine is empty : pop 
                                    elif machine_state[i][j] == 9999 and len(stage_queue[i]) != 0:               
                                        machine_state[i][j] = stage_queue[i].pop(0)
                                        job_flow[machine_state[i][j]] += 1
                                        machine_chk[machine_state[i][j]].append(j)
                                        start_time[machine_state[i][j]].append(time)   
                                        fit_eval(i,stage_queue[i], rule_num, job_temp, stage_queue, job_flow, completion_time, re_job_temp) 
                            
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
                                    job_flow[rework_machine] += 1
                                    machine_chk[rework_machine].append("RS")
                                    start_time[rework_machine].append(time)

                                    fit_eval(i,stage_queue[i], rule_num, job_temp, stage_queue, job_flow, completion_time, re_job_temp) 
                            
        
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

                                            fit_eval(i,stage_queue[i], rule_num, job_temp, stage_queue, job_flow, completion_time, re_job_temp) 
                            

                                        # return to the original stage 2
                                        elif job_temp[rework_machine][-1][-1] == 2:
                                    
                                            for k in range(job_temp[rework_machine][2][1], job_temp[rework_machine][2][2]):
                                                job_temp[rework_machine][3][k] = re_job_temp[rework_machine][3][k]
                                            stage_queue[job_temp[rework_machine][2][1]].append(rework_machine)   
                                            job_flow[rework_machine] += 1                 
                                            completion_time[rework_machine].append(time)
                                            rework_machine = 9999

                                      
                                            fit_eval(i,stage_queue[i], rule_num, job_temp, stage_queue, job_flow, completion_time, re_job_temp) 
                            
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
                        #print("total_tardiness : ", total_tardiness)
                        tt += total_tardiness
          
               
        avg_tard.append(tt)

    
        tt = 0
        cnt += 1
    return avg_tard


def mutation(GP_tree, tree2):
    tree = copy.deepcopy(tree2)
    tmp = []
    t = 0
    while t < 10:
        for i in range(0, len(tree)):
            if len(tree[-1]) == 0:
                del tree[-1]

        n = 1
        num = random.randint(1, 3)
        while n <= num:    
            level = random.randint(0, len(tree)-1)
            index = random.randint(0, len(tree[level])-1)
            while tree[level][index] not in Terminal:
                level = random.randint(0, len(tree)-1)
                index = random.randint(0, len(tree[level])-1)

            x = random.randint(0, 5)
            tree[level][index] = Terminal[x]
            n += 1

        if tree not in tmp:        
            tmp.append(tree)
            t += 1

    ttmp = []
    for i in range(0, len(tmp)):
        ttmp.append(make_rule(tmp[i]))

    small = simulator(ttmp)

    work = 0
    m = small[0]
    for i in range(0, len(tmp)):
        if small[i] <= m:
            m = small[i]
            t = i
 
    if avg_tard[GP_tree.index(tree2)] > small[t]:
        tree2 = ttmp[t]
        work = 1
    return work

    
def prec(tree):

    tree_prec = [[0]]
    for i in range(0, len(tree)-1):
        tmp = []
        for j in range(0, len(tree[i])):
            if tree[i][j] in function:
                for x in range(0, 2):
                    tmp.append(j)
        tree_prec.append(tmp)
    return tree_prec


#crossover
def crossover(tree1, tree2):

    tree = copy.deepcopy(tree1)

    try:
        level = random.randint(1, min_depth)
        index = random.randint(0, len(tree[level])-1)
   
    except:
        while tree2[level][index] in Terminal or tree2[level][index] in function:
            level = random.randint(1, min_depth)
            index = random.randint(0, len(tree[level]-1))
    
    p_t1 = prec(tree)
    p_t2 = prec(tree2)

    # remove tree2 
    r_list = []
    r_list.append(index)
    tree2[level][index] = tree[level][index]
    
    for i in range(level+1, len(tree2)):
        r_list1 = []
        
        j  = 0
        while j < len(tree2[i]):
            if p_t2[i][j] in r_list:
                r_list1.append(j)
                p_t2[i][j] = "N"
                tree2[i][j] = "N"
                
            j += 1
        r_list = copy.deepcopy(r_list1)

    i = 0
    while i < len(tree2):
        j = 0
        while j < len(tree2[i]):
            if tree2[i][j] == "N":
                del tree2[i][j]
                del p_t2[i][j]
                j -= 1
            j += 1   
        i += 1             

    r_list = [index]

    co = []
    # Tree1 ==> Tree2
    for i in range(level+1, len(tree)):
        rt_list = []
        temp = []
        for j in range(0, len(tree[i])):
            if p_t1[i][j] in r_list:
                rt_list.append(j)
                temp.append(tree[i][j])
        r_list = copy.deepcopy(rt_list)
        co.append(temp)   

    # CO 붙이기
    r_list = [index]
    for i in range(level, len(tree2)-1):
        temp = []
        for j in range(0, len(tree2[i])):
            if j in r_list and tree2[i][j] in function:
                for x in range(0, 2):
                    p_t2[i+1].append(j)
                    p_t2[i+1].sort()
                    tree2[i+1].insert(p_t2[i+1].index(j), co[i-level].pop(-1))
              
                for k in range(0, len(p_t2[i+1])):
                    if p_t2[i+1][k] in r_list:
                        temp.append(k)
                
        r_list = copy.deepcopy(temp) 




# # fitness result ==> sorting
avg_tard = simulator(GP_rules)

for rules in range(0, len(GP_rules)):
    for rules2 in range(0, len(avg_tard)):
        if avg_tard[rules] < avg_tard[rules2]:
            GP_rules[rules], GP_rules[rules2] = GP_rules[rules2], GP_rules[rules]
            GP_Tree[rules], GP_Tree[rules2] = GP_Tree[rules2], GP_Tree[rules]
            avg_tard[rules], avg_tard[rules2] = avg_tard[rules2], avg_tard[rules]
    
print(GP_rules)
print(avg_tard)


for t in range(0, generation):
    for fre in range(0, 10):
        for fre1 in range(0, len(GP_Tree[fre])):
            if GP_Tree[fre][fre1] in Terminal:
                Frequency[Terminal.index(GP_Tree[fre][fre1])] += 1

    for re_gen in range(0, population*2):        
        # regeneration
        if re_gen >= population:
            re_g[re_gen] += 1
            if re_g[re_gen] >= 5:          
                GP_Tree[re_gen] = re_generation()

        # 1
        #crossover
        x = random.randint(0, 10)
        crossover(GP_Tree[x], GP_Tree[re_gen])    
                    
        # mutation
        work = mutation(GP_Tree, GP_Tree[re_gen])

        # 2
        if work == 0:
            mutation(GP_Tree, GP_Tree[re_gen])
            candidate = []
            k = 0
            while k < 10:
                x = random.randint(0, 10)
                crossover(GP_Tree[x], GP_Tree[re_gen]) 
                temp_tree = copy.deepcopy(GP_Tree[re_gen])
                if temp_tree not in candidate:
                    candidate.append(temp_tree)
                    k += 1
                    
            temp_rule = []
            for k in range(0, len(candidate)):
                temp_rule.append(make_rule(candidate[k]))
            temp_result = simulator(temp_rule)

            m = temp_result[0]
            for k in range(0, len(temp_rule)):
                if temp_result[k] <= m:
                    m = temp_result[k]
                    t = k
            if avg_tard[re_gen] > temp_result[t]:
                GP_Tree[re_gen] = candidate[t]


    GP_rules = []
    for gp in range(0, len(GP_Tree)):
        GP_rules.append(make_rule(GP_Tree[gp]))


    avg_tard = simulator(GP_rules)
    for rules in range(0, len(GP_rules)):
        for rules2 in range(0, len(avg_tard)):
            if avg_tard[rules] < avg_tard[rules2]:
                GP_rules[rules], GP_rules[rules2] = GP_rules[rules2], GP_rules[rules]
                GP_Tree[rules], GP_Tree[rules2] = GP_Tree[rules2], GP_Tree[rules]
                avg_tard[rules], avg_tard[rules2] = avg_tard[rules2], avg_tard[rules]
   
        
end_time = time.time()
print(end_time - start_time)


