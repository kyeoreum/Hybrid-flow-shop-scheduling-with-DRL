# read data
from os import path

job = [3, 5, 7]
stage = [5, 10]
machine = [3,5,7]

for jo in job:
    for st in stage:
        for ma in machine:
            for num in range(0, 10):
                f = open("C:/Users/user/Desktop/CPLEXData/CPLEXData_"+str(jo)+"_"+str(st)+"_"+str(ma)+"_"+str(num)+".txt",'r')

                buff = f.readline()
                temp = []
                job_temp = []

                while buff != '':
                    temp.append(buff)
                    buff = f.readline()

                job_num = int(temp[0])
                stage_num = int(temp[1])
                alternative_machine = int(temp[2])
                qtl_num = int(temp[3])

                del temp[0:4]

                path_count = temp[0]
                path_count = path_count.split(' ')
                path_count = [int(line.strip()) for line in path_count]
                tmp = int(path_count[1] - path_count[0] + path_count[3] - path_count[2] + 1)

                del temp[0:2]

                # path
                path = []
                for i in (temp[0:tmp+1]):
                    path.append(i)

                for i in range(0, len(path)):
                    path[i] = path[i].rstrip()
                    path[i] = path[i].split(' ')
                del path[-1]

                for i in range(0, len(path)):
                    for j in range(0, len(path[i])):
                        path[i][j] = int(path[i][j])
                del temp[0:tmp+1]

                rework_setup = []
                qtl_1 = []
                qtl_2 = []
                job_processing = []
                job_due_date = []
                job_ready_time = []
                i = 0
                while i < len(temp):
                    rework_setup.append(temp[i])
                    qtl_1.append(temp[i+1])
                    qtl_2.append(temp[i+2])
                    job_processing.append(temp[i+3])
                    job_due_date.append(temp[i+4])
                    job_ready_time.append(temp[i+5])
                    i += 7

                
                def preprocessing(start,end,line):
                    for i in range(start, end):
                        line[i] = line[i].rstrip()
                        line[i] = line[i].split(' ')

                def strtoint(start, end, line):
                    for i in range(start, end):
                        for j in range(start, len(line[i])):
                            line[i][j] = int(line[i][j])


                preprocessing(0, len(rework_setup), rework_setup)
                strtoint(0, len(rework_setup), rework_setup)

                preprocessing(0, len(qtl_1), qtl_1)
                strtoint(0, len(qtl_1), qtl_1)

                preprocessing(0, len(qtl_2), qtl_2)
                strtoint(0, len(qtl_2), qtl_2)

                preprocessing(0, len(job_processing), job_processing)
                strtoint(0, len(job_processing), job_processing)

                preprocessing(0, len(job_due_date), job_due_date)
                strtoint(0, len(job_due_date), job_due_date)

                preprocessing(0, len(job_ready_time), job_ready_time)
                strtoint(0, len(job_ready_time), job_ready_time)


                # Mathematical Model
                c = open("C:/Users/user/Desktop/new_LP/LPfile_"+str(jo)+"_"+str(st)+"_"+str(ma)+"_"+str(num)+".lp", 'w')

                c.write("Objective Function")
                c.write("\n")
                c.write("Minimize Tmax")
                c.write("\n")
                c.write("Subject to")
                c.write("\n")
                c.write("\n")


                route = len(path)

                #Constraint 1
                for i in range(1, job_num+1):
                    for p in range(1, route+1):
                        if p < route:
                            c.write("Z("+ str(i)+ ","+ str(p)+") + ")
                        else:
                            c.write("Z("+str(i)+","+str(p)+") = 1")
                    c.write("\n")

                c.write("\n")

                #Constraint 2
                for i in range(1, job_num+1):
                    for p in range(0, route):
                        for pp in range(0, len(path[p])):
                                for k in range(1, alternative_machine+1):
                                    if k < alternative_machine and path[p][pp] != 100:
                                        c.write("X("+str(i)+","+str(p+1)+","+str(path[p][pp])+","+str(k)+") + " )
                                    elif path[p][pp] == 100:
                                        c.write("X("+str(i)+","+str(p+1)+","+str(path[p][pp])+","+str(0)+") - Z("+str(i)+","+str(p+1)+") = 0" )
                                        break

                                    else:
                                        c.write("X("+str(i)+","+str(p+1)+","+str(path[p][pp])+","+str(k)+") - Z("+str(i)+","+str(p+1)+") = 0" )
                                c.write("\n")
                        c.write("\n")        
                c.write("\n")


                #Constraint 3
                for i in range(1, job_num+1):
                    for p in range(0, route):
                        for pp in range(0, 1):
                                for k in range(1, alternative_machine+1):
                                    c.write("S("+str(i)+","+str(p+1)+","+str(path[p][pp])+","+str(k)+") - 9999X("+str(i)+","+str(p+1)+","+str(path[p][pp])+","+str(k)+") - R("+str(i)+") >= -9999")                    
                                    c.write("\n")        
                c.write("\n")




                for i in range(1,job_num+1):
                    c.write("R("+str(i)+") = "+str(job_ready_time[i-1][0]) + str("\n"))
                c.write("\n")


                #Constraint 4
                for i in range(1, job_num+1):
                    for p in range(0, route):
                        for pp in range(0, len(path[p])):
                                for k in range(1, alternative_machine+1):
                                    if path[p][pp] == 100:
                                        c.write("S("+str(i)+","+str(p+1)+","+str(path[p][pp])+","+str(0)+") + "+"C("+str(i)+","+str(p+1)+","+str(path[p][pp])+","+str(0)+")"+" - "+"9999X("+str(i)+","+str(p+1)+","+str(path[p][pp])+","+str(0)+") <= 0"+str("\n"))
                                        break
                                    else:
                                        c.write("S("+str(i)+","+str(p+1)+","+str(path[p][pp])+","+str(k)+") + "+"C("+str(i)+","+str(p+1)+","+str(path[p][pp])+","+str(k)+")"+" - "+"9999X("+str(i)+","+str(p+1)+","+str(path[p][pp])+","+str(k)+") <= 0 ")
                                    c.write("\n")
                        c.write("\n")
                c.write("\n")

                #Constraint 5
                for i in range(1, job_num+1):
                    for p in range(0, route):
                        for pp in range(0, len(path[p])):
                                for k in range(1, alternative_machine+1):
                                    if path[p][pp] == 100:
                                        c.write("S("+str(i)+","+str(p+1)+","+str(path[p][pp])+","+str(0)+") + "+ "T("+str(i)+","+str(path[p][pp])+","+str(0)+") - "  +"C("+str(i)+","+str(p+1)+","+str(path[p][pp])+","+str(0)+")"+" + "+"999999X("+str(i)+","+str(p+1)+","+str(path[p][pp])+","+str(0)+") <= 999999 "+str("\n"))
                                        break
                                    else:

                                        c.write("S("+str(i)+","+str(p+1)+","+str(path[p][pp])+","+str(k)+") + "+ "T("+str(i)+","+str(path[p][pp])+","+str(k)+") - "  +"C("+str(i)+","+str(p+1)+","+str(path[p][pp])+","+str(k)+")"+" + "+"999999X("+str(i)+","+str(p+1)+","+str(path[p][pp])+","+str(k)+") <= 999999 ")
                                    c.write("\n")
                        c.write("\n")
                c.write("\n")

                # T data generator
                for i in range(1,job_num+1):
                    for p in range(0, route):
                        for pp in range(0, len(path[p])):
                            for k in range(1, alternative_machine+1):
                                if path[p][pp] == 100:
                                    c.write("T("+str(i)+","+str(path[p][pp])+","+str(0)+") = "+str(rework_setup[i-1][0]) + str("\n"))
                                    break
                                elif path[p][pp] > 100:     
                                    c.write("T("+str(i)+","+str(path[p][pp])+","+str(k)+") = "+str(job_processing[i-1][path[p][pp]-100-1])+str("\n"))
                                else:
                                    c.write("T("+str(i)+","+str(path[p][pp])+","+str(k)+") = "+str(job_processing[i-1][path[p][pp]-1])+str("\n"))
                                    
                c.write("\n")



                #Constraint 6
                for i in range(1, job_num+1):
                    for p  in range(0, route):
                        for pp in range(0, len(path[p])):
                                for k in range(1, alternative_machine+1):

                                    for i_r in range(1, job_num+1):
                                        for p_r in range(0, route):
                                            for pp_r in range(0,len(path[p_r])):
                                                if (i != i_r) and path[p][pp] != 100 and path[p_r][pp_r] != 100:
                                                    c.write("S("+str(i)+","+str(p+1)+","+str(path[p][pp])+","+str(k)+") + "+"9999Y("+str(i)+","+str(p+1)+","+str(path[p][pp])+","+str(i_r)+","+str(p_r+1)+","+str(path[p_r][pp_r])+","+str(k)+") - " + "C("+str(i_r)+","+str(p_r+1)+","+str(path[p_r][pp_r])+","+str(k)+") >= 0")
                                                    c.write("\n")
                                                if (i != i_r) and path[p][pp] == 100 and path[p_r][pp_r] == 100 and k == 1:
                                                    c.write("S("+str(i)+","+str(p+1)+","+str(path[p][pp])+","+str(0)+") + "+"9999Y("+str(i)+","+str(p+1)+","+str(path[p][pp])+","+str(i_r)+","+str(p_r+1)+","+str(path[p_r][pp_r])+","+str(0)+") - " + "C("+str(i_r)+","+str(p_r+1)+","+str(path[p_r][pp_r])+","+str(0)+") >= 0")
                                                    c.write("\n")
                                    
                                                

                                c.write("\n")
                    c.write("\n")                    
                                        
                #Consraint 7
                for i in range(1, job_num+1):
                    for p in range(0, route):
                        for pp in range(0 ,len(path[p])):
                            for k in range(1, alternative_machine +1):
                                
                                for i_r in range(1, job_num+1):
                                    for p_r in range(0, route):
                                        for pp_r in range(0,len(path[p_r])):
                                            if (i != i_r) and path[p][pp] != 100 and path[p_r][pp_r] != 100:
                                                c.write("S("+str(i)+","+str(p+1)+","+str(path[p][pp])+","+str(k)+") - "+"9999Y("+str(i_r)+","+str(p_r+1)+","+str(path[p_r][pp_r])+","+str(i)+","+str(p+1)+","+str(path[p][pp])+","+str(k)+") - " + "C("+str(i_r)+","+str(p_r+1)+","+str(path[p_r][pp_r])+","+str(k)+") >= -9999")
                                                c.write("\n")
                                            if (i != i_r) and path[p][pp] == 100 and path[p_r][pp_r] == 100 and k == 1:
                                                c.write("S("+str(i)+","+str(p+1)+","+str(path[p][pp])+","+str(0)+") - "+"9999Y("+str(i_r)+","+str(p_r+1)+","+str(path[p_r][pp_r])+","+str(i)+","+str(p+1)+","+str(path[p][pp])+","+str(0)+") - " + "C("+str(i_r)+","+str(p_r+1)+","+str(path[p_r][pp_r])+","+str(0)+") >= -9999")
                                                
                                                c.write("\n")
                                    
                                            
                                        
                c.write("\n")

                #Constraint 8
                for i in range(1, job_num+1):
                    for p in range(0, route):
                        for pp in range(1 ,len(path[p])):
                            for k in range(1, alternative_machine +1):
                                if path[p][pp] == 100:
                                    c.write("S("+str(i)+","+str(p+1)+","+str(path[p][pp])+","+str(0)+") - ")
                                    break
                                elif k < alternative_machine:
                                    c.write("S("+str(i)+","+str(p+1)+","+str(path[p][pp])+","+str(k)+") + ")
                                elif k == alternative_machine:
                                    c.write("S("+str(i)+","+str(p+1)+","+str(path[p][pp])+","+str(k)+") - ")
                            
                            for k_r in range(1, alternative_machine +1):
                                if path[p][pp-1] == 100:
                                    c.write("C("+str(i)+","+str(p+1)+","+str(path[p][pp-1])+","+str(0)+") >= 0 "+str("\n"))
                                    break
                                elif k_r < alternative_machine:
                                    c.write("C("+str(i)+","+str(p+1)+","+str(path[p][pp-1])+","+str(k_r)+") - ")
                                elif k_r == alternative_machine:
                                    c.write("C("+str(i)+","+str(p+1)+","+str(path[p][pp-1])+","+str(k_r)+") >= 0 "+str("\n"))
                            
                            
                c.write("\n")


                #Constraint 9
                for i in range(1, job_num+1):
                    for p in range(0, 1):
                        for pp in range(1, len(path[p])):
                            for pp_r in range(0, pp):    
                                if (qtl_1[i-1][1] == path[p][pp_r] and qtl_1[i-1][2] == path[p][pp]) or (qtl_2[i-1][1] == path[p][pp_r] and qtl_2[i-1][2] == path[p][pp]):
                                    for k in range(1, alternative_machine+1):
                                    
                                            if k < alternative_machine:
                                                c.write("S("+str(i)+","+"1"+","+str(path[p][pp])+","+str(k)+") + ")
                                            elif k == alternative_machine:
                                                c.write("S("+str(i)+","+"1"+","+str(path[p][pp])+","+str(k)+") - ")

                                    for k in range(1, alternative_machine+1):
                
                                            if k < alternative_machine :
                                                c.write("C("+str(i)+","+"1"+","+str(path[p][pp_r])+","+str(k)+") - ")
                                            elif k == alternative_machine:
                                                c.write("C("+str(i)+","+"1"+","+str(path[p][pp_r])+","+str(k)+") - Q("+str(i)+","+str(path[p][pp_r])+","+str(path[p][pp])+")Z("+ str(i)+ ","+"1"+") <= 0"+str("\n"))
                                            

                        
                    c.write("\n") 


                    
                #Constraint 10
                for i in range(1, job_num+1):
                    for p in range(1, route):
                        for q in range(0, 1):
                            for pp in range(1, len(path[p])):
                                for qq in range(1, len(path[q])):
                                    for pp_r in range(0, pp):    
                                        for qq_r in range(0, qq):
                                            if (qtl_1[i-1][1] == path[q][qq_r] and qtl_1[i-1][2] == path[q][qq]) or (qtl_2[i-1][1] == path[q][qq_r] and qtl_2[i-1][2] == path[q][qq]):
                                                for k in range(1, alternative_machine+1):
                                                
                                                        if k < alternative_machine:
                                                            c.write("S("+str(i)+","+str(p+1)+","+str(path[p][pp])+","+str(k)+") + ")
                                                        elif k == alternative_machine:
                                                            c.write("S("+str(i)+","+str(p+1)+","+str(path[p][pp])+","+str(k)+") - ")

                                                for k in range(1, alternative_machine+1):
                                                
                                                        if k < alternative_machine :
                                                            c.write("C("+str(i)+","+str(p+1)+","+str(path[p][pp_r])+","+str(k)+") - ")
                                                        elif k == alternative_machine and (qtl_1[i-1][1] == path[q][qq_r] and qtl_1[i-1][2] == path[q][qq]) or (qtl_2[i-1][1] == path[q][qq_r] and qtl_2[i-1][2] == path[q][qq]):
                                                            c.write("C("+str(i)+","+str(p+1)+","+str(path[p][pp_r])+","+str(k)+") - Q("+str(i)+","+str(path[q][qq_r])+","+str(path[q][qq])+")Z("+ str(i)+ ","+ str(p+1)+") >= 0"+str("\n"))

                                        
                        
                    
                    c.write("\n")   



                for i in range(1, job_num+1):
                    for p in range(0, 1):
                        for pp in range(1, len(path[p])):
                            for pp_r in range(0, pp):    
                                for k in range(1, alternative_machine+1):
                                    
                                    if qtl_1[i-1][1] == path[p][pp_r] and qtl_1[i-1][2] == path[p][pp]:
                                        c.write("Q("+str(i)+","+str(path[p][pp_r])+","+str(path[p][pp])+") = "+str(qtl_1[i-1][0])+str("\n"))
                c.write("\n") 
                    

                for i in range(1, job_num+1):
                    for p in range(0, 1):
                        for pp in range(1, len(path[p])):
                            for pp_r in range(0, pp):    
                                for k in range(1, alternative_machine+1):
                                    if qtl_2[i-1][1] == path[p][pp_r] and qtl_2[i-1][2] == path[p][pp]:
                                        c.write("Q("+str(i)+","+str(path[p][pp_r])+","+str(path[p][pp])+") = "+str(qtl_2[i-1][0])+str("\n"))
                                

                c.write("\n") 



                #Constraint Tmax
                for i in range(1, job_num+1):
                    for p in range(0, route):
                        for pp in range(len(path[p])-1,len(path[p])):
                            c.write("C("+str(i)+") - ")
                            for k in range(1, alternative_machine+1):
                                if k < alternative_machine:
                                    c.write("C("+str(i)+","+str(p+1)+","+str(path[p][pp])+","+str(k)+") - ")
                                elif k == alternative_machine:
                                    c.write("C("+str(i)+","+str(p+1)+","+str(path[p][pp])+","+str(k)+") >= 0"+str("\n"))
                                    

                c.write("\n")
                for i in range(1, job_num+1):
                    
                    c.write("T("+str(i)+") - C("+str(i)+") + d("+str(i)+") >= 0 "+str("\n"))
                           
                c.write("\n")
                c.write("\n")

                c.write("Tmax -")
                for i in range(1, job_num+1):
                    if i < job_num:
                        c.write("T("+str(i)+")-")
                    else:
                        c.write("T("+str(i)+") >= 0")
                c.write("\n")


                for i in range(1,job_num+1):
                    c.write("d("+str(i)+") = "+str(job_due_date[i-1][0]) + str("\n"))
                c.write("\n")


                #Constraint 10
                for i in range(1, job_num+1):
                    for p in range(0, route):
                        for pp in range(0, len(path[p])):
                            for k in range(1, alternative_machine+1):
                                if path[p][pp] == 100:
                                    c.write("S("+str(i)+","+str(p+1)+","+str(path[p][pp])+","+str(0)+") >= 0"+str("\n"))
                                    break
                                else:
                                    c.write("S("+str(i)+","+str(p+1)+","+str(path[p][pp])+","+str(k)+") >= 0"+str("\n"))

                c.write("\n")

                for i in range(1, job_num+1):
                    for p in range(0, route):
                        for pp in range(0, len(path[p])):
                            for k in range(1, alternative_machine+1):
                                if path[p][pp] == 100:
                                    c.write("C("+str(i)+","+str(p+1)+","+str(path[p][pp])+","+str(0)+") >= 0"+str("\n"))
                                    break
                                else:
                                    c.write("C("+str(i)+","+str(p+1)+","+str(path[p][pp])+","+str(k)+") >= 0"+str("\n"))
                                    

                c.write("\n")
                for i in range(1, job_num+1):
                    c.write("C("+str(i)+") >= 0"+str("\n"))
                c.write("\n")

                c.write("BINARY")
                c.write("\n")
                for i in range(1, job_num+1):
                    for p in range(0, route):
                        c.write("Z("+str(i)+","+str(p+1)+")"+str("\n"))

                c.write("\n")
                c.write("\n")
                for i in range(1, job_num+1):
                    for p in range(0, route):
                        for pp in range(0, len(path[p])):
                            for k in range(1, alternative_machine+1):
                                if path[p][pp] == 100:
                                    c.write("X("+str(i)+","+str(p+1)+","+str(path[p][pp])+","+str(0)+")"+str("\n"))
                                    break
                                else:
                                    c.write("X("+str(i)+","+str(p+1)+","+str(path[p][pp])+","+str(k)+")"+str("\n"))

                c.write("\n")
                c.write("\n")

                for i in range(1, job_num+1):
                    for p  in range(0, route):
                        for pp in range(0, len(path[p])):
                                for k in range(1, alternative_machine+1):
                                    for i_r in range(1, job_num+1):
                                        for p_r in range(0, route):
                                            for pp_r in range(0,len(path[p_r])):
                                                if (i != i_r) and path[p][pp] != 100 and path[p_r][pp_r] != 100:
                                                    c.write("Y("+str(i)+","+str(p+1)+","+str(path[p][pp])+","+str(i_r)+","+str(p_r+1)+","+str(path[p_r][pp_r])+","+str(k)+")")
                                                    c.write("\n")
                                                if (i != i_r) and path[p][pp] == 100 and path[p_r][pp_r] == 100 and k == 1:
                                                    c.write("Y("+str(i)+","+str(p+1)+","+str(path[p][pp])+","+str(i_r)+","+str(p_r+1)+","+str(path[p_r][pp_r])+","+str(0)+")")
                                                    c.write("\n")
                                c.write("\n")
                    c.write("\n")                    
                                        
                c.write("END")
