'''
Comparison of Alg_disjoint and Gurobi implementation for solving instances of RAP-DIBC.

'''



import gurobipy as gp
from gurobipy import GRB

import math
import numpy
import itertools
import heapq
import random
random.seed(42)

import time
import numpy as np
import copy
from Alg_disjoint import RAP_disjoint

import pandas as pd
import csv



#For the EV evaluation


EV = pd.read_csv('EV_100days_40houses_synthetic.csv')

Start_interval = 4*18 
End_interval = Start_interval + 56
num_var = 56
num_interval = 2

Rate_min = 1100
Rate_max = 6600
Delta_t = 1/4
R_factors = [.25,.5,1]

lower_var = [0]*num_var
upper_fixed = [0]
lower_fixed = [Rate_min]
upper_var = [Rate_max]*num_var

num_house = 40
num_days = 100

Time_array_EV = np.zeros((num_house,num_days,len(R_factors)))
Time_array_gurobi_EV = np.zeros((num_house,num_days,len(R_factors)))

for house in range(0,40):
    infile = open('EV_100days_40houses.csv', 'r')  # CSV file
    BaseLoad = []
    for row in csv.reader(infile):
        BaseLoad.append(row[house])
    infile.close()   

    for day in range(0,100):

        b = BaseLoad[day*96 + Start_interval : day*96 + End_interval]
        b = [float(b[i]) for i in range(0,len(b))]
        
        for R_factor in range(0,len(R_factors)):
            print('House:',house,'Day:',day,'R_factor:',R_factors[R_factor])
            R = 39000 / Delta_t * R_factors[R_factor]
        
            try:
            
                # Create a new model
                m = gp.Model("mip1")
                m.setParam('Timelimit', 3600)
                # Create variables
                #x = m.addVar(vtype=GRB.BINARY, name="x")
                #y = m.addVar(vtype=GRB.BINARY, name="y")
                #z = m.addVar(vtype=GRB.BINARY, name="z")
                
                #Create binary variables
                a = m.addMVar(shape = (num_var,num_interval), vtype = GRB.BINARY, name = "a")
                xx = m.addMVar(shape = num_var, vtype = GRB.CONTINUOUS, name = "xx")
            
                # Set objective
                #m.setObjective(x + y*y + 2 * z, GRB.MAXIMIZE)
                bb = np.array(b)
                #m.setObjective(gp.quicksum((xx[i] + bb[i])*(xx[i] + bb[i]) for i in range(num_var)))
                m.setObjective(gp.quicksum((xx[i] + bb[i])*(xx[i] + bb[i]) for i in range(num_var)))
            
                # Add constraint: x + 2 y + 3 z <= 4
                #m.addConstr(x + 2 * y + 3 * z <= 4, "c0")
                
                for i in range(0,num_var):
                    l = np.empty(num_interval)
                    l[0] = lower_var[i]
                    for j in range(1,num_interval):
                        l[j] = lower_fixed[j-1]
                    u = np.empty(num_interval)
                    for j in range(0,num_interval - 1):
                        u[j] = upper_fixed[j]
                    u[num_interval - 1] = upper_var[i]
                    
                    
                    m.addConstr(gp.quicksum(l[j] * a[i, j] for j in range(num_interval)) - xx[i] <= 0)
                    m.addConstr(gp.quicksum(u[j] * a[i, j] for j in range(num_interval)) -xx[i]>= 0)
                    m.addConstr(gp.quicksum(a[i,j] for j in range(num_interval)) == 1)
                
                m.addConstr(gp.quicksum(xx[i] for i in range(num_var)) == R)
                
    
            
                # Add constraint: x + y >= 1
                #m.addConstr(x + y >= 1, "c1")
            
                # Optimize model
                m.optimize()
                runtime = m.Runtime
                print('Runtime is',runtime)
                #print(m.display())
            
                #for v in m.getVars():
                #    print('%s %g' % (v.VarName, v.X))
            
                print('Obj: %g' % m.ObjVal)
            
            except gp.GurobiError as e:
                print('Error code ' + str(e.errno) + ': ' + str(e))
            
            except AttributeError:
                print('Encountered an attribute error')
            
            Time_array_gurobi_EV[house,day,R_factor] = runtime
            
            
            #Our algorithm
            HELP_time_01 = time.perf_counter_ns()
            Answer = RAP_disjoint(R,b,lower_fixed,upper_fixed,lower_var,upper_var)
            HELP_time_02 = time.perf_counter_ns()
            print('First time: ',HELP_time_01)
            print('Second time:',HELP_time_02)
            print('The runtime was',(HELP_time_02 - HELP_time_01) / (10 ** 9) )
            Time_array_EV[house,day,R_factor] = (HELP_time_02 - HELP_time_01)  / (10 ** 9)

    
        
        




#For scalability

NUM_VAR = [[10,20,50,100,200,500,1000,2000,5000,10000,20000,50000,100000],
           [10,20,50,100,200,500,1000,2000,5000,10000],
           [10,20,50,100,200,500,1000]]
NUM_INT = [2,
           3,
           4]
num_samples = 1
Time_array = np.zeros((len(NUM_VAR[0]),len(NUM_INT),num_samples))
Time_array_gurobi = np.zeros((len(NUM_VAR[0]),len(NUM_INT),num_samples))

for int_index, num_interval in enumerate(NUM_INT):
    for var_index, num_var in enumerate(NUM_VAR[int_index]):

        for sample in range(0,num_samples):
            print('Variables:',num_var,'; intervals:',num_interval,'; sample',sample)
            
            #Initialize parameters
            b = [0]*num_var
            b[num_var - 1] = 0
            for i in range(num_var-2,-1,-1):
                b[i] = b[i + 1] + random.uniform(0,1)

            lower_fixed = [0]*(num_interval - 1)
            upper_fixed = [0]*(num_interval - 1)
            upper_fixed[0] = 2
            lower_fixed[0] = upper_fixed[0] + random.uniform(0,1)
            for j in range(1,num_interval - 1):
                upper_fixed[j] = lower_fixed[j - 1] + random.uniform(0,1)
                lower_fixed[j] = upper_fixed[j] + random.uniform(0,1)
            lower_var = [0]*num_var
            upper_var = [0]*num_var
            lower_var[num_var - 1] = 1
            for i in range(num_var - 2,-1,1):
                lower_var[i] = lower_var[i + 1] - random.uniform(0,1 / num_var)
            upper_var[0] = lower_fixed[num_interval - 2] + 1
            for i in range(1,num_var):
                upper_var[i] = upper_var[i - 1] - random.uniform(0,1 / num_var)
            R = random.uniform(sum(lower_var), sum(upper_var))

            
            #Gurobi
            try:
            
                # Create a new model
                m = gp.Model("mip1")
                m.setParam('Timelimit', 3600)
                
                #Create variables
                a = m.addMVar(shape = (num_var,num_interval), vtype = GRB.BINARY, name = "a")
                xx = m.addMVar(shape = num_var, vtype = GRB.CONTINUOUS, name = "xx")
                       
                # Set the objective
                bb = np.array(b)
                m.setObjective(gp.quicksum((xx[i] + bb[i])*(xx[i] + bb[i]) for i in range(num_var)))
            

                # Set constraints                
                for i in range(0,num_var):
                    l = np.empty(num_interval)
                    l[0] = lower_var[i]
                    for j in range(1,num_interval):
                        l[j] = lower_fixed[j-1]
                    u = np.empty(num_interval)
                    for j in range(0,num_interval - 1):
                        u[j] = upper_fixed[j]
                    u[num_interval - 1] = upper_var[i]
                    
                    
                    m.addConstr(gp.quicksum(l[j] * a[i, j] for j in range(num_interval)) - xx[i] <= 0)
                    m.addConstr(gp.quicksum(u[j] * a[i, j] for j in range(num_interval)) -xx[i]>= 0)
                    m.addConstr(gp.quicksum(a[i,j] for j in range(num_interval)) == 1)
                
                m.addConstr(gp.quicksum(xx[i] for i in range(num_var)) == R)
                

            

                # Optimize model
                m.optimize()
                runtime = m.Runtime
                print('Runtime is',runtime)

            
            except gp.GurobiError as e:
                print('Error code ' + str(e.errno) + ': ' + str(e))
            
            except AttributeError:
                print('Encountered an attribute error')
            
            Time_array_gurobi[var_index,int_index,sample] = runtime
            
            
            #Our algorithm
            HELP_time_01 = time.perf_counter_ns()
            Answer = RAP_disjoint(R,b,lower_fixed,upper_fixed,lower_var,upper_var)
            HELP_time_02 = time.perf_counter_ns()
            print('The runtime was',(HELP_time_02 - HELP_time_01) / (10 ** 9) )
            Time_array[var_index,int_index,sample] = (HELP_time_02 - HELP_time_01)  / (10 ** 9)
            
np.save('Time_data_alg',Time_array)
np.save('Time_data_gurobi',Time_array_gurobi)