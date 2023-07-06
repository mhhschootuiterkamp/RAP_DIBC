"""
Algorithm for solving (quadratic) instances of RAP-DIBCs, as presented in the paper:
    
"Symmetric separable convex resource allocation problems with structured disjoint interval bound constraints"
by M. H. H. Schoot Uiterkamp

"""

import math
import itertools
import heapq
import random
random.seed(42)

'''
Conditions on input:
    - List b is sorted non-increasingly;
    - lower_fixed (l^{\tilde}) and upper_fixed (u^{\tilde}), together lower_var (l_{i,1}) and upper_var (u_{i,n}) should satisfy conditions in the paper:
'''
def RAP_disjoint(R,b,lower_fixed, upper_fixed, lower_var, upper_var):
    
    #Dimensions
    num_var = len(b)
    num_intervals = len(lower_fixed) + 1  
    
    #Create partition trunks collection
    list_partitions = list(itertools.combinations_with_replacement(range(-1,num_var), num_intervals - 2))
    
    #Initialize bookkeeping for best solution and partition
    current_best_obj = math.inf
    current_best_partition = None
    current_best_mult = None
    
    #Start iterating over trunks of partitions
    for partition in list_partitions:
        partition_full = list(partition)
        if len(partition_full) == 0:
            partition_full.append(-1)
        else:
            partition_full.append(partition[-1])
        
        #Set lower and upper bounds of variables, given the current partition
        lower_bounds = [0]*num_var
        for i in range(0,partition_full[0] + 1):
            lower_bounds[i] = lower_var[i]
        for j in range(1,num_intervals - 2):
            for i in range(partition_full[j-1] + 1 , partition_full[j] + 1):
                lower_bounds[i] = lower_fixed[j-1]
        for i in range(partition_full[num_intervals - 2] + 1 , num_var):
            lower_bounds[i] = lower_fixed[num_intervals - 2]
                
        upper_bounds = [0]*num_var
        for i in range(0,partition_full[0] + 1):
            upper_bounds[i] = upper_fixed[0]
        for j in range(1,num_intervals - 2):
            for i in range(partition_full[j-1] + 1, partition_full[j] + 1):
                upper_bounds[i] = upper_fixed[j]
        for i in range(partition_full[num_intervals - 2] + 1, num_var):
            upper_bounds[i] = upper_var[i]
                
        #Calculate breakpoints
        lower_breakpoints = [lower_bounds[i] + b[i] for i in range(0,num_var)]
        upper_breakpoints = [upper_bounds[i] + b[i] for i in range(0,num_var)]
        
        #Initialize bookkeeping parameters
        HELP_var_Bounded = sum(lower_bounds)
        HELP_var_Free = 0
        HELP_var_num_free = 0
        HELP_var_value_Bound = sum([math.pow(lower_breakpoints[i],2) for i in range(0,num_var)])
        
        #Initialize breakpoint heaps
        heap_lower_BPs = [[lower_breakpoints[i],i,0] for i in range(0,num_var)]
        heap_upper_BPs = [[upper_breakpoints[i],i,0] for i in range(0,num_var)]
        heapq.heapify(heap_lower_BPs)
        heapq.heapify(heap_upper_BPs)
        HELP_num_lower_heap = num_var
        HELP_num_upper_heap = num_var
        
        HELP_feasible_check_lower = sum(lower_bounds)
        HELP_feasible_check_upper = sum(upper_bounds)
       
        #Start breakpoint search procedure
        while partition_full[-1] < num_var:
            if HELP_feasible_check_lower > R:
                Opt_mult = -math.inf
            elif HELP_feasible_check_upper < R:
                break
            else:
                FLAG_found = 0
                while FLAG_found == 0:
                    FLAG_found_lower = 0
                    while FLAG_found_lower == 0:
                        if HELP_num_lower_heap > 0:
                            Candidate_BP_lower = heap_lower_BPs[0][0]                            
                            if len(partition_full) == 1:
                                HELP_part_bound = -1
                            else:
                                HELP_part_bound = partition_full[-2]                           
                            if HELP_part_bound < heap_lower_BPs[0][1] and heap_lower_BPs[0][1] < partition_full[-1] and heap_lower_BPs[0][2] == 0:
                                heapq.heappop(heap_lower_BPs)
                                HELP_num_lower_heap -= 1
                            else:
                                FLAG_found_lower = 1
                        else:
                            Candidate_BP_lower = math.inf
                            FLAG_found_lower = 1                            
                    FLAG_found_upper = 0       
                    while FLAG_found_upper == 0:
                        if HELP_num_upper_heap > 0:
                            Candidate_BP_upper = heap_upper_BPs[0][0]                            
                            if len(partition_full) == 1:
                                HELP_part_bound = -1
                            else:
                                HELP_part_bound = partition_full[-2]                           
                            if HELP_part_bound < heap_upper_BPs[0][1] and heap_upper_BPs[0][1] <= partition_full[-1] and heap_upper_BPs[0][2] == 0:
                                heapq.heappop(heap_upper_BPs)
                                HELP_num_upper_heap -= 1
                            else:
                                FLAG_found_upper = 1
                        else:
                            Candidate_BP_upper = math.inf
                            FLAG_found_upper = 1        
                    
                    #Final selection candidate breakpoint
                    Candidate_BP = min(Candidate_BP_lower, Candidate_BP_upper)
                    HELP_resource = HELP_var_Bounded + HELP_var_num_free * Candidate_BP - HELP_var_Free
                    
                    if HELP_resource == R:
                        Opt_mult = Candidate_BP
                        Obj_value = HELP_var_value_Bound + HELP_var_num_free * math.pow(Opt_mult,2)
                        FLAG_found = 1
                    elif HELP_resource > R:
                       Opt_mult = (R - HELP_var_Bounded + HELP_var_Free) / HELP_var_num_free
                       Obj_value = HELP_var_value_Bound + HELP_var_num_free * math.pow(Opt_mult,2)                   
                       FLAG_found = 1
                    else:
                        if Candidate_BP_lower < Candidate_BP_upper:
                            HELP_index = heap_lower_BPs[0][1]
                            HELP_var_Bounded -= lower_bounds[HELP_index]
                            HELP_var_Free += b[HELP_index]
                            HELP_var_num_free += 1
                            HELP_var_value_Bound -= math.pow(lower_bounds[HELP_index] + b[HELP_index],2)
                            heapq.heappop(heap_lower_BPs)
                            HELP_num_lower_heap -= 1
                        else:
                            HELP_index = heap_upper_BPs[0][1]
                            HELP_var_Bounded += upper_bounds[HELP_index]
                            HELP_var_Free -= b[HELP_index]
                            HELP_var_num_free -= 1
                            HELP_var_value_Bound += math.pow(upper_bounds[HELP_index] + b[HELP_index],2)
                            heapq.heappop(heap_upper_BPs)
                            HELP_num_upper_heap -= 1
    
                #Compare to currently best obj_value
                if Obj_value < current_best_obj:
                    current_best_obj = Obj_value
                    current_best_partition = partition_full
                    current_best_mult = Opt_mult                
            
            #Update partition
            partition_full[-1] += 1
            HELP_index = partition_full[-1]
            
            if partition_full[-1] >= num_var:
                break
            else:
                #First update round (no removal of breakpoints due to labeling structure)
                if Opt_mult < lower_breakpoints[HELP_index]:
                    HELP_var_Bounded -= lower_bounds[HELP_index]
                    HELP_var_value_Bound -= math.pow(lower_bounds[HELP_index] + b[HELP_index],2)
                elif lower_breakpoints[HELP_index] < Opt_mult and Opt_mult < upper_breakpoints[HELP_index]:
                    HELP_var_Free -= b[HELP_index]
                    HELP_var_num_free -= 1
                else:
                    HELP_var_Bounded -= upper_bounds[HELP_index]
                    HELP_var_value_Bound -= math.pow(upper_bounds[HELP_index] + b[HELP_index],2)
                    
                    
                HELP_feasible_check_lower -= lower_bounds[HELP_index]
                HELP_feasible_check_upper -= upper_bounds[HELP_index]
                
                #Compute new BPs...
                if len(lower_fixed) == 1:
                    lower_breakpoints[HELP_index] = lower_var[HELP_index] + b[HELP_index]
                else:
                    lower_breakpoints[HELP_index] = lower_fixed[-2] + b[HELP_index]
                upper_breakpoints[HELP_index] = upper_fixed[-1] + b[HELP_index]
    
                if len(lower_fixed) == 1:
                    lower_bounds[HELP_index] = lower_var[HELP_index]
                else:
                    lower_bounds[HELP_index] = lower_fixed[-2]
                upper_bounds[HELP_index] = upper_fixed[-1]            
                
                HELP_feasible_check_lower += lower_bounds[HELP_index]
                HELP_feasible_check_upper += upper_bounds[HELP_index]
                
                #Second update round
                if Opt_mult < lower_breakpoints[HELP_index]:
                    HELP_var_Bounded += lower_bounds[HELP_index]
                    HELP_var_value_Bound += math.pow(lower_bounds[HELP_index] + b[HELP_index],2)
                    heapq.heappush(heap_lower_BPs, [lower_breakpoints[HELP_index],HELP_index,1])
                    HELP_num_lower_heap += 1
                    heapq.heappush(heap_upper_BPs, [upper_breakpoints[HELP_index],HELP_index,1])
                    HELP_num_upper_heap += 1
                elif lower_breakpoints[HELP_index] < Opt_mult and Opt_mult < upper_breakpoints[HELP_index]:
                    HELP_var_Free += b[HELP_index]
                    HELP_var_num_free += 1
                    heapq.heappush(heap_upper_BPs, [upper_breakpoints[HELP_index],HELP_index,1])
                    HELP_num_upper_heap += 1
                else:
                    HELP_var_Bounded += upper_bounds[HELP_index]
                    HELP_var_value_Bound += math.pow(upper_bounds[HELP_index] + b[HELP_index],2)             

    
    #Calculate final answer using optimal multiplier and partition
    lower_bounds = [0]*num_var
    for i in range(0,current_best_partition[0] ):
        lower_bounds[i] = lower_var[i]
    for j in range(1,num_intervals - 2):
        for i in range(current_best_partition[j-1]  , current_best_partition[j] ):
            lower_bounds[i] = lower_fixed[j-1]
    for i in range(current_best_partition[num_intervals - 2]  , num_var):
        lower_bounds[i] = lower_fixed[num_intervals - 2]

    upper_bounds = [0]*num_var
    for i in range(0,current_best_partition[0] ):
        upper_bounds[i] = upper_fixed[0]
    for j in range(1,num_intervals - 2):
        for i in range(current_best_partition[j-1] , current_best_partition[j] ):
            upper_bounds[i] = upper_fixed[j]
    for i in range(current_best_partition[num_intervals - 2] , num_var):
        upper_bounds[i] = upper_var[i]

    #Calculate breakpoints
    lower_breakpoints = [lower_bounds[i] + b[i] for i in range(0,num_var)]
    upper_breakpoints = [upper_bounds[i] + b[i] for i in range(0,num_var)]

    Final_solution = [min(upper_bounds[i],max(lower_bounds[i],current_best_mult - b[i])) for i in range(0,num_var)]
    return Final_solution