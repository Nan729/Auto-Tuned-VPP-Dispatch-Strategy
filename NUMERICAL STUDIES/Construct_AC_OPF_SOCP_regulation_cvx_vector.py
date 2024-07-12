import cvxpy as cp
import numpy as np
import mosek

# graph function 1 
def bus_id(Buses):
    count = 0
    dict_buses = {}
    for bus in Buses:
        dict_buses[bus[0]] = count
        count += 1
    return dict_buses

# graph function 2: receive a bus i as attribute and returns all lines connected to it.
def line_connected(i,Branches,Buses):
    tbuses = []
    dict_buses = bus_id(Buses)
    count = 0
    ls = []
    len_b=len(Branches)
    for branch in Branches:
        if dict_buses[branch[0]] == i:
            #push!(tbuses,dict_buses[branch.tbus])
            tbuses.append(dict_buses[branch[1]])
            #push!(ls,count)
            ls.append(count)
        if dict_buses[branch[1]] == i:
            #push!(tbuses,dict_buses[branch.fbus])
            tbuses.append(dict_buses[branch[0]])
            #push!(ls,count)
            ls.append(count+len_b)
        count += 1
    return (tbuses,ls)

def Gen_L_bus(Branches,Buses):
    L_bus = np.zeros((len(Buses),2*len(Branches)))
    for i in range(len(Buses)):
        tbuses = []
        dict_buses = bus_id(Buses)
        count = 0
        ls = []
        len_b=len(Branches)
        for branch in Branches:
            if dict_buses[branch[0]] == i:
                #push!(ls,count)
                # ls.append(count)
                L_bus[i,count] = 1
            if dict_buses[branch[1]] == i:
                #push!(ls,count)
                # ls.append(count+len_b)
                L_bus[i,count+len_b] = 1
            count += 1
    return L_bus

def Gen_sqrd_from_to(Branches,Buses):
    sqrd_from = np.zeros((2*len(Branches),len(Buses)))
    sqrd_to = np.zeros((2*len(Branches),len(Buses)))
    for i in range(len(Buses)):
        dict_buses = bus_id(Buses)
        count = 0
        ls = []
        len_b=len(Branches)
        for branch in Branches:
            if dict_buses[branch[0]] == i:
    #             tbuses.append(dict_buses[branch[1]])
    #             ls.append(count)
                  sqrd_to[count,dict_buses[branch[1]]] = 1
                  sqrd_from[count, i] = 1 
            if dict_buses[branch[1]] == i:
    #             tbuses.append(dict_buses[branch[0]])
    #             ls.append(count+len_b)
                  sqrd_to[count+len_b,dict_buses[branch[0]]] = 1
                  sqrd_from[count+len_b, i] = 1 
            count += 1
    return sqrd_from, sqrd_to

def From_nodes(i,Branches,Buses):
    tbuses = []
    ls = []
    count = 0
    dict_buses = bus_id(Buses)
    for branch in Branches:
        if dict_buses[branch[0]] == i:
            #push!(tbuses,dict_buses[branch.tbus])
            tbuses.append(dict_buses[branch[1]])
            #push!(ls,count)
            ls.append(count)
        count += 1
    return (tbuses,ls)

def To_nodes(i,Branches,Buses):
    tbuses = []
    ls = []
    count = 0
    dict_buses = bus_id(Buses)
    len_b=len(Branches)
    for branch in Branches:
        if dict_buses[branch[1]] == i:
            #push!(tbuses,dict_buses[branch.fbus])
            tbuses.append(dict_buses[branch[0]])
            #push!(ls,count)
            ls.append(count+len_b)
        count += 1
    return (tbuses,ls)

# 2*len(branch) must find a way to sort them & call their index instead of tuples (i,j,l) in Julia
# with their original order
# i from bus
# j to bus
def SR_matching(i,j,dict_buses,Branches):
    count = 0
    l_index = []
    pos_flag= []
    for branch in Branches:
        if dict_buses[branch[0]] == i and dict_buses[branch[0]] == j:
            l_index=count
            pos_flag=True
        if dict_buses[branch[0]] == j and dict_buses[branch[0]] == i:
            l_index=count+len(Branches)
            pos_flag=False
    count += 1
    return l_index,pos_flag

def G(i,Generators,Buses):
    G_i = []
    count = 0
    dict_buses = bus_id(Buses)
    for gen in Generators:
        if dict_buses[gen[0]] == i:
            #push!(G_i,count)
            G_i.append(count)
        count += 1
    return G_i

def Gen_G_bus(Generators,Buses):
    G_bus = np.zeros((len(Buses), len(Generators)))
    for i in range(len(Buses)):
        count = 0 ## count from 0, represent column
        dict_buses = bus_id(Buses)
        for gen in Generators:
            if dict_buses[gen[0]] == i:
                G_bus[i,count] = 1
                #G_i.append(count)
            count += 1
    return G_bus

def add_constants(Branches):
    YR = np.zeros(len(Branches))
    YI = np.zeros(len(Branches))
    TR = np.zeros(len(Branches))
    TI = np.zeros(len(Branches))
    B = np.zeros(len(Branches))
    thermal_lim = np.zeros(len(Branches))
    # ang_min = np.zeros(len(Branches))
    # ang_max = np.zeros(len(Branches))
    r=2
    x=3
    b=4
    ratio=8
    angle=9
    rateA=5
    angmin=11
    angmax=12
    for l in range(len(Branches)):
        YR[l] = Branches[l,r]/(Branches[l,r]**2+Branches[l,x]**2)
        YI[l] = -Branches[l,x]/(Branches[l,r]**2+Branches[l,x]**2)
        TR[l] = Branches[l,ratio]*np.cos(Branches[l,angle])
        TI[l] = Branches[l,ratio]*np.sin(Branches[l,angle])
        B[l] = Branches[l,b]
        thermal_lim[l] = Branches[l,rateA]
        # ang_min[l] = Branches[l,angmin]*np.pi/180
        # ang_max[l] = Branches[l,angmax]*np.pi/180
    return YR,YI,TR,TI,B,thermal_lim

def add_global_variables(Buses, Generators, Branches, thermal_lim, gen_max, gen_min):

    Qmin=4
    Qmax=3
    p_g=cp.Variable((len(Generators)))
    constraints=[p_g>=gen_min, p_g<=gen_max]
    q_g=cp.Variable((len(Generators)))
    constraints+=[q_g>=Generators[:,Qmin], q_g<=Generators[:,Qmax]]
    # ### generator capability curve
    # # P1,Q1min, P2, Q2min
    # P1=10
    # Q1min=12
    # P2=11
    # Q2min=14
    # Q1max=13
    # Q2max=15
    # a=Generators[:,P1]/100.0
    # b=Generators[:,Q1min]/100.0
    # c=Generators[:,P2]/100.0
    # d=Generators[:,Q2min]/100.0
    # slope=np.divide((d-b),(c-a))
    # xx1=np.multiply(b,c)
    # xx2=np.multiply(a,d)
    # inter=np.divide((xx1-xx2),(c-a))
    # constraints+=[q_g>=cp.multiply(slope, p_g)+inter]
    # # P1,Q1max, P2, Q2max
    # a=Generators[:,P1]/100.0
    # b=Generators[:,Q1max]/100.0
    # c=Generators[:,P2]/100.0
    # d=Generators[:,Q2max]/100.0
    # slope=np.divide((d-b),(c-a))
    # xx1=np.multiply(b,c)
    # xx2=np.multiply(a,d)
    # inter=np.divide((xx1-xx2),(c-a))
    # constraints+=[q_g<=cp.multiply(slope, p_g)+inter]

    SR=cp.Variable(2*(len(Branches)))
    SI=cp.Variable(2*(len(Branches)))
#     len_b=len(Branches)
#     thermal_cat = np.concatenate([thermal_lim, thermal_lim])
# #         constraints+=[SR[l]<=thermal_lim[l]]
# #         constraints+=[SR[l]>=-thermal_lim[l]]
# #         constraints+=[SI[l]<=thermal_lim[l]]
# #         constraints+=[SI[l]>=-thermal_lim[l]]
#     constraints += [ SR <= thermal_cat ]
#     constraints += [ SR >= -thermal_cat ]
#     constraints += [ SI <= thermal_cat ]
#     constraints += [ SI >= -thermal_cat ]
#         constraints+=[SR[l+len_b]<=thermal_lim[l]]
#         constraints+=[SR[l+len_b]>=-thermal_lim[l]]
#         constraints+=[SI[l+len_b]<=thermal_lim[l]]
#         constraints+=[SI[l+len_b]>=-thermal_lim[l]]    
    return p_g, q_g, SR, SI, constraints

def add_voltage_variables(Buses):
    Vmax=11
    Vmin=12
    VR=cp.Variable((len(Buses)))
    VI=cp.Variable((len(Buses)))
    constraints=[VR>= Buses[:,-Vmax], VR<= Buses[:,Vmax]]
    constraints+=[VI>= Buses[:,-Vmax], VI<= Buses[:,Vmax]]
    constraints+=[VI[0]==0]
        # arr1 = ones(length(Buses))
        # @variable(model, - Buses[i].Vmax <= VR[i=1:length(Buses)] <= Buses[i].Vmax , start = arr1[i])
        # @variable(model, - Buses[i].Vmax <= VI[i=1:length(Buses)] <= Buses[i].Vmax , start = arr1[i])
    return VR, VI, constraints

#power balance constraints: bus i
# def constraint_power_balance(Buses, Generators, Branches, sqrd_volt, i, p_g, q_g, SR, SI, R_0):
#     # @constraint(model, sum(p_g[i] for i in G(i,Generators,Buses)) - Buses[i].Pd/100.0 + Buses[i].Gs/100.0*sqrd_volt == sum(SR[(i,j,l)] for (j,l) in zip(δ(i,Branches,Buses)...)))
#     # @constraint(model, sum(q_g[i] for i in G(i,Generators,Buses)) - Buses[i].Qd/100.0 + Buses[i].Bs/100.0*sqrd_volt == sum(SI[(i,j,l)] for (j,l) in zip(δ(i,Branches,Buses)...)))
#     Pd=2
#     Qd=3
#     Gs=4
#     Bs=5
#     if i==0:
#         constraints=[cp.sum([p_g[k] for k in G(i,Generators,Buses)]) + Buses[i,Gs]*sqrd_volt == cp.sum([SR[l] for (j,l) in zip(*line_connected(i,Branches,Buses))])+Buses[i,Pd]+R_0]
#     else:
#         constraints=[cp.sum([p_g[k] for k in G(i,Generators,Buses)]) + Buses[i,Gs]*sqrd_volt == cp.sum([SR[l] for (j,l) in zip(*line_connected(i,Branches,Buses))])+Buses[i,Pd]]
#     constraints+=[cp.sum([q_g[k] for k in G(i,Generators,Buses)])-Buses[i,Qd] + Buses[i,Bs]*sqrd_volt == cp.sum([SI[l] for (j,l) in zip(*line_connected(i,Branches,Buses))])]
#     return constraints
#######################################################################################################
def constraint_power_balance(Buses, Generators, Branches, sqrd_volt, p_g, q_g, SR, SI, R_all, L_bus, G_bus):
    # @constraint(model, sum(p_g[i] for i in G(i,Generators,Buses)) - Buses[i].Pd/100.0 + Buses[i].Gs/100.0*sqrd_volt == sum(SR[(i,j,l)] for (j,l) in zip(δ(i,Branches,Buses)...)))
    # @constraint(model, sum(q_g[i] for i in G(i,Generators,Buses)) - Buses[i].Qd/100.0 + Buses[i].Bs/100.0*sqrd_volt == sum(SI[(i,j,l)] for (j,l) in zip(δ(i,Branches,Buses)...)))
    Pd=2
    Qd=3
    Gs=4
    Bs=5

    #print(sqrd_volt.shape)
    
    constraints = [  G_bus @ p_g +  cp.multiply(Buses[:,Gs],sqrd_volt) == L_bus @ SR + Buses[:,Pd] +  R_all]
    constraints += [  G_bus @ q_g + cp.multiply(Buses[:,Bs],sqrd_volt) == L_bus @ SI + Buses[:,Qd]]
    
#     if i==0:
#         constraints=[cp.sum([p_g[k] for k in G(i,Generators,Buses)]) + Buses[i,Gs]*sqrd_volt == cp.sum([SR[l] for (j,l) in zip(*line_connected(i,Branches,Buses))])+Buses[i,Pd]+R_0]
#     else:
#         constraints=[cp.sum([p_g[k] for k in G(i,Generators,Buses)]) + Buses[i,Gs]*sqrd_volt == cp.sum([SR[l] for (j,l) in zip(*line_connected(i,Branches,Buses))])+Buses[i,Pd]]
#     constraints+=[cp.sum([q_g[k] for k in G(i,Generators,Buses)])-Buses[i,Qd] + Buses[i,Bs]*sqrd_volt == cp.sum([SI[l] for (j,l) in zip(*line_connected(i,Branches,Buses))])]
    return constraints


# def voltage_bounds_sqrd(Buses, sqrd_volt, i):
#     Vmax=11
#     Vmin=12
#     constraints=[sqrd_volt <= Buses[i,Vmax]**2, sqrd_volt >= Buses[i,Vmin]**2]
#     # Buses[i].Vmin^2 <= sqrd_volt <= Buses[i].Vmax^2)
#     return constraints
def voltage_bounds_sqrd(Buses, sqrd_volt):
    Vmax=11
    Vmin=12
    constraints=[sqrd_volt <= np.square(Buses[:,Vmax]), sqrd_volt >= np.square(Buses[:,Vmin])]
    return constraints

# line constraints for i,j,l
# def constraint_line_balance(YR, YI, B, sqrd_volt_i, sqrd_volt_j, sum_product_voltages,  diff_product_voltages, l, SR, SI, len_b):
#     #YR[l], YI[l], B[l],sqrd_volt[i], sqrd_volt[j], sum_product_voltages,  diff_product_voltages,  l
#     # @constraint(model, SR[(i,j,l)] == YR*sqrd_volt_i + (-YR)*sum_product_voltages + (-YI)*(diff_product_voltages))
            
#     # @constraint(model, SI[(i,j,l)] == -(YI+B/2)*sqrd_volt_i - (-YI)*sum_product_voltages + (-YR)*diff_product_voltages)

#     # @constraint(model, SR[(j,i,l)] == YR*sqrd_volt_j + (-YR)*sum_product_voltages + (-YI)*(-(diff_product_voltages)))

#     # @constraint(model, SI[(j,i,l)] == -(YI+B/2)*sqrd_volt_j - (-YI)*sum_product_voltages + (-YR)*(-(diff_product_voltages)))
#     constraints= [SR[l] == YR*sqrd_volt_i + (-YR)*sum_product_voltages + (-YI)*(diff_product_voltages)]
#     constraints+= [SI[l] == -(YI+B/2)*sqrd_volt_i - (-YI)*sum_product_voltages + (-YR)*diff_product_voltages]
#     constraints+= [SR[l+len_b] == YR*sqrd_volt_j + (-YR)*sum_product_voltages + (-YI)*(-(diff_product_voltages)) ]
#     constraints+= [SI[l+len_b] == -(YI+B/2)*sqrd_volt_j - (-YI)*sum_product_voltages + (-YR)*(-(diff_product_voltages))]
#     return constraints
def constraint_line_balance(YR, YI, B, sqrd_volt, sum_product_voltages,  diff_product_voltages, SR, SI, len_b, sqrd_from):
    constraints= [SR == cp.multiply(YR, sqrd_from@ sqrd_volt) + cp.multiply((-YR), sum_product_voltages) + cp.multiply((-YI),(diff_product_voltages))]
    
    constraints+= [SI == cp.multiply((-(YI+B/2)), sqrd_from@sqrd_volt) - cp.multiply((-YI),sum_product_voltages) + cp.multiply((-YR),diff_product_voltages) ]
    return constraints

##########################################################################################################

def line_thermal_bounds_sqrd(thermal_lim, SR, SI,len_b):    
    # @constraint(model, SI[(i,j,l)]^2+SR[(i,j,l)]^2 <= thermal_lim[l]^2)
    # @constraint(model, SI[(j,i,l)]^2+SR[(j,i,l)]^2 <= thermal_lim[l]^2)
#     constraints=[SI[l]**2+SR[l]**2 <= thermal_lim[l]**2]
#     constraints+=[SI[l+len_b]**2+SR[l+len_b]**2 <= thermal_lim[l]**2]
    thermal_cat = np.concatenate([thermal_lim, thermal_lim])
    # constraints = [ cp.square(SI)+ cp.square(SR) <= np.square(thermal_cat)]
    constraints = [ cp.SOC(thermal_cat[i], cp.hstack([SI[i], SR[i]]) ) for i in range(2*len_b)]
    return constraints


def add_objective(GeneratorCosts, p_g, R_0, penalty_info, SOC_index, setpoint, SOC_last, theta_sqrt, theta_linear):
    c2=4
    c1=5
    c0=6
    constraints = []
    const_v = np.sum(GeneratorCosts[:,c0])
    square_v = GeneratorCosts[:,c2] @ cp.square(p_g)
    affine_v = GeneratorCosts[:,c1] @ p_g
    obj = square_v + affine_v + const_v
#     obj = cp.sum([GeneratorCosts[k,c2]*p_g[k]**2+GeneratorCosts[k,c1]*p_g[k]+GeneratorCosts[k,c0] for k in range(len(GeneratorCosts))])
    # add penalty-related
    penalty_price = penalty_info[0:2]
    allowed_error = penalty_info[2]
#     deviation = cp.Variable((2,1))
#     constraints += [deviation[0]== R_0-setpoint-allowed_error]
#     constraints += [deviation[1]== -R_0+setpoint-allowed_error]
#     penalty=penalty_price[0]*cp.maximum(deviation[0],0)+penalty_price[1]*cp.maximum(deviation[1],0)
    vio_rate = cp.hstack([cp.maximum((R_0-setpoint-allowed_error),0), cp.maximum((-R_0+setpoint-allowed_error),0)])
    penalty = penalty_price @ vio_rate
#     penalty = penalty_price[0]*cp.maximum((R_0-setpoint-allowed_error),0)+penalty_price[1]*cp.maximum((-R_0+setpoint-allowed_error),0)
    obj=obj + penalty

    # add cost-to-go in the future
    index_start=int(SOC_index[0,0])
    ############ no use SOC_last = SOC_index[:,3]
    ############ theta = SOC_index[:,1:3]
#     SOC_new=cp.Variable((len(SOC_index),1))
#     for k in range(len(SOC_index)):
#         constraints += [SOC_new[k]==SOC_last[k]-p_g[k+index_start]]

    SOC_new=cp.Variable((len(SOC_index)))
    constraints += [SOC_new == SOC_last - p_g[index_start:]]
    value_func = cp.sum_squares(theta_sqrt @ SOC_new- theta_linear.T[:,0])
    obj=obj+value_func
    
#     value=cp.Variable((1))
#     constraints += [cp.SOC(value, (theta_sqrt @(SOC_last - p_g[index_start:]) - theta_linear.T[:,0]))]
#     obj=obj+ cp.square(value)
    
    return obj, constraints

def SOCP_relaxation(Buses,Branches):
    W=cp.Variable((2*len(Buses),2*len(Buses)))
    s = lambda x: x+len(Buses)
    constraints=[]
    for i in range(len(Buses)):
        for (j,l) in zip(*From_nodes(i,Branches,Buses)):
            left_array = cp.bmat([[2*(W[i,j]+W[s(i),s(j)])], [W[i,i]+W[s(i),s(i)]-W[j,j]-W[s(j),s(j)]]])
            constraints += [cp.SOC( W[i,i]+W[s(i),s(i)]+W[j,j]+W[s(j),s(j)], left_array )]
            #constraints += [cp.SOC( W[i,i]+W[s(i),s(i)]+W[j,j]+W[s(j),s(j)], 2*(W[i,j]+W[s(i),s(j)]), W[i,i]+W[s(i),s(i)]-W[j,j]-W[s(j),s(j)] )]
    #print(constraints)
    return W, constraints

def Construct_OPF_regulation_v1(ppc, penalty_info, SOC_index):
    # penalty-related
    R_0 = cp.Variable((1))
    Buses= np.array(ppc["bus"][0,0])
    Generators = np.array(ppc["gen"][0,0])
    Branches = np.array(ppc["branch"][0,0])
    GeneratorCosts = np.array(ppc["gencost"][0,0])

    gen_max= cp.Parameter((len(Generators)))
    gen_min= cp.Parameter((len(Generators)))

    YR,YI,TR,TI,B,thermal_lim = add_constants(Branches)
    p_g, q_g, SR, SI, constraints = add_global_variables(Buses, Generators, Branches, thermal_lim, gen_max, gen_min)
    print('add_global_variables: now constraints number' + str(len(constraints)))

    setpoint= cp.Parameter((1))
    theta_sqrt= cp.Parameter((len(SOC_index),len(SOC_index)))
    theta_linear=cp.Parameter((1,len(SOC_index)))
    SOC_last = cp.Parameter((len(SOC_index)))

    obj, constraints_new  = add_objective(GeneratorCosts, p_g, R_0, penalty_info, SOC_index, setpoint, SOC_last, theta_sqrt, theta_linear)
    constraints+=constraints_new
    print('obj: now constraints number' + str(len(constraints)))
    
    CON_num = len(constraints)
    W, constraints_new  = SOCP_relaxation(Buses, Branches)
    constraints+=constraints_new
    CON_num_2 = len(constraints)
    print('constraints number' + str(CON_num_2-CON_num))
    print('SOC: now constraints number' + str(len(constraints)))

    s = lambda x: x+len(Buses)
#     sqrd_volt = [W[i,i]+W[s(i),s(i)] for i in range(len(Buses))]
    eye_mat = np.eye((2*len(Buses)))
    right_mat = np.ones((2*len(Buses),1))
    left_mat = np.zeros((len(Buses),2*len(Buses)))
    for i in range(len(Buses)):
        left_mat[i,i]=1
        left_mat[i,s(i)]=1
    sqrd_volt = left_mat @ (cp.multiply(eye_mat, W)) @ right_mat[:,0]
    
    
    L_bus = Gen_L_bus(Branches,Buses)
    G_bus = Gen_G_bus(Generators,Buses)
    R_all = cp.hstack([R_0, np.zeros((len(Buses)-1))])

    BB_start=len(constraints)
#     for i in range(len(Buses)):
#         constraints_new=constraint_power_balance(Buses, Generators, Branches, sqrd_volt[i], i, p_g, q_g, SR, SI, R_0)
#         constraints+=constraints_new
#         constraints_new=voltage_bounds_sqrd(Buses, sqrd_volt[i], i)
#         constraints+=constraints_new 

    constraints_new=constraint_power_balance(Buses, Generators, Branches, sqrd_volt, p_g, q_g, SR, SI, R_all, L_bus, G_bus )
    constraints+=constraints_new
    print('constraint_power_balance: now constraints number' + str(len(constraints)))

    constraints_new=voltage_bounds_sqrd(Buses, sqrd_volt)
    constraints+=constraints_new  
    print('voltage_bounds_sqrd: now constraints number' + str(len(constraints)))
    
    
    sqrd_from, sqrd_to = Gen_sqrd_from_to(Branches,Buses)
    cat_piece = np.zeros((len(Branches),len(Buses)))
    row_from = sqrd_from[0:len(Branches),:]
    col_to = sqrd_to[0:len(Branches),:]
    row_choose = np.bmat([[row_from, cat_piece], [cat_piece, row_from]])
    col_choose = np.bmat([[col_to, cat_piece], [cat_piece, col_to]])
    eye_piece = np.eye((len(Branches)))
    left_cat = np.bmat([[eye_piece, eye_piece], [eye_piece, eye_piece]])
    sum_product_voltages_all = left_cat @ cp.multiply( col_choose, row_choose@ W )@ np.ones((2*len(Buses),1))[:,0]
    
    row_choose_diff = np.bmat([[cat_piece, row_from],[row_from, cat_piece]])
    left_cat_diff = np.bmat([[eye_piece, -eye_piece], [-eye_piece, eye_piece]])
    diff_product_voltages_all = left_cat_diff @ cp.multiply( col_choose, row_choose_diff@ W )@ np.ones((2*len(Buses),1))[:,0]
    
    len_b=len(Branches)
    YR_cat = np.concatenate([YR, YR])
    YI_cat = np.concatenate([YI, YI])
    B_cat = np.concatenate([B, B])
    constraints_new=constraint_line_balance(YR_cat, YI_cat, B_cat,sqrd_volt, sum_product_voltages_all,  diff_product_voltages_all, SR, SI, len_b, sqrd_from)
    constraints+=constraints_new  
    print('constraint_line_balance: now constraints number' + str(len(constraints)))
    
    
    
    
#     for i in range(len(Buses)):
#         for (j,l) in zip(*From_nodes(i,Branches,Buses)):
#             sum_product_voltages  = W[i,j]+W[s(i),s(j)]
#             ## to be changed: it is wrong
#             # diff_product_voltages = W[s(j),i]-W[s(i),j]
#             diff_product_voltages=W[s(i),j]-W[i,s(j)]
#             len_b=len(Branches)
#             constraints_new=constraint_line_balance(YR[l], YI[l], B[l],sqrd_volt[i], sqrd_volt[j], sum_product_voltages,  diff_product_voltages,  l, SR, SI, len_b)

            
    constraints_new=line_thermal_bounds_sqrd(thermal_lim, SR, SI,len_b)
    constraints+=constraints_new 
    
    print('all constraints number' + str(len(constraints)))

    prob = cp.Problem(cp.Minimize(obj), constraints)

    # Pmin=9
    # Pmax=8
    # gen_max.value = Generators[:,Pmax]
    # gen_min.value =Generators[:,Pmin]
    # setpoint.value = np.array([penalty_info[3]])
    # theta_sqrt.value = SOC_index[:,3:]
    # theta_linear.value= SOC_index[:,[2]].T
    # SOC_last.value = SOC_index[:,1]

    # prob.solve(verbose=True,solver=cp.MOSEK)

    return prob, constraints, R_0, p_g, gen_max, gen_min, setpoint, theta_sqrt, theta_linear, SOC_last