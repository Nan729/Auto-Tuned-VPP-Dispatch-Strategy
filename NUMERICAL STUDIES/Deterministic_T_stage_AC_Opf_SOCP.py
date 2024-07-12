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

def add_global_variables(Buses, Generators, Branches, thermal_lim, stage_num, wind_series):
    Pmin=9
    Pmax=8
    Qmin=4
    Qmax=3
    p_g=cp.Variable((len(Generators), stage_num))
    q_g=cp.Variable((len(Generators), stage_num))
    constraints=[]
    for t in range(stage_num):
#         Generators[5:7,Pmax] = wind_series[:,t]
#         Generators[5:7,Pmin] = wind_series[:,t]
#         constraints+=[p_g[:,t]>=Generators[:,Pmin], p_g[:,t]<=Generators[:,Pmax]]
# #         print(Generators[5:7,Pmax])
# #         print(Generators[5:7,Pmin])
        constraints+=[p_g[0:5,t]>=Generators[0:5,Pmin], p_g[0:5,t]<=Generators[0:5,Pmax]]
        constraints+=[p_g[5:7,t]>=Generators[5:7,Pmin], p_g[5:7,t]<=wind_series[:,t]]
        constraints+=[p_g[7:17,t]>=Generators[7:17,Pmin], p_g[7:17,t]<=Generators[7:17,Pmax]]
        constraints+=[q_g[:,t]>=Generators[:,Qmin], q_g[:,t]<=Generators[:,Qmax]]
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

    SR=cp.Variable((2*(len(Branches)),stage_num))
    SI=cp.Variable((2*(len(Branches)),stage_num))
    len_b=len(Branches)
    # for t in range(stage_num):
    #     for l in range(len_b):
    #         constraints+=[SR[l,t]<=thermal_lim[l]]
    #         constraints+=[SR[l,t]>=-thermal_lim[l]]
    #         constraints+=[SI[l,t]<=thermal_lim[l]]
    #         constraints+=[SI[l,t]>=-thermal_lim[l]]
    #         constraints+=[SR[l+len_b,t]<=thermal_lim[l]]
    #         constraints+=[SR[l+len_b,t]>=-thermal_lim[l]]
    #         constraints+=[SI[l+len_b,t]<=thermal_lim[l]]
    #         constraints+=[SI[l+len_b,t]>=-thermal_lim[l]]    
    #     #print(len(constraints))
    #         # @variable(model, Generators[i].Pmin/100.0 <= p_g[i=1:length(Generators)] <= Generators[i].Pmax/100.0, start = arr2[i])
    #         # @variable(model, Generators[i].Qmin/100.0 <= q_g[i=1:length(Generators)] <= Generators[i].Qmax/100.0, start = arr2[i])
    #         # SR = @variable(model,[(i,j,l) in [(i,j,l) for i = 1:length(Buses) for (j,l) in zip(δ(i,Branches,Buses)...)]], base_name = "SR",lower_bound = -thermal_lim[l], upper_bound = thermal_lim[l], start = arr3[l])
    #         # SI = @variable(model,[(i,j,l) in [(i,j,l) for i = 1:length(Buses) for (j,l) in zip(δ(i,Branches,Buses)...)]], base_name = "SI",lower_bound = -thermal_lim[l], upper_bound = thermal_lim[l], start = arr3[l])
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

######## power balance constraints: bus i
def constraint_power_balance(Buses, Generators, Branches, sqrd_volt, i, p_g, q_g, SR, SI, R_0):
    # @constraint(model, sum(p_g[i] for i in G(i,Generators,Buses)) - Buses[i].Pd/100.0 + Buses[i].Gs/100.0*sqrd_volt == sum(SR[(i,j,l)] for (j,l) in zip(δ(i,Branches,Buses)...)))
    # @constraint(model, sum(q_g[i] for i in G(i,Generators,Buses)) - Buses[i].Qd/100.0 + Buses[i].Bs/100.0*sqrd_volt == sum(SI[(i,j,l)] for (j,l) in zip(δ(i,Branches,Buses)...)))
    Pd=2
    Qd=3
    Gs=4
    Bs=5
    # constraints=[cp.sum([p_g[k] for k in G(i,Generators,Buses)]) + Buses[i,Gs]*sqrd_volt == cp.sum([SR[l] for (j,l) in zip(*line_connected(i,Branches,Buses))])+Buses[i,Pd]]
    if i==0:
        constraints=[cp.sum([p_g[k] for k in G(i,Generators,Buses)]) + Buses[i,Gs]*sqrd_volt == cp.sum([SR[l] for (j,l) in zip(*line_connected(i,Branches,Buses))])+Buses[i,Pd]+R_0]
    else:
        constraints=[cp.sum([p_g[k] for k in G(i,Generators,Buses)]) + Buses[i,Gs]*sqrd_volt == cp.sum([SR[l] for (j,l) in zip(*line_connected(i,Branches,Buses))])+Buses[i,Pd]]
    constraints+=[cp.sum([q_g[k] for k in G(i,Generators,Buses)])-Buses[i,Qd] + Buses[i,Bs]*sqrd_volt == cp.sum([SI[l] for (j,l) in zip(*line_connected(i,Branches,Buses))])]
    return constraints
##################################################################################################################3


def voltage_bounds_sqrd(Buses, sqrd_volt):
    Vmax=11
    Vmin=12
    constraints=[sqrd_volt <= Buses[:,Vmax]**2, sqrd_volt >= Buses[:,Vmin]**2]
    # Buses[i].Vmin^2 <= sqrd_volt <= Buses[i].Vmax^2)
    return constraints

# line constraints for i,j,l
def constraint_line_balance(YR, YI, B, sqrd_volt_i, sqrd_volt_j, sum_product_voltages,  diff_product_voltages, l, SR, SI, len_b):
    #YR[l], YI[l], B[l],sqrd_volt[i], sqrd_volt[j], sum_product_voltages,  diff_product_voltages,  l
    # @constraint(model, SR[(i,j,l)] == YR*sqrd_volt_i + (-YR)*sum_product_voltages + (-YI)*(diff_product_voltages))
            
    # @constraint(model, SI[(i,j,l)] == -(YI+B/2)*sqrd_volt_i - (-YI)*sum_product_voltages + (-YR)*diff_product_voltages)

    # @constraint(model, SR[(j,i,l)] == YR*sqrd_volt_j + (-YR)*sum_product_voltages + (-YI)*(-(diff_product_voltages)))

    # @constraint(model, SI[(j,i,l)] == -(YI+B/2)*sqrd_volt_j - (-YI)*sum_product_voltages + (-YR)*(-(diff_product_voltages)))
    constraints= [SR[l] == YR*sqrd_volt_i + (-YR)*sum_product_voltages + (-YI)*(diff_product_voltages)]
    constraints+= [SI[l] == -(YI+B/2)*sqrd_volt_i - (-YI)*sum_product_voltages + (-YR)*diff_product_voltages]
    constraints+= [SR[l+len_b] == YR*sqrd_volt_j + (-YR)*sum_product_voltages + (-YI)*(-(diff_product_voltages)) ]
    constraints+= [SI[l+len_b] == -(YI+B/2)*sqrd_volt_j - (-YI)*sum_product_voltages + (-YR)*(-(diff_product_voltages))]
    return constraints

def line_thermal_bounds_sqrd(thermal_lim, SR, SI,len_b):    
    # @constraint(model, SI[(i,j,l)]^2+SR[(i,j,l)]^2 <= thermal_lim[l]^2)
    # @constraint(model, SI[(j,i,l)]^2+SR[(j,i,l)]^2 <= thermal_lim[l]^2)
    thermal_cat = np.concatenate([thermal_lim, thermal_lim])
    constraints = [ cp.SOC(thermal_cat[i], cp.hstack([SI[i], SR[i]]) ) for i in range(2*len_b)]
    return constraints

def constraint_phase_angle_diff(ang_max, ang_min, sum_product_voltages,  diff_product_voltages, l):
    constraints = [diff_product_voltages <= np.tan(ang_max[l])*sum_product_voltages]
    constraints+= [diff_product_voltages >= np.tan(ang_min[l])*sum_product_voltages]
    # ## it is wrong 
    # constraints = [np.arctan(diff_product_voltages/sum_product_voltages) <= ang_max[l]]
    # constraints += [np.arctan(diff_product_voltages/sum_product_voltages) >= ang_min[l]]
    return constraints

def add_objective(GeneratorCosts, p_g, R_0, penalty_info, setpoint, stage_num):
    c2=4
    c1=5
    c0=6
    obj=0
    penalty_price = penalty_info[0:2]
    ########### no use: setpoint = penalty_info[3]
    allowed_error = penalty_info[2]*np.ones((1, stage_num))
    deviation = cp.Variable((2,stage_num))
    constraints = [deviation[[0],:]== R_0-setpoint-allowed_error]
    constraints += [deviation[[1],:]==-R_0+setpoint-allowed_error]
    for t in range(stage_num):
#         obj = obj+ cp.sum([GeneratorCosts[k,c2]*p_g[k,t]**2+GeneratorCosts[k,c1]*p_g[k,t]+GeneratorCosts[k,c0] for k in range(len(GeneratorCosts))])
        # obj = obj+ cp.sum([GeneratorCosts[k,c2]*p_g[k,t]**2+GeneratorCosts[k,c1]*p_g[k,t]+GeneratorCosts[k,c0] for k in range(5)])
        # obj = obj+ cp.sum([GeneratorCosts[k,c1]*p_g[k,t]+GeneratorCosts[k,c0] for k in range(5,len(GeneratorCosts))])
        const_v = np.sum(GeneratorCosts[:,c0])
        square_v = GeneratorCosts[:,c2] @ cp.square(p_g[:,t])
        affine_v = GeneratorCosts[:,c1] @ p_g[:,t]
        obj = obj+ square_v + affine_v + const_v
        ### penalty
        penalty=penalty_price[0]*cp.maximum(deviation[0,t],0)+penalty_price[1]*cp.maximum(deviation[1,t],0)
        obj=obj + penalty
    return obj, constraints

def SOCP_relaxation(Buses,Branches, W_t, sqrd_volt_t):
    # W=cp.Variable((2*len(Buses),2*len(Buses)))
    s = lambda x: x+len(Buses)
    constraints=[]
    for i in range(len(Buses)):
        for (j,l) in zip(*From_nodes(i,Branches,Buses)):
            left_array=cp.bmat([[2*(W_t[i,j]+W_t[s(i),s(j)])], [W_t[i,i]+W_t[s(i),s(i)]-W_t[j,j]-W_t[s(j),s(j)]]])
            constraints += [cp.SOC( W_t[i,i]+W_t[s(i),s(i)]+W_t[j,j]+W_t[s(j),s(j)], left_array )]
            #constraints += [cp.SOC( W[i,i]+W[s(i),s(i)]+W[j,j]+W[s(j),s(j)], 2*(W[i,j]+W[s(i),s(j)]), W[i,i]+W[s(i),s(i)]-W[j,j]-W[s(j),s(j)] )]
            #@constraint(model, [W[i,i]+W[s(i),s(i)]+W[j,j]+W[s(j),s(j)], 2*(W[i,j]+W[s(i),s(j)]), W[i,i]+W[s(i),s(i)]-W[j,j]-W[s(j),s(j)]] in SecondOrderCone())
        #print(constraints)
        # constraints += [sqrd_volt_t[i] == W_t[i,i]+W_t[s(i),s(i)]]
    eye_mat = np.eye((2*len(Buses)))
    right_mat = np.ones((2*len(Buses),1))
    left_mat = np.zeros((len(Buses),2*len(Buses)))
    for i in range(len(Buses)):
        left_mat[i,i]=1
        left_mat[i,s(i)]=1
    constraints += [sqrd_volt_t == left_mat @ (cp.multiply(eye_mat, W_t)) @ right_mat[:,0]]
    return constraints

def Conduct_Deterministic_T_stage_OPF(ppc, stage_num, penalty_info, setpoint, SOC_initial, demand_series, wind_series):
    # penalty-related
    R_0 = cp.Variable((1,stage_num))
    Buses= np.array(ppc["bus"][0,0])
    Generators = np.array(ppc["gen"][0,0])
    Branches = np.array(ppc["branch"][0,0])
    GeneratorCosts = np.array(ppc["gencost"][0,0])
    YR,YI,TR,TI,B,thermal_lim = add_constants(Branches)
    p_g, q_g, SR, SI, constraints = add_global_variables(Buses, Generators, Branches, thermal_lim, stage_num, wind_series)
#     print(len(constraints))
    obj, constraints_new = add_objective(GeneratorCosts, p_g, R_0, penalty_info, setpoint, stage_num)
    constraints+=constraints_new
#     print(len(constraints))
    W=cp.Variable((2*len(Buses)*stage_num,2*len(Buses)))
    sqrd_volt=cp.Variable((stage_num, len(Buses)))
    for t in range(stage_num):
        constraints_new  = SOCP_relaxation(Buses, Branches, W[t*2*len(Buses):(t+1)*2*len(Buses),:], sqrd_volt[t,:])
        constraints+=constraints_new
#     print(len(constraints))
    
    Pd=2
    L_bus = Gen_L_bus(Branches,Buses)
    G_bus = Gen_G_bus(Generators,Buses)
    #########################################################################################
    for i in range(len(Buses)):
        for t in range(stage_num):
            Buses[:,Pd]=demand_series[:,t]
            constraints_new=constraint_power_balance(Buses, Generators, Branches, sqrd_volt[t,i], i, p_g[:,t], q_g[:,t], SR[:,t], SI[:,t], R_0[:,t])
            constraints+=constraints_new
    ##############################################################################################

    
    for t in range(stage_num):
        constraints_new=voltage_bounds_sqrd(Buses, sqrd_volt[t,:])
        constraints+=constraints_new   
#     print(len(constraints))
    
    s = lambda x: x+len(Buses)
    for i in range(len(Buses)):
        for (j,l) in zip(*From_nodes(i,Branches,Buses)):
            for t in range(stage_num):
                W_t=W[t*2*len(Buses):(t+1)*2*len(Buses),:]
                sum_product_voltages  = W_t[i,j]+W_t[s(i),s(j)]
                ## to be changed: it is wrong
                # diff_product_voltages = W[s(j),i]-W[s(i),j]
                diff_product_voltages=W_t[s(i),j]-W_t[i,s(j)]
                len_b=len(Branches)
                constraints_new=constraint_line_balance(YR[l], YI[l], B[l],sqrd_volt[t,i], sqrd_volt[t,j], sum_product_voltages,  diff_product_voltages,  l, SR[:,t], SI[:,t], len_b)
                constraints+=constraints_new  
                # # Try remove this
                # constraints_new=constraint_phase_angle_diff(ang_max, ang_min, sum_product_voltages, diff_product_voltages, l)
                # constraints+=constraints_new  
    for t in range(stage_num):
        constraints_new=line_thermal_bounds_sqrd(thermal_lim, SR[:,t], SI[:,t],len_b)
        constraints+=constraints_new 
    BB_start=len(constraints)
#     print(len(constraints))
    
    SOC=cp.Variable((4, stage_num))
    BA_cap=1.0

    for t in range(stage_num-1):
        ### add ramping
        ramp_rate=0.7
        Pmax=8
        constraints_new =[ p_g[0:5,t+1]-p_g[0:5,t] <= ramp_rate* Generators[0:5,Pmax]]
        constraints+=constraints_new 
        constraints_new =[ p_g[0:5,t]-p_g[0:5,t+1] <= ramp_rate* Generators[0:5,Pmax]]
        constraints+=constraints_new

    for t in range(stage_num):
        ## ADD SOC 
        if t==0:
            #print(SOC_initial)
            constraints_new= [SOC[:,t] == SOC_initial-p_g[13:17,t]]
        else:
            constraints_new= [SOC[:,t] == SOC[:,t-1]-p_g[13:17,t]]
        constraints+=constraints_new
        for k in range(4):
            constraints+=[SOC[k,t]>=0]
            constraints+=[SOC[k,t]<=BA_cap]

    prob = cp.Problem(cp.Minimize(obj), constraints)
    prob.solve(verbose=False,solver=cp.MOSEK,mosek_params={mosek.dparam.optimizer_max_time: 100.0, mosek.iparam.intpnt_solve_form: mosek.solveform.dual})

    return prob, BB_start, constraints, p_g, R_0