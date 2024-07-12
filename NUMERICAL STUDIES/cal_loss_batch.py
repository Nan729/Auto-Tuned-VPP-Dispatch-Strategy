import torch
import numpy as np
from Construct_AC_OPF_SOCP_regulation_cvx_vector import Construct_OPF_regulation_v1
from Construct_policy import Construct_policy
# from update_state_batch import update_PPC_batch
from update_state_batch import update_info_torch_batch
from cvxpylayers.torch import CvxpyLayer
from gen_random_series import gen_random_series
import copy
import time

def cal_loss_batch_stage(ppc, p_g_batch, R_0_batch, penalty_info, setpoint_batch):
    batch_size = p_g_batch.shape[0]
    GeneratorCosts = np.array(ppc["gencost"][0,0])
    c2=torch.from_numpy(GeneratorCosts[:,4])
    c1=torch.from_numpy(GeneratorCosts[:,5])
    c0=torch.from_numpy(GeneratorCosts[:,6])

    #cost =torch.sum(c2.float()*torch.square(p_g)+c1.float()*p_g+c0.float())
    cost_batch =torch.sum(torch.square(p_g_batch)*c2.float()+p_g_batch*c1.float()+c0.float(), axis=1)

    allowed_error = penalty_info[2]
    penalty_price = torch.from_numpy(np.array(penalty_info[0:2]))
    # penalty= penalty_price[0]*torch.maximum((R_0-setpoint-allowed_error),torch.Tensor([0]))+penalty_price[1]*torch.maximum(-R_0+setpoint-allowed_error,torch.Tensor([0]))
    penalty_batch= penalty_price[0]*torch.maximum((R_0_batch-setpoint_batch-allowed_error),torch.Tensor([0]))+penalty_price[1]*torch.maximum(-R_0_batch+setpoint_batch-allowed_error,torch.Tensor([0]))
    
    loss_batch = cost_batch.squeeze()+ penalty_batch.squeeze()
    loss = torch.sum(loss_batch)/batch_size
    return loss


def cal_loss_batch_all_variant(stage_num, penalty_info, ppc_origin, theta_sqrt_all, theta_linear_all, SOC_last_solve, storage_index, Initial_Pmax, demand_series, category, batch_size, seed):
    ppc=copy.deepcopy(ppc_origin)
    losses=0.0
    for t in range(stage_num): 
        ## demand change
        Pd=2
        Pmin=9
        Pmax=8
        Generators = np.array(ppc["gen"][0,0])
        wind_series_batch, setpoint_series_batch = gen_random_series(Generators, stage_num, category, batch_size, seed)
        ppc["bus"][0,0][:,Pd]=demand_series[:,t]
        ### ppc["gen"][0,0][5:7,Pmax]=wind_series[:,t]  ##TOCHANGE MAYBE
        # prob, constraints, R_0, p_g, gen_max, gen_min, setpoint, theta_sqrt, theta_linear, SOC_last=Construct_OPF_regulation_v1(ppc, penalty_info, storage_index)
        # policy = CvxpyLayer(prob,[gen_max, gen_min, setpoint, theta_sqrt, theta_linear, SOC_last], [p_g, R_0])
        policy = Construct_policy(ppc, penalty_info, storage_index)
        theta_size=len(storage_index)

        if t==(stage_num-1):
            theta_sqrt = torch.zeros((theta_size,theta_size),dtype=torch.float64)
            theta_linear = torch.zeros((1,theta_size),dtype=torch.float64)
        else:
            theta_sqrt = theta_sqrt_all[t]
            theta_linear = theta_linear_all[[t]]

        theta_sqrt_batch = theta_sqrt.repeat(batch_size, 1, 1)
        theta_linear_batch = theta_linear.repeat(batch_size, 1, 1)
        
        if t==0:
            [gen_max, gen_min, SOC_last]=map(torch.from_numpy,[ppc["gen"][0,0][:,Pmax], ppc["gen"][0,0][:,Pmin], SOC_last_solve.T])
            setpoint_batch = setpoint_series_batch[:,:,t]
            SOC_last_batch = SOC_last.squeeze().repeat(batch_size, 1)
            gen_max_batch = gen_max.repeat(batch_size, 1)
            gen_max_batch[:, 5:7] = wind_series_batch[:, :, t]
            gen_min_batch = gen_min.repeat(batch_size, 1)
        else:
            ### needs to be changed from tensor to ppc and future information
            setpoint_batch = setpoint_series_batch[:,:,t]
            gen_max_batch, gen_min_batch, SOC_last_batch= update_info_torch_batch(ppc, p_g_batch, SOC_last_batch, storage_index, Initial_Pmax, wind_series_batch[:, :, t])


        p_g_batch, R_0_batch= policy(gen_max_batch, gen_min_batch, setpoint_batch, theta_sqrt_batch, theta_linear_batch, SOC_last_batch, solver_args={'solve_method':'ECOS'})
        
        # print(policy)
        # print(sum(p.numel() for p in policy.parameters() if p.require_grad))
        # exit(1)
        # p_g, R_0= policy(gen_max, gen_min, setpoint, theta_sqrt, theta_linear, SOC_last)
        ### loss
        # print(p_g)
#         print(p_g_batch.shape)
#         # print(R_0)
#         print(R_0_batch.shape)
        # p_g_record=p_g
        # SOC_record=SOC_last
        loss= cal_loss_batch_stage(ppc, p_g_batch, R_0_batch, penalty_info, setpoint_batch)
        losses = losses+ loss
        # print(loss)


        ### needs to be changed from tensor to ppc and future information
        #P_variables=p_g.detach().numpy()

        ###### ????????? 
        #ppc,SOC_last_solve=update_PPC_batch(ppc,P_variables, SOC_last_solve, Initial_Pmax)

    return losses

# def cal_loss_all_variant_with_return(stage_num, penalty_info, ppc_origin, theta_sqrt_all, theta_linear_all, SOC_last_solve, storage_index, Initial_Pmax, demand_series, wind_series, setpoint_series):
#     ppc=copy.deepcopy(ppc_origin)
#     losses=0.0
#     p_record=np.zeros((17, stage_num))
#     r_record=np.zeros((stage_num,1))
#     for t in range(stage_num): 
#         ## demand change
#         Pd=2
#         Pmin=9
#         Pmax=8
#         ppc["bus"][0,0][:,Pd]=demand_series[:,t]
#         ppc["gen"][0,0][5:7,Pmax]=wind_series[:,t]
#         # prob, constraints, R_0, p_g, gen_max, gen_min, setpoint, theta_sqrt, theta_linear, SOC_last=Construct_OPF_regulation_v1(ppc, penalty_info, storage_index)
#         # policy = CvxpyLayer(prob,[gen_max, gen_min, setpoint, theta_sqrt, theta_linear, SOC_last], [p_g, R_0])
#         policy = Construct_policy(ppc, penalty_info, storage_index)
#         theta_size=len(storage_index)

#         if t==(stage_num-1):
#             theta_sqrt = torch.zeros((theta_size,theta_size),dtype=torch.float64)
#             theta_linear = torch.zeros((1,theta_size),dtype=torch.float64)
#         else:
#             theta_sqrt = theta_sqrt_all[t]
#             theta_linear = theta_linear_all[[t]]
#             # print(theta_sqrt.requires_grad)
#             # print(theta_linear.requires_grad)
        
#         if t==0:
#             [gen_max, gen_min, setpoint, SOC_last]=map(torch.from_numpy,[ppc["gen"][0,0][:,Pmax], ppc["gen"][0,0][:,Pmin], np.array([setpoint_series[0,t]]), SOC_last_solve.T])
#             SOC_last = SOC_last.squeeze()
#         else:
#             ### needs to be changed from tensor to ppc and future information
#             [setpoint]=map(torch.from_numpy,[np.array([setpoint_series[0,t]])])
#             gen_max, gen_min, SOC_last= update_info_torch(ppc, p_g, SOC_last, storage_index, Initial_Pmax)
#             # if (SOC_last.detach().numpy()==SOC_last_solve).all():
#             #     print('OK for SOC_last')
#             # if (gen_max.detach().numpy()==ppc["gen"][0,0][:,Pmax]).all():
#             #     print('OK for gen_max')
#             # if (gen_min.detach().numpy()==ppc["gen"][0,0][:,Pmin]).all():
#             #     print('OK for gen_min')
#             ###SOC_last= SOC_last - p_g[storage_index[:,0]]
#         # print(setpoint.shape)
#         # print(theta_sqrt.shape)
#         # print(theta_linear.shape)
#         # print(SOC_last.shape)
#         p_g, R_0= policy(gen_max, gen_min, setpoint, theta_sqrt, theta_linear, SOC_last, solver_args={'solve_method':'ECOS'})
#         p_record[:,t]=p_g.detach().numpy()
#         r_record[t]=R_0.detach().numpy()
#         # print(policy)
#         # print(sum(p.numel() for p in policy.parameters() if p.require_grad))
#         # exit(1)
#         # p_g, R_0= policy(gen_max, gen_min, setpoint, theta_sqrt, theta_linear, SOC_last)
#         ### loss
#         # print(p_g)
#         # print(p_g.shape)
#         # print(R_0)
#         # print(R_0.shape)
#         # p_g_record=p_g
#         # SOC_record=SOC_last
#         loss= cal_loss_stage(ppc, p_g, R_0, SOC_last, theta_sqrt, theta_linear, penalty_info, setpoint)
#         losses = losses+ loss
#         # print(loss)


#         ### needs to be changed from tensor to ppc and future information
#         P_variables=p_g.detach().numpy()

#         ppc,SOC_last_solve=update_PPC(ppc,P_variables, SOC_last_solve, Initial_Pmax)

#     return losses, p_record, r_record
def cal_loss_batch_all_variant_with_time(stage_num, penalty_info, ppc_origin, theta_sqrt_all, theta_linear_all, SOC_last_solve, storage_index, Initial_Pmax, demand_series, category, batch_size, seed):

    ppc=copy.deepcopy(ppc_origin)
    losses=0.0
    time_record = np.zeros((1, stage_num))
    for t in range(stage_num): 
        ## demand change
        time_start = time.time()
        Pd=2
        Pmin=9
        Pmax=8
        Generators = np.array(ppc["gen"][0,0])
        wind_series_batch, setpoint_series_batch = gen_random_series(Generators, stage_num, category, batch_size, seed)
        ppc["bus"][0,0][:,Pd]=demand_series[:,t]
        ### ppc["gen"][0,0][5:7,Pmax]=wind_series[:,t]  ##TOCHANGE MAYBE
        # prob, constraints, R_0, p_g, gen_max, gen_min, setpoint, theta_sqrt, theta_linear, SOC_last=Construct_OPF_regulation_v1(ppc, penalty_info, storage_index)
        # policy = CvxpyLayer(prob,[gen_max, gen_min, setpoint, theta_sqrt, theta_linear, SOC_last], [p_g, R_0])
        policy = Construct_policy(ppc, penalty_info, storage_index)
        theta_size=len(storage_index)

        if t==(stage_num-1):
            theta_sqrt = torch.zeros((theta_size,theta_size),dtype=torch.float64)
            theta_linear = torch.zeros((1,theta_size),dtype=torch.float64)
        else:
            theta_sqrt = theta_sqrt_all[t]
            theta_linear = theta_linear_all[[t]]

        theta_sqrt_batch = theta_sqrt.repeat(batch_size, 1, 1)
        theta_linear_batch = theta_linear.repeat(batch_size, 1, 1)
        
        if t==0:
            [gen_max, gen_min, SOC_last]=map(torch.from_numpy,[ppc["gen"][0,0][:,Pmax], ppc["gen"][0,0][:,Pmin], SOC_last_solve.T])
            setpoint_batch = setpoint_series_batch[:,:,t]
            SOC_last_batch = SOC_last.squeeze().repeat(batch_size, 1)
            gen_max_batch = gen_max.repeat(batch_size, 1)
            gen_max_batch[:, 5:7] = wind_series_batch[:, :, t]
            gen_min_batch = gen_min.repeat(batch_size, 1)
        else:
            ### needs to be changed from tensor to ppc and future information
            setpoint_batch = setpoint_series_batch[:,:,t]
            gen_max_batch, gen_min_batch, SOC_last_batch= update_info_torch_batch(ppc, p_g_batch, SOC_last_batch, storage_index, Initial_Pmax, wind_series_batch[:, :, t])


        p_g_batch, R_0_batch= policy(gen_max_batch, gen_min_batch, setpoint_batch, theta_sqrt_batch, theta_linear_batch, SOC_last_batch, solver_args={'solve_method':'ECOS'})
        
        # print(policy)
        # print(sum(p.numel() for p in policy.parameters() if p.require_grad))
        # exit(1)
        # p_g, R_0= policy(gen_max, gen_min, setpoint, theta_sqrt, theta_linear, SOC_last)
        ### loss
        # print(p_g)
#         print(p_g_batch.shape)
#         # print(R_0)
#         print(R_0_batch.shape)
        # p_g_record=p_g
        # SOC_record=SOC_last
        loss= cal_loss_batch_stage(ppc, p_g_batch, R_0_batch, penalty_info, setpoint_batch)
        losses = losses+ loss
        # print(loss)
        time_record[:,t] = time.time()-time_start


        ### needs to be changed from tensor to ppc and future information
        #P_variables=p_g.detach().numpy()

        ###### ????????? 
        #ppc,SOC_last_solve=update_PPC_batch(ppc,P_variables, SOC_last_solve, Initial_Pmax)

    return losses, time_record