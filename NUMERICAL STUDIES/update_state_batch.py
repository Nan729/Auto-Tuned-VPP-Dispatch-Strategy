import numpy as np
import torch
# def update_PPC_batch(ppc, P_variables, SOC_last, Initial_Pmax):
#     ### ramping information
#     ramp_rate=0.7
#     Generators = np.array(ppc["gen"][0,0][0:5,:])
#     # print(len(Generators))
#     Pmax=8
#     Pmin=9
#     PG_last=P_variables[0:5]

#     # print(P_variables)
#     # print(Initial_Pmax)
#     PG_MAX = np.min(np.array([PG_last + ramp_rate*Initial_Pmax, Initial_Pmax]), axis=0)
#     PG_MIN = np.max(np.array([PG_last - ramp_rate*Initial_Pmax, np.zeros((len(PG_last)))]), axis=0)
#     # print(PG_MAX)
#     # print(PG_MIN)
#     ppc["gen"][0,0][0:5,Pmax]=PG_MAX
#     ppc["gen"][0,0][0:5,Pmin]=PG_MIN


#     ### storage information
#     P_BA_last=P_variables[13:17]
#     # update the SOC
#     SOC_last=SOC_last-P_BA_last
#     BA_cap=1.0*np.ones((1,4))
#     BA_limit=0.5*np.ones((1,4))
#     BA_MIN=np.max(np.array([-BA_limit, -(BA_cap-SOC_last)]), axis=0)
#     BA_MAX=np.min(np.array([BA_limit, SOC_last]), axis=0)
#     ppc["gen"][0,0][13:17,Pmax]=BA_MAX
#     ppc["gen"][0,0][13:17,Pmin]=BA_MIN
#     # print(BA_MAX)
#     # print(BA_MIN)

#     return ppc, SOC_last

def update_info_torch_batch(ppc, p_g_batch, SOC_last_batch, storage_index, Initial_Pmax, wind_series):
    SOC_last_batch= SOC_last_batch - p_g_batch[:, storage_index[:,0]]
    batch_size = p_g_batch.shape[0]
    
    Generators = np.array(ppc["gen"][0,0])
    Pmax=8
    Pmin=9
    gen_max = torch.from_numpy(Generators[:,Pmax])
    gen_max_batch = gen_max.repeat(batch_size, 1)
    gen_min = torch.from_numpy(Generators[:,Pmin])
    gen_min_batch = gen_min.repeat(batch_size, 1)
######### Generators
    ramp_rate=0.7
    initial_pmax= torch.from_numpy(Initial_Pmax)
    initial_pmax = initial_pmax.repeat(batch_size, 1)
    PG_last = p_g_batch[:, 0:5]
    initial_pmin= torch.zeros(batch_size, 5)
    # print(PG_last+ramp_rate*initial_pmax)
    gen_max_batch[:, 0:5] = torch.minimum(PG_last+ramp_rate*initial_pmax, initial_pmax)
    gen_min_batch[:, 0:5] = torch.maximum(PG_last-ramp_rate*initial_pmax, initial_pmin)
######### wind: unchanged for now
    gen_max_batch[:, 5:7] = wind_series
######### demand reponse:unchanged
######### storage
    BA_cap=torch.from_numpy(1.0*np.ones((1,4)))
    BA_cap=BA_cap.repeat(batch_size, 1)
    BA_limit=torch.from_numpy(0.5*np.ones((1,4)))
    BA_limit=BA_limit.repeat(batch_size, 1)
    gen_max_batch[:, storage_index[:,0]] = torch.minimum(BA_limit, SOC_last_batch)
    gen_min_batch[:, storage_index[:,0]] = torch.maximum(-BA_limit, -(BA_cap-SOC_last_batch))


    return gen_max_batch, gen_min_batch, SOC_last_batch