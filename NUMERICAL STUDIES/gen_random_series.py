import numpy as np
import torch
def gen_random_series(Generators, stage_num, category, batch_size, seed):
    Pmax=8
    wind_basic = Generators[5:7,[Pmax]]
    if category=='uniform':  ### uniform
        torch.manual_seed(seed)
        wind_basic_torch = torch.from_numpy(wind_basic)
        wind_series_batch = wind_basic_torch + 0.9* (torch.rand(batch_size, 2, stage_num, dtype=torch.float64)-0.5)
        setpoint_series_batch = (torch.rand(batch_size, 1, stage_num, dtype=torch.float64)-0.5)*3
        
    elif category=='DRO':
        torch.manual_seed(seed)
        wind_basic_torch = torch.from_numpy(wind_basic)
        a = torch.ones(batch_size, 2, stage_num)* 0.5
        wind_deviation = (torch.bernoulli(a)-0.5)* 0.9
        wind_series_batch = (wind_basic_torch + wind_deviation).double()
        
        b = torch.ones(batch_size, 1, stage_num)* 2/3
        setpoint_series_batch = ((torch.bernoulli(b)-0.5)*3.0).double()
        
    elif category=='gaussian':
        torch.manual_seed(seed)
        wind_basic_torch = torch.from_numpy(wind_basic)*torch.ones(2, stage_num, dtype=torch.float64)
        wind_basic_batch = wind_basic_torch.repeat(batch_size, 1, 1)
        wind_delta = 0.9/3
        wind_series_batch = torch.normal(mean=wind_basic_batch, std=wind_delta)
        wind_min = wind_basic_torch - 0.45*torch.ones(2, stage_num, dtype=torch.float64)
        wind_max = wind_basic_torch + 0.45*torch.ones(2, stage_num, dtype=torch.float64)
        wind_series_batch = torch.clamp(wind_series_batch, min=wind_min, max=wind_max)

        setpoint_basic_torch = 0.5*torch.ones(1, stage_num, dtype=torch.float64)
        setpoint_basic_batch = setpoint_basic_torch.repeat(batch_size, 1, 1)
        setpoint_delta = 1.5/3
        setpoint_series_batch = torch.normal(mean=setpoint_basic_batch, std=setpoint_delta)
        setpoint_min = -1.5* torch.ones(1, stage_num, dtype=torch.float64)
        setpoint_max = 1.5* torch.ones(1, stage_num, dtype=torch.float64)
        setpoint_series_batch = torch.clamp(setpoint_series_batch, min = setpoint_min, max = setpoint_max)
    
    
    elif category=='worst':
        with open('./data/determinitic_param_case_10.npy', 'rb') as f:
            wind_series_value = torch.from_numpy(np.load(f))
            setpoint_series_value = torch.from_numpy(np.load(f))
        wind_series_batch = wind_series_value.repeat(batch_size, 1, 1)
        setpoint_series_batch = setpoint_series_value.repeat(batch_size, 1, 1)
#         wind_basic_torch = torch.from_numpy(wind_basic)*torch.ones(2, stage_num, dtype=torch.float64)
#         wind_series_batch = wind_basic_torch.repeat(batch_size, 1, 1)+ 0.45*torch.ones(2, stage_num, dtype=torch.float64)
#         setpoint_basic_torch = 1.5 *torch.ones(1, stage_num, dtype=torch.float64)
#         setpoint_series_batch = setpoint_basic_torch.repeat(batch_size, 1, 1)
        
    elif category=='average':
        wind_basic_torch = torch.from_numpy(wind_basic)*torch.ones(2, stage_num, dtype=torch.float64)
        wind_series_batch = wind_basic_torch.repeat(batch_size, 1, 1)
#         setpoint_basic_torch = 0.5*torch.ones(1, stage_num, dtype=torch.float64)
        setpoint_basic_torch = 0.5*torch.ones(1, stage_num, dtype=torch.float64)
        setpoint_series_batch = setpoint_basic_torch.repeat(batch_size, 1, 1)
    
    return wind_series_batch, setpoint_series_batch