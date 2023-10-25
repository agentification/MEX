import numpy as np
import matplotlib.pyplot as plt
import os
from plot_utils import *

result_dir = "./results"
# result_dir  ="./learning_curves/HalfCheetah/"
# runs_names = ["IO_Ant-v3_eta_1e-3"]
config['ylim']= (0., 12000.)
config['smooth_range']= 20
# env = "hopper-vel-sparse"
env = "HalfCheetah-v3"
runs_names = [f"TD3_{env}_baseline", f"IO_{env}_eta_1e-3_cql_exp_inverse",f"IO_{env}_eta_1e-4_cql_exp_inverse", f"IO_{env}_eta_5e-3_cql_exp_inverse", f"IO_{env}_eta_5e-4_cql_exp_inverse"]
# runs_names = [f"IO_{env}_eta_1e-3", f"IO_{env}_eta_1e-3_cql", f"IO_{env}_eta_1e-2_cql" ,f"TD3_{env}_baseline"]
# runs_names = [f"IO_{env}_eta_1e-2",f"IO_{env}_eta_1e-4",f"IO_{env}_eta_1e-3", f"TD3_{env}_baseline"]
files = os.listdir(result_dir)
datas = []
for runs_name in runs_names:
    run_files = [os.path.join(result_dir, runs_name+f"_{i}.npy") for i in range(5)]
    # run_data = []
    # print(run_files)
    # for run_file in run_files:
    #     data = np.load(os.path.join(result_dir, run_file))
    #     # print(data)
    #     print(len(data))
    #     run_data.append(data)

    datas.append(data_read_npy(run_files))

plot_all(datas, runs_names, 1)
plt.title(f'{env}', size=30)
legend()
plt.show()
