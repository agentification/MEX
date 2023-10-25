import os
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd

min_step = 1e6
import numpy as np
import matplotlib.pyplot as plt
import os
from plot_utils import *


lw = 1.2

result_dir = "./results"
# env = "hopper-vel-sparse"
env = "Hopper-v3"
# runs_names = [f"TD3_{env}_baseline", f"IO_{env}_eta_1e-3_cql_exp_inverse",f"IO_{env}_eta_1e-4_cql_exp_inverse", f"IO_{env}_eta_5e-3_cql_exp_inverse", f"IO_{env}_eta_5e-4_cql_exp_inverse"]
# runs_names = [f"TD3_{env}_baseline", f"IO_{env}_adjusted_v2_eta_1e-3_cql_exp_inverse"]
runs_names = [f"TD3_{env}_baseline", f"IO_{env}_eta_5e-4_cql_exp_inverse"]
# runs_names = [f"IO_{env}_eta_1e-3", f"IO_{env}_eta_1e-3_cql", f"IO_{env}_eta_1e-2_cql" ,f"TD3_{env}_baseline"]
# runs_names = [f"IO_{env}_eta_1e-2",f"IO_{env}_eta_1e-4",f"IO_{env}_eta_1e-3", f"TD3_{env}_baseline"]
files = os.listdir(result_dir)
datas = []
labels = ["TD3", "MEX-MF"]
colors = ['red','green']
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
fig, ax = plt.subplots(1, 1, figsize=(6, 4.5))
for i,data in enumerate(datas):
    all_df_list = []
    for line in data[-1]:
        df = pd.DataFrame({"env_step": data[0]*1e6, "episode_reward": line})
        all_df_list.append(df)
    all_df = pd.concat(all_df_list)
    sns.lineplot(x='env_step', y='episode_reward', data=all_df.reset_index(level=0), ax=ax, linewidth=lw, color=colors[i],label=labels[i])
#
ax.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
ax.set_xlabel('Steps')
ax.set_ylabel('Return')
ax.set_title(env)
fig.tight_layout()
save = f'./{env}.pdf'
d = os.path.dirname(save)
if not os.path.exists(d):
    os.makedirs(d)

fig.savefig(save, transparent=True)