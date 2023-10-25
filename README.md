# Maximize to Explore: One Objective Function Fusing Estimation, Planning, and Exploration

Code for  [Maximize to Explore: One Objective Function Fusing Estimation, Planning, and Exploration](https://arxiv.org/abs/2305.18258). Paper accepted at NeurIPS 2023!

Authors: [Zhihan Liu](https://scholar.google.com/citations?user=uEl_TtkAAAAJ&hl=en)&ast;, [Miao Lu](https://miaolu3.github.io)&ast;, [Wei Xiong](https://weixiongust.github.io/WeiXiongUST/index.html)&ast;, [Han Zhong](https://hanzhong-ml.github.io), [Hao Hu](http://mousehu.cn), [Shenao Zhang](https://shenao-zhang.github.io), [Sirui Zheng](https://openreview.net/profile?id=~Sirui_Zheng2), [Zhuoran Yang](https://zhuoranyang.github.io), [Zhaoran Wang](https://zhaoranwang.github.io) (&ast; indicates equal contribution)

## Model-Based **MEX-MB**

### Installation

The code can be set up by:
    
    git clone https://github.com/agentification/MEX.git
    cd MEX/MEX_MB
    pip install -e ".[dev]"

### Basic example

Below we provide an example to train **MEX-MB** in a single environment, e.g., Ant-v2:

    python ./mbrl/examples/main.py algorithm=mbpo overrides=mbpo_ant comment=mbpo device=cuda:0 seed=0

To train **MEX-MB** in other environments, change the ``overrides`` argument to the ones in ``MEX_MB/mbrl/examples/conf/overrides``.

#### Sparse Environments

The sparse environments are implemented in the ``MEX_MB/mujoco`` folder, which can replace the original ``gym/envs/mujoco`` to enable training in the sparse-reward tasks.

## Model-Free **MEX-MF**

### Installation
**MEX-MF** is trained using Python 3.7 and PyTorch 1.2. 

Other dependencies can be set up by:

    cd MEX/MEX_MF
    pip install -r env.txt

### Experiments
The results in the paper can be reproduced by running:

    ./run_experiments.sh

Below we provide an example to train **MEX-MF** in a single environment, e.g., HalfCheetah-v2:

    python main.py --env HalfCheetah-v2 --policy IO

#### Sparse Environments
For sparse tasks, please specify --sparse, e.g.,

    python main.py --env walker-vel-sparse --sprase --policy IO

## Citation

```bibtex
@article{liu2023one,
  title={One Objective to Rule Them All: A Maximization Objective Fusing Estimation and Planning for Exploration},
  author={Liu, Zhihan and Lu, Miao and Xiong, Wei and Zhong, Han and Hu, Hao and Zhang, Shenao and Zheng, Sirui and Yang, Zhuoran and Wang, Zhaoran},
  journal={arXiv preprint arXiv:2305.18258},
  year={2023}
}
```
