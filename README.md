

# Bi-level Multiobjective Learning Framework

Source code for the paper "**[Bi-Level Multiobjective Evolutionary Learning: A Case Study on Multitask Graph Neural Topology Search
](https://ieeexplore.ieee.org/abstract/document/10065594)**".


We present a bi-level multiobjective learning framework (BLMOL), coupling the decision-making process with the optimization process of the upper-level MOP (UL-MOP) by introducing LL preference r. 

We consider a novel case study on multitask graph neural topology search. It aims to find a set of Pareto topologies and their Pareto weights, representing different tradeoffs across tasks at UL and LL, respectively. The found graph neural network is employed to solve multiple tasks simultaneously, including graph classification, node classification, and link prediction.


## Running
### Generate the surrogate data
1. python gen_surr_data.py

### Construct surrogate models
2. python con_surr_model.py

### Evolutionary search with surrogate models                   
3. python search.py
           
PS: NSGAII, a powerful MOEA, is employed to optimize UL objectives in BLMOL. LS, a popular GPMOA, is embedded in BLMOL for training weights.

## Results
Please refer to our paper.

## Reference
Please cite the paper whenever our proposed BLMOL is used to produce published results or incorporated into other software:
```
@ARTICLE{10065594,
  author={Wang, Chao and Jiao, Licheng and Zhao, Jiaxuan and Li, Lingling and Liu, Xu and Liu, Fang and Yang, Shuyuan},
  journal={IEEE Transactions on Evolutionary Computation}, 
  title={Bi-Level Multiobjective Evolutionary Learning: A Case Study on Multitask Graph Neural Topology Search}, 
  year={2024},
  volume={28},
  number={1},
  pages={208-222},
  keywords={Task analysis;Topology;Machine learning;Multitasking;Optimization;Search problems;Decision making;Bi-level multiobjective learning;multitask learning (MTL);neural topology search},
  doi={10.1109/TEVC.2023.3255263}}

```


