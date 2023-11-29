# ReLU-QP
A GPU Accelerated Quadratic Programming Solver for Model-Predictive Control. A Python implementation can be found [here](https://github.com/RoboticExplorationLab/ReLUQP-py).

# Installation

In the Julia REPL, run the following:
```
using Pkg
Pkg.add(url="https://github.com/RoboticExplorationLab/ReLUQP.jl.git")
```

# Examples
To run the examples, first set up the examples environment by running the following in a Julia REPL
in the examples folder (this only needs to be done once).
```
using Pkg
Pkg.activate(".")
Pkg.add(url="https://github.com/RoboticExplorationLab/ReLUQP.jl.git")
Pkg.instantiate()
```
Each of the examples can then be run. We recommend using VSCode and stepping through the code with the
REPL. The examples are:
- atlas/atlas_balancing.jl
- quadruped_with_arm/quadruped_pickup.jl

## Citation
If you find this code useful, please consider citing our paper:
```
@article{bishop_relu-qp_2023,
        title = {{ReLU}-{QP}: A {GPU}-Accelerated Quadratic Programming Solver for Model-Predictive Control},
        url = {http://arxiv.org/abs/2311.18056},
        author = {Bishop, Arun L. and Zhang, John Z. and Gurumurthy, Swaminathan and Tracy, Kevin and Manchester, Zachary},
        year = {2023}
}
```