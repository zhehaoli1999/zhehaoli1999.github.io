---
layout: post
title:  "How to choose & design proper AutoDiff tools"
---

Automatic differentiation (AutoDiff) tools are needed in the computer graphics community in many places nowadays. In essence, AutoDiff works with the computational graph. According to [the talk of TinyAD](https://www.youtube.com/watch?v=FGG07HoVFEk), AutoDiff can be categorized based on several dimensions: 

| Dimension | Description | Note | 
|-------|--------|----| 
| **Mode** | Forward-mode vs. Reverse-mode (or backward-mode) |  |
| **Differentiation Time** | Before compilation vs. compile time vs. run time| The trade-off of compiler optimization opportunity vs. flexibility  |
| **Computation Graph**  | Whether explicitly expressed | 

Currently, there are a lot of domain-specified autodiff tools aiming to solve the differentiable programming problems in geometry processing, simulation or rendering, and here I categorized some of the most popular tools with the above dimensions: 

| Tools | Description | 
| ---- |--- | 
| [DiffTaichi](https://github.com/taichi-dev/difftaichi) | For simulation | 
| [TinyAD](https://github.com/patr-schm/TinyAD) |For geometry| 
| [Dr.jit](https://github.com/mitsuba-renderer/drjit) |For Rendering | 


|  | Forward-mode | Reverse-mode | 
| ---- |--- |--- | 
| **Before Compilation** | | | 
| **Compile Time** |  TinyAD, DiffTaichi  | DiffTaichi  | 
| **Run Time** | | | 

These days I have been thinking what kind of autodiff tools are needed for physics simulation in computer graphics, from my experience, there are several metrics: 

| Metric | Description | Motivation | 
|-------|--------| ------| 
| **Non-invasiveness** | The less change to original simulation code, the better | I don't want to change my forward simulation code too much |
| **Versatility** | Able to handle complex control flow, support forward & backward modes | Simulation code can be complex, sometimes involving linear system solving |
| **Efficiency**   | Gradient computation efficiency | For differentiable simulation, speed matters |
| **Customization**   | Customization of gradient computation| I'd like to easily control which part need to be differentiated or to be ignored |


So, I try to survey and test the above tools in the computer graphics and robotics communities nowadays to get a comparison:  


| Tool | Non-invasiveness | Versatility | Efficiency| 
| ----| ----|---| ---| 
| DiffTaichi |  Bad, I don't want to change my C++ code to taichi | Not-bad, though still in developing | Good |  
| TinyAD |   |   |  | 
| Dr.jit |    |  | 

