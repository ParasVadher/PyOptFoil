# PyOptFoil
PyOptFoil is a Python tool for the optimization of aerofoils which utilises the panel-based solver [XFOIL](https://web.mit.edu/drela/Public/web/xfoil/).

The code uses the Bezier-Parsec 3333 method for parameterizing aerofoil shapes with the Differential Evolution (DE/rand-to-best/1) optimization algorithm. 
More options for parameterization method and optimization algorithm to come with future updates. 

## Usage
The code below demostrates usage of PyOptFoil to obtain an aerofoil which minimizes drag at a desired lift coefficient of 0.6 and Reynolds number of 500,000 within the incidence range [-1,5]. 

```python
from pyoptfoil.opt import opt
from pyoptfoil.algorithms.de import DE

c_bounds={'gamma_le':(0.0001,0.5),'x_c':[0.3,0.6],'y_c':[0.00,0.1],'k_c':[-1,-0.01],'z_te':[0.0,0.00],'alpha_te':[0.0001,0.5]}
t_bounds={'r_le':[-0.04,-0.001],'x_t':[0.15,0.4],'y_t':[0.1,0.2],'k_t':[-1,0.1],'dz_te':[0.0,0.001],'beta_te':[0.001,0.3]}
bounds = {**t_bounds, **c_bounds}

pop_size = 120  # population size
gens = 50  # number of generations
param_method = 'BP3333'  # parameterization method
f = 0.85  # mutation factor
cr = 1  # crossover probability

optimizer = DE(bounds,pop_size,gens,param_method,f,cr)

opt(optimizer,0.6,5e5,0,(-1,5,0.5))
```

## Fit Mode
The code can also be used to obtain the parameters which best fit a known aerofoil shape for a given parameterization method. An example of this is demonstrated below.

```python
from pyoptfoil.fit import fit
from pyoptfoil.algorithms.de import DE
import numpy as np

c_bounds={'gamma_le':(0.0001,0.5),'x_c':[0.3,0.6],'y_c':[0.00,0.1],'k_c':[-1,-0.01],'z_te':[0.0,0.00],'alpha_te':[0.0001,0.5]}
t_bounds={'r_le':[-0.04,-0.001],'x_t':[0.15,0.4],'y_t':[0.1,0.2],'k_t':[-1,0.1],'dz_te':[0.0,0.001],'beta_te':[0.001,0.3]}
bounds = {**t_bounds, **c_bounds}

pop_size = 120  # population size
gens = 50  # number of generations
param_method = 'BP3333'  # parameterization method
f = 0.85  # mutation factor
cr = 1  # crossover probability

x_u = np.asarray([1., 0.95, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.25, 0.2, 0.15, 0.1, 0.075, 0.05, 0.025, 0.0125, 0])
y_u = np.asarray([0.0013, 0.0114, 0.0208, 0.0375, 0.0518, 0.0636, 0.0724, 0.078, 0.0788, 0.0767, 0.0726, 0.0661, 0.0563, 0.0496, 0.0413, 0.0299, 0.0215, 0.])
x_l = np.asarray([0., 0.0125, 0.025, 0.05, 0.075, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.])
y_l = np.asarray([0., -0.0165, -0.0227, -0.0301, -0.0346, -0.0375, -0.041, -0.0423, -0.0422, -0.0412, -0.038, -0.0334, -0.0276, -0.0214, -0.015, -0.0082, -0.0048, -0.0013])


optimizer = DE(bounds,pop_size,gens,param_method,f,cr)

fit(optimizer,x_u,y_u,x_l,y_l)
```

###### Author: Paras Vadher
