import libcoanda
import chaospy
import assemble_mat as asmat
import gpc
import utils
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
from joblib import Parallel, delayed

libcoanda.init_MPI(sys.argv)

fe_degree=1
upload=True
n_glob_ref=3
viscosity = 0.9
viscosity_var = 0.001
N_PC=3
load_initial_guess = False
n_blocks_to_load=10
grid = "06"
verbose  = False

max_iter=100
solver_type="automatic"
strategy="newton"
abs_tolerance=1e-7
rel_tolerance=1e-15 
NonlinearSolver="Line Search Based"       
direction_Method="Newton"                 
direction_SD_ScalingType="F 2-Norm"        
linesearch_Method="Full Step"             
linesearch_FS_FullStep=1.0
linesearch_BT_MinimumStep=0.001 
linesearch_BT_DefaultStep=1.0
linesearch_BT_RecoveryStep=1.0 
linesearch_BT_MaximumStep=1.0 
linesearch_BT_Reduction_Factor=0.5


deg=N_PC-1
distribution = "Normal"
mesh_path = f"../mesh/mesh{grid}.msh"
out_path = f"coanda_deg{deg}_mean{viscosity}_var{viscosity_var}_{grid}_{distribution}.vtu"

c = libcoanda.StationaryCoanda(fe_degree=fe_degree, 
                               upload=upload, 
                               n_glob_ref=n_glob_ref, 
                               mesh_input_file=mesh_path, 
                               viscosity=viscosity, 
                               var=viscosity_var, 
                               N_PC=N_PC,
                               load_initial_guess=load_initial_guess, 
                               n_blocks_to_load=n_blocks_to_load,
                               output_file=out_path,
                               verbose=verbose)

c.initialize()
# ---------------------------------- set and get parameters ------------------------------------------------
xi = chaospy.Normal()

dof_u = c.get_dof_u()
dof_p = c.get_dof_p()
coord = c.get_coord_list()
coord_expansion = utils.separate(coord, dof_u, dof_p)
sol = np.load(f"../results/solution_deg{deg}_mean{viscosity}_var{viscosity_var}_{grid}_{distribution}.npy")

p = np.array([[14.6807, 3.8331]])
polynomials_evaluated = np.zeros((len(p)*3, N_PC))

for i in range(N_PC):
    res = c.get_point_val(p, sol, i)
    polynomials_evaluated[::3,i] = res[:,0]
    polynomials_evaluated[1::3,i] = res[:,1]
    polynomials_evaluated[2::3,i] = res[:,2]
gpc_form = gpc.GPC_Expansion(deg, xi, polynomials_evaluated)
gpc_form.expand()
#sol_expansion = utils.separate(gpc_form.expansion, dof_u, dof_p)
gpc_var = gpc_form.compute_variance()
#gpc_var_expansion = utils.separate(gpc_var, dof_u, dof_p)
gpc_mean = gpc_form.compute_mean()
#gpc_mean_expansion = utils.separate(gpc_mean, dof_u, dof_p)

n_reconstructed_samples = 100
samples = xi.sample(n_reconstructed_samples)
#print(gpc_form.expansion)
idx=1 #0-->ux  1-->uy  2-->p
print(c.rank())
if c.rank()==0: 
    print(gpc_form.expansion)
    for i in range(len(p)):
        poly = gpc_form.expansion[i*3+idx]
        #print(poly)
        sol_samples = []
        for sample in samples:
            sol_samples.append(poly(sample))# - gpc_mean[i*3+idx])
        fig, ax = plt.subplots(1,1)
        ax.plot(samples, sol_samples, 'o')
        ax.grid()
        ax.set_title(f"{p[i]}")
        if idx==0:
            fig.savefig(f"../plots/deg{deg}_mean{viscosity}_var{viscosity_var}_{grid}_{distribution}_point_{np.round(p[i][0],3)}_{np.round(p[i][1],3)}_ux.pdf")
        if idx==1:
            fig.savefig(f"../plots/deg{deg}_mean{viscosity}_var{viscosity_var}_{grid}_{distribution}_point_{np.round(p[i][0],3)}_{np.round(p[i][1],3)}_uy.pdf")
        if idx==2:
            fig.savefig(f"../plots/deg{deg}_mean{viscosity}_var{viscosity_var}_{grid}_{distribution}_point_{np.round(p[i][0],3)}_{np.round(p[i][1],3)}_p.pdf")
