import libcoanda
import chaospy
import assemble_mat as asmat
import gpc
import utils
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys

libcoanda.init_MPI(sys.argv)

fe_degree=1
upload=True
n_glob_ref=4
viscosity = 0.9
viscosity_var = 0.001
N_PC=5
load_initial_guess = True
n_blocks_to_load=10
grid = "06"
verbose  = False

max_iter=100
solver_type="automatic"
strategy="newton"
abs_tolerance=1e-13
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
distribution = "Uniform"
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
dof_u = c.get_dof_u()
dof_p = c.get_dof_p()
xi = chaospy.Uniform(-1.73205167,1.73205167)
viscosity_mean = float((chaospy.Uniform()*(np.sqrt(viscosity_var/0.08333))).upper/2)
tmp = asmat.Stiffness_probamat_assembly( xi, deg, 1)
mat = [(viscosity) * tmp.assemble_matrix(2,a = [xi.lower[0]],b = [xi.upper[0]]) + np.sqrt(viscosity_var) * tmp.assemble_matrix(2,func = [lambda x: x],a = [xi.lower[0]],b = [xi.upper[0]]),
       tmp.assemble_matrix(2,a = [xi.lower[0]],b = [xi.upper[0]])] + [
           tmp.assemble_matrix(3,a = [xi.lower[0]],b = [xi.upper[0]])[i] for i in range(N_PC)
       ]
for i in range(N_PC+2):
    c.set_stochastic_mat(i, mat[i])
# --------------------------------------run--------------------------------------------------------
c.run(max_iter=max_iter, 
      solver_type=solver_type, 
      strategy=strategy, 
      abs_tolerance=abs_tolerance, 
      rel_tolerance=rel_tolerance, 
      NonlinearSolver=NonlinearSolver, 
      direction_Method=direction_Method, 
      direction_SD_ScalingType=direction_SD_ScalingType, 
      linesearch_Method=linesearch_Method, 
      linesearch_FS_FullStep=linesearch_FS_FullStep, 
      linesearch_BT_MinimumStep=linesearch_BT_MinimumStep, 
      linesearch_BT_DefaultStep=linesearch_BT_DefaultStep, 
      linesearch_BT_RecoveryStep=linesearch_BT_RecoveryStep, 
      linesearch_BT_MaximumStep=linesearch_BT_MaximumStep, 
      linesearch_BT_Reduction_Factor=linesearch_BT_Reduction_Factor)

# --------------------------------------- extract solution ----------------------------------------------------
coord = c.get_coord_list()
coord_expansion = utils.separate(coord, dof_u, dof_p)
sol = c.get_sol()
np.save(f"solution_deg{deg}_mean{viscosity}_var{viscosity_var}_{grid}_{distribution}.npy", sol)
gpc_form = gpc.GPC_Expansion(deg, xi, sol)
gpc_form.expand()
sol_expansion = utils.separate(gpc_form.expansion, dof_u, dof_p)
gpc_var = gpc_form.compute_variance()
gpc_var_expansion = utils.separate(gpc_var, dof_u, dof_p)
gpc_mean = gpc_form.compute_mean()
gpc_mean_expansion = utils.separate(gpc_mean, dof_u, dof_p)
c.set_sol_variance(gpc_var)
c.save_sol()


# --------------------------------------- plot polynomials ----------------------------------------------------
#idx_max_var = np.argmax(gpc_var_expansion[1])
#coord_max_var = coord_expansion[1][idx_max_var]

#poly = sol_expansion[1][idx_max_var]
#n_reconstructed_samples = 100
#samples = xi.sample(n_reconstructed_samples)
#sol_samples = []
#for sample in samples:
#    sol_samples.append(poly(sample))

#plt.plot(samples, sol_samples, 'o')
#plt.grid()
#plt.title(f"{coord_max_var}")
#plt.savefig(f"deg{deg}_mean{viscosity}_var{viscosity_var}_{grid}_{distribution}.pdf")
