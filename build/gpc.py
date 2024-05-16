import chaospy as cp
from tqdm import tqdm
import numpy as np

class GPC_Expansion():
    '''
    assumption: Lagrange Finite Elements
    '''
    def __init__(self, poly_degree: int, rv, coefficients):
        self.poly_degree = poly_degree
        self.rv = rv
        self.coefficients = coefficients
        self.stochastic_bases, self.stochastic_bases_norm = cp.generate_expansion(self.poly_degree, self.rv, retall=True)
        self.mean = self.compute_mean()
        self.variance = self.compute_variance()
            
        #self.deterministic_bases = np.eye(self.dim_samples)

    def stochastic_bases_linear_comb(self,coeffs, poly):
        return (coeffs*poly).sum()

    def compute_mean(self):
        return self.coefficients[:,0]

    def compute_variance(self):
        return np.sum([self.stochastic_bases_norm[i]*(self.coefficients[:,i])**2 for i in range(1,self.coefficients.shape[1])], axis=0)
        
    def expand(self, n_samples_integral=1000):
        terms = [self.coefficients[:,i] * self.stochastic_bases[i] for i in range(len(self.stochastic_bases))]
        self.expansion = terms[0]
        for i in range(1, len(terms)):
            self.expansion = self.expansion + terms[i]
        return
    
    def reconstruct(self, n_reconstructed_samples: int, dof_idx=None):
        samples = rv.sample(n_reconstructed_samples)
        sol_samples = []
        for sample in tqdm(samples):
            if dof_idx is None:
                sol_samples.append(np.array([elem.subs('x0',sample) for elem in self.expansion]))
            else:
                sol_samples.append(np.array([self.expansion[idx].subs('x0',sample) for idx in dof_idx]))
        return samples, sol_samples