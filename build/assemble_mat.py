import numpy as np
import chaospy as cp
import scipy as sp
import functools
from joblib import Parallel, delayed
import sympy

'''
This file contains the implementation of the parallel assembly of the "probabilistic stiffnes matrices"
'''

def to_sympy_monomial(poly,n_var,i, n_init=0):
    '''
    Convert the i-th monomial of the given chaospy polynomial into a sympy expression
    '''
    n_, c_, e_ = np.array(poly.names), np.array(poly.coefficients), np.array(poly.exponents)
    n_monom = len(e_)
    assert i<n_monom, f"the polynomial has only {n_monom} monomials"
    # create symbols for each variable
    x = [sympy.symbols(f"x{i}") for i in range(n_init,n_var+n_init)]
    # identify the variables present the i-th monomial
    x_present = [int([k for k in n_[l]][-1]) for l in range(len(n_))]
    # build the monomial
    monom = c_[i]
    for j in range(len(x_present)): monom = monom * x[x_present[j]]**e_[i,j]
    return monom

def to_sympy(poly,n_var, n_init=0):
    '''
    Convert the given chaospy polynomial into a sympy expression
    '''
    n_, c_, e_ = np.array(poly.names), np.array(poly.coefficients), np.array(poly.exponents)
    n_monom = len(e_)
    polyn = 0
    for i in range(n_monom): polyn = polyn + to_sympy_monomial(poly,n_var,i, n_init)
    return polyn


class Stiffness_probamat_assembly():
    '''
    Class that implements the assembly of a general stiffness matrix in a probability space with a certain associated probability measure.
    The expression to be implemented is formed by a product between a polynomial argument and a general lambda function.
    - prob_measure: a chaospy stochastic vector 
    - degree: maximum degree of the askey-scheme polynomials taken orthogonal with respect to the prob_measure
    - polynomials: list of the considered askey-scheme polynomials
    - n_polynomials: number of considered askey-scheme polynomials
    - n: number of components of the stochastic process with measure prob_measure
    '''
    def __init__(self, prob_measure_, degree_: int, n: int, polynomials_=None):
        self.prob_measure = prob_measure_
        self.degree = degree_
        if polynomials_ is None:
            self.polynomials = cp.generate_expansion(self.degree, self.prob_measure)
        else: self.polynomials = polynomials_
        self.n_polynomials = len(self.polynomials)
        self.n = n
    def argument_poly(self, poly:list):
        '''
        Return a sympy expression of the product of the polynomials indicized by the poly argument
        - poly: list of indices indicating the polynomials to be taken in consideration
        '''
        arg = self.polynomials[0]
        for i in range(len(poly)):
            arg = arg * self.polynomials[poly[i]]
        if type(arg).__name__=="ndpoly":
            return to_sympy(arg,self.n)
        return arg
    
    def argument_func(self, func:list):
        '''
        Return a lambda function that is the product of the single lambda functions passed as argument
        - func: list of lambda functions to be considered in the multiplication
        '''
        if (len(func)>0): arg = functools.reduce(lambda f,g: lambda *x: f(*x)*g(*x), func)
        else: arg = lambda *x: 1
        return arg
    def argument(self, poly:list = [0], func:list = []):
        '''
        Return the final argument to be integrated in the prob_measure domain
        - poly: list of indices of the polynomials to be considered
        - func: list of lambda functions to be considered
        '''
        x = [sympy.symbols(f"x{i}") for i in range(self.n)]
        argument_list = [sympy.lambdify(x,self.argument_poly(poly)),self.argument_func(func), lambda *x: self.prob_measure.pdf(x)]
        argument = functools.reduce(lambda f,g: lambda *x: f(*x)*g(*x), argument_list)
        return argument
    def integrate(self, poly:list = [0], func:list = [],a:list = [-1], b:list = [1]):
        '''
        Integrate the expression given by the product of the indicated polynomials and lambda functions
        - poly: list of indices of the polynomials to be considered
        - func: list of lambda functions to be considered
        - a: list of length n of lower bounds of the n-dimensional space of integration
        - b: list of length n of upper bounds of the n-dimensional space of integration
        '''
        assert np.logical_and(len(a)==self.n, len(b)==self.n), f"the lenght of the integration extrema lists must be equal to {self.n}"
        argument = self.argument(poly, func)
        int_argument = sp.integrate.nquad(argument,[[a[i],b[i]] for i in range(self.n)])
        return int_argument[0]
    def compute_component(self, idx:int, dim: int, components: list, func,a: list,b: list):
        '''
        Compute certain components of the final stiffness matrix:
        - dim: number of dimensions of the stiffness matrix
        - components: list of the indices of the components to be computed
        - func: list of lambda functions to be considered
        - a: list of length n of lower bounds of the n-dimensional space of integration
        - b: list of length n of upper bounds of the n-dimensional space of integration
        '''
        #matrix = np.zeros((self.n_polynomials ** dim))
        matrix = np.zeros((len(components)))
        for i in range(len(components)):
            matrix[i]=self.integrate([(components[i]//self.n_polynomials**j)%self.n_polynomials for j in range(dim)[::-1]],func,a=a,b=b)
        return matrix
    def assemble_matrix(self, dim: int, func:list = [],a:list = [-1], b:list = [1],**kwargs):
        '''
        Assemble the desired stiffness matrix in parallel:
        - dim: number of dimensions of the stiffness matrix
        - func: list of lambda functions to be considered
        - a: list of length n of lower bounds of the n-dimensional space of integration
        - b: list of length n of upper bounds of the n-dimensional space of integration
        '''
        num_cores = kwargs.pop('num_cores', 1)  # Number of cores to use, -1 for all available cores
        part = (self.n_polynomials**dim)//num_cores
        ranges_ = [range(part*i,part*(i+1)) for i in range(num_cores-1)]+[range(part*(num_cores-1),self.n_polynomials**dim)]
        delayed_funcs = [delayed(self.compute_component)(i, dim, ranges_[i], func,a,b) for i in range(num_cores)]
        parallel_pool = Parallel(n_jobs=num_cores)
        matrix = np.array([])
        for i in parallel_pool(delayed_funcs): matrix = np.concatenate([matrix,i])
        return matrix.reshape([self.n_polynomials for _ in range(dim)])
        
