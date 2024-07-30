# SSFEM for Coanda-Effect

Implementation of the SSFEM pipeline relying on the __Deal-II__ C++ library, and it is tested for the Coand&#259 Effect, a benchmark in Fluid Dynamics explained in _A stochastic perturbation approach to nonlinear bifurcating problems_ [https://arxiv.org/abs/2402.16803](https://arxiv.org/abs/2402.16803) 

![pipeline](./pipeline.png)

### Activate the ssfem python environment

```
source ./ssfem/bin/activate
```

### build the code

```
cd build
cmake -DCMAKE_BUILD_TYPE=Release -Dpybind11_DIR={PATH_TO_PYBIND11} ..
make
```

### run the code in parallel

```
mpirun -np 4 python test.py
```