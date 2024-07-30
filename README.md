# SSFEM for Coanda-Effect

This repository implements the SSFEM ![pipeline](./pipeline.png) relying on the deal-II c++ library, tested in the case descripted in [A stochastic perturbation approach to nonlinear bifurcating problems](https://arxiv.org/abs/2402.16803)

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