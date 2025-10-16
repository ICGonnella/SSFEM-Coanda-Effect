# SSFEM for Coanda-Effect

Implementation of the SSFEM pipeline relying on the __Deal-II__ C++ library tested for the Coanda Effect, a benchmark in Fluid Dynamics explained in [_A stochastic perturbation approach to nonlinear bifurcating problems_](https://arxiv.org/abs/2402.16803).

![pipeline](./pipeline.png)

### Activate the ssfem python environment

```
python3.9 -m venv ssfem
source ssfem/bin/activate
python --version           # should show Python 3.9.x
pip install -r requirements.txt
```

### build the code

```
cd build
export PYBIND11_DIR=$(python -m pybind11 --cmakedir)
cmake -DCMAKE_BUILD_TYPE=Release -Dpybind11_DIR="$PYBIND11_DIR" ..
make
```

### run the code in parallel

```
mpirun -np 4 python test.py
```