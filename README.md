# SSFEM for Coanda-Effect

### activate the ssfem python environment

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