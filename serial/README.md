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
cd ..
```

### run the code

The parameters `viscosity_mean`, `viscosity_var`, and `npc` can be adjusted depending on your needs.

> **Important:**  
> If you change the `npc` parameter, make sure to update the same value inside `coanda_ssfem.prm`.

```
python assemble_mat.py --viscosity_mean 0.9 --viscosity_var 0.001 --npc 2
build/coanda_ssfem ./coanda_ssfem.prm
```