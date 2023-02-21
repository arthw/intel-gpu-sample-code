# Intel GPU Sample Code

## Setup Environment

```
python -m venv env_itex
source env_itex/bin/activate
pip install --upgrade pip
pip install tensorflow  intel-extension-for-tensorflow[gpu] intel-extension-for-tensorflow-lib intel-optimization-for-horovod
pip install gin gin-config tensorflow-addons tensorflow-model-optimization tensorflow-datasets
```

Activate oneAPI Running Environment
```
source /opt/intel/oneapi/setvars.sh
```

## Download Sample Code

1. Single GPU

```
wget https://raw.githubusercontent.com/arthw/intel-gpu-sample-code/main/tf2_keras_mnist_gpu.py
python tf2_keras_mnist_gpu.py
```


2. Multiple GPU

```
wget https://raw.githubusercontent.com/arthw/intel-gpu-sample-code/main/tf2_hvd_keras_mnist_gpu.py
python tf2_hvd_keras_mnist_gpu.py
```
