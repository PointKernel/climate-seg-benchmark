# Deep Learning Climate Segmentation Benchmark

Reference implementation for the climate segmentation benchmark, based on the
Exascale Deep Learning for Climate Analytics codebase here:
https://github.com/azrael417/ClimDeepLearn, and the paper:
https://arxiv.org/abs/1810.01993

## How to get the data

For now there is a smaller dataset (~200GB total) available to get things started.
It is hosted via Globus:

https://app.globus.org/file-manager?origin_id=bf7316d8-e918-11e9-9bfc-0a19784404f4&origin_path=%2F

and also available via https:

https://portal.nersc.gov/project/dasrepo/deepcam/climseg-data-small/

## How to prepare virtual environment


### Prerequisites (Cori)
```bash
module load python/3.7-anaconda-2019.07 cuda/10.2.89
```

### [Install TensorFlow (Conda)](https://www.tensorflow.org/install/pip?lang=python3#package-location)
Create a new virtual environment `py3.7-tf1.15` by choosing Python 3.7:
```bash
conda create -n py3.7-tf1.15 pip python=3.7
```

Activate the virtual environment and install TensorFlow (Python 3.7 GPU support):
```bash
source activate py3.7-tf1.15
(env) pip install --ignore-installed --upgrade tensorflow-gpu==1.15
conda deactivate
```

### [Install PyCUDA](https://wiki.tiker.net/PyCuda/Installation/Linux)
Download PyCUDA and unpack it:
```bash
wget https://files.pythonhosted.org/packages/5e/3f/5658c38579b41866ba21ee1b5020b8225cec86fe717e4b1c5c972de0a33c/pycuda-2019.1.2.tar.gz
tar xvf pycuda-*.tar.gz
```

Configure and build within the Conda env:
```bash
cd pycuda-*
source activate py3.7-tf1.15
(env) python configure.py --cuda-root=${CUDA_HOME}
(env) make install -j8
conda deactivate
```

## How to run the benchmark

Submission scripts are in `run_scripts`.

### Running at NERSC

To submit to the Cori KNL system, do

```bash
# This example runs on 64 nodes.
cd run_scripts
sbatch -N 64 train_cori.sh
```

To submit to the Cori GPU system, do

```bash
# 8 ranks per node, 1 per GPU
module purge
module load esslurm
cd run_scripts
sbatch -N 4 train_corigpu.sh
```
