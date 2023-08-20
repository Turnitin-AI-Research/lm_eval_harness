This is a fork of Eleuther AI's [Language Model Evaluation Harness](https://github.com/EleutherAI/lm-evaluation-harness). The code has been significantly enhanced to run experiments for the paper [TRANSFORMER DYNAMICS AS MOVEMENT THROUGH EMBEDDING SPACE](). New scripts have been written to run these experiments (explained below). Please use the original repo if you want to run the older / original scripts since those may no longer work reliably in this modified repo.

# Dev Environment
We used Ubuntu 20.04 (LTS) machines with NVIDIA 2080Ti, A10 and A100 GPUs. These tests should work on Ubuntu machines with other GPUs as well as long as you have the CUDA environment setup properly.

## Python environment setup
All tests were run on python version 3.9. A higher version should work too, but we haven't tried that.

Ensure that python3.9 and python3.9-venv is installed and available
```
  $ sudo apt install python3.9
  $ sudo apt install python3.9-venv
  $ sudo apt install python3.9-dev
```

Then setup a python virtual environment called 'lme' in the directory .venv_lme as follows.
```
$ python3.9 -m venv --prompt lme .venv_lme
$ source .venv_lme/bin/activate
$ pip install pip-tools
$ pip-compile env_files/lme.in > env_files/lme.txt
$ pip-sync env_files/lme.txt
```
All code runs under the above virtual environment.

`env_files/lme.in` is a dependency file where all the environment packages are listed. This gets compiled into a requirements file `lme.txt` which is used to populate the python virtual environment as above. If you choose to use a different version of python, or edit `lme.in`, then recreate `lme.txt` as below:

```
$ cd env_files
$ pip-compile lme.in > lme.txt
$ pip-sync lme.txt
```

# Running the experiments
The following scripts were used to generate results under the *Experiments* section of the paper. Each script supports hundreds of configurations all of which
can't be run at the same time. You'll need to edit the `run` function at the top of each script with the specific configurations you want to run. If you have multiple GPUs, these scripts will run multiple tests in parallel (via ray) based on the NUM_GPUS_PER_RUN argument (which defaults to 1). You'll need to adjust this argument for models that require more than one GPU and in that case the script will automatically apply model-parallelism. Rerunning the script only runs the remaining configurations. If you want to override this behaviour, supply --overwrite-results flag and the script will then run all specified configurations. Supply the --help flag to view script usage. All scripts are run from the top folder. For e.g.

```
(lme) ~/src/lm_eval_harness$ scripts/run_parallel_gen_t5.py --help
2023-08-20 02:26:06.974751: I tensorflow/core/util/port.cc:110] oneDNN custom operations are on. You may see slightly different numerical results due to floating-point round-off errors from different computation orders. To turn them off, set the environment variable `TF_ENABLE_ONEDNN_OPTS=0`.
2023-08-20 02:26:07.069512: I tensorflow/core/platform/cpu_feature_guard.cc:182] This TensorFlow binary is optimized to use available CPU instructions in performance-critical operations.
To enable the following instructions: AVX2 AVX512F AVX512_VNNI FMA, in other operations, rebuild TensorFlow with the appropriate compiler flags.
2023-08-20 02:26:08.354494: W tensorflow/compiler/tf2tensorrt/utils/py_utils.cc:38] TF-TRT Warning: Could not find TensorRT
INFO: Showing help with the command 'run_parallel_gen_t5.py -- --help'.


NAME
    run_parallel_gen_t5.py

SYNOPSIS
    run_parallel_gen_t5.py <flags>

FLAGS
    -s, --shutdown_at_exit=SHUTDOWN_AT_EXIT
        Type: bool
        Default: False
    -o, --overwrite_results=OVERWRITE_RESULTS
        Type: bool
        Default: False
    -N, --NUM_GPUS_PER_RUN=NUM_GPUS_PER_RUN
        Type: int
        Default: 1
    -c, --cluster=CLUSTER
        Type: str
        Default: 'local'
(END)
```
1. **scripts/run_parallel_gen_gpt.py**
Run this script to reproduce results of *Test 1: Token Generation Conditioned on Aggregated Concepts* on decoder-only models. Results get written to the folder `lmeval_results_gen/`.
1. **scripts/run_parallel_gen_t5.py**
Run this script to reproduce results of *Test 1: Token Generation Conditioned on Aggregated Concepts* on encoder-decoder models. Results get written to the folder `lmeval_results_t5/`
1. **scripts/run_parallel_sim_gpt.py**
Run this script to reproduce results of *Test 2: Concept Similarity* on decoder-only models. Results get written to the folder `lmeval_results_sim_latest/`.
1. **scripts/run_parallel_sim_t5.py**
Run this script to reproduce results of *Test 2: Concept Similarity* on encoder-decoder models. Results get written to the folder `lmeval_results_t5/`.

Ignore the remaining scripts.

# Analyzing the results
Results are analyzed via notebooks in the `notebooks/` folder.

First run all of the following notebooks to generate intermediate results: dist_sim_charts_t5.ipynb, dist_sim_charts_gpt.ipynb, dist_gen_charts_t5.ipynb and dist_gen_charts_gpt.ipynb. These will write some files to the `cache` folder.

Next run the notebooks dist_gen_charts.ipynb and dist_sim_charts.ipynb.

The notebooks should be self explanatory. Ignore the remaining ones.
