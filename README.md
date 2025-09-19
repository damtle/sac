**This repository is no longer maintained. Please use our new [Softlearning](https://github.com/rail-berkeley/softlearning) package instead.**

# Soft Actor-Critic

Soft actor-critic is a deep reinforcement learning framework for training maximum entropy policies in continuous domains. The algorithm is based on the paper [Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor](https://arxiv.org/abs/1801.01290) presented at ICML 2018.

## PyTorch (Python 3.10 & Windows) implementation

The repository now ships with a self-contained PyTorch implementation (`torch_sac`) that targets Python 3.10 and runs on Windows and Linux alike. It removes the old `rllab` and TensorFlow dependencies and was tested with the MuJoCo-based Walker2d benchmark.

### Quick start on Windows

1. Ensure you have a working [MuJoCo](https://mujoco.readthedocs.io/en/stable/) installation (Gymnasium's `mujoco` extra installs the precompiled binaries) and [Microsoft C++ Redistributable](https://learn.microsoft.com/cpp/windows/latest-supported-vc-redist) if it is not already available on your machine.
2. Create and activate a Python 3.10 virtual environment:

   ```powershell
   py -3.10 -m venv .venv
   .venv\Scripts\activate
   python -m pip install --upgrade pip
   ```

3. Install the lightweight dependency set:

   ```powershell
   pip install -r requirements-windows.txt
   ```

4. Launch training (Walker2d by default):

   ```powershell
   python examples/torch_train.py --env-id Walker2d-v4 --total-steps 1000000
   ```

   Add `--device cuda` if you have a CUDA-capable GPU with a matching PyTorch build.

Training artefacts are written under `runs/torch_sac/<env>_<timestamp>_seed<seed>` and include `progress.csv`, periodic checkpoints, and the final policy weights. Command line flags mirror the fields in `torch_sac.TrainConfig` (batch size, entropy tuning, evaluation cadence, etc.) so experiments can be scripted without editing the codebase. You can also use the package programmatically:

```python
from torch_sac import TrainConfig, train

config = TrainConfig(env_id="Walker2d-v4", total_steps=200_000, log_dir="runs/walker")
run_dir = train(config)
print(f"results stored in {run_dir}")
```

### Analysing training runs

The `progress.csv` file written during training captures evaluation scores, episode returns, losses, and entropy statistics. The helper script below turns one or more such logs into publication-ready figures and summary tables:

```powershell
python examples/torch_analyze.py runs/torch_sac/Walker2d-v4_*/progress.csv \
    --labels seed1 seed2 seed3 \
    --success-thresholds 3000 \
    --title "Walker2d-v4 SAC (3 seeds)"
```

Key outputs are saved under `analysis/` by default:

- `learning_curves.png` (or `.pdf`/`.svg`): 2×2 panel figure showing evaluation performance, training episode returns, critic/actor losses, and entropy temperature dynamics with optional smoothing.
- `summary_metrics.csv`: aggregated statistics covering final/best evaluation return, area under the evaluation curve, time-to-threshold, and descriptive analytics for the latest training episodes. Use `--latex` or `--markdown` to export the same table for direct inclusion in papers.

Smoothing and window lengths are configurable via `--smoothing-window` and `--episode-window`. Pass multiple log files to compare runs or seeds; the script automatically colours and labels each trajectory.

The legacy TensorFlow implementation and documentation remain below for archival purposes.

This implementation uses Tensorflow. For a PyTorch implementation of soft actor-critic, take a look at [rlkit](https://github.com/vitchyr/rlkit) by [Vitchyr Pong](https://github.com/vitchyr).

See the [DIAYN documentation](./DIAYN.md) for using SAC for learning diverse skills.

# Getting Started

Soft Actor-Critic can be run either locally or through Docker.

## Prerequisites

You will need to have [Docker](https://docs.docker.com/engine/installation/) and [Docker Compose](https://docs.docker.com/compose/install/) installed unless you want to run the environment locally.

Most of the models require a [Mujoco](https://www.roboti.us/license.html) license.

## Docker installation

If you want to run the Mujoco environments, the docker environment needs to know where to find your Mujoco license key (`mjkey.txt`). You can either copy your key into `<PATH_TO_THIS_REPOSITY>/.mujoco/mjkey.txt`, or you can specify the path to the key in your environment variables:

```
export MUJOCO_LICENSE_PATH=<path_to_mujoco>/mjkey.txt
```

Once that's done, you can run the Docker container with

```
docker-compose up
```

Docker compose creates a Docker container named `soft-actor-critic` and automatically sets the needed environment variables and volumes.

You can access the container with the typical Docker [exec](https://docs.docker.com/engine/reference/commandline/exec/)-command, i.e.

```
docker exec -it soft-actor-critic bash
```

See examples section for examples of how to train and simulate the agents.

To clean up the setup:
```
docker-compose down
```

## Local installation

To get the environment installed correctly, you will first need to clone [rllab](https://github.com/rll/rllab), and have its path added to your PYTHONPATH environment variable.

1. Clone rllab
```
cd <installation_path_of_your_choice>
git clone https://github.com/rll/rllab.git
cd rllab
git checkout b3a28992eca103cab3cb58363dd7a4bb07f250a0
export PYTHONPATH=$(pwd):${PYTHONPATH}
```

2. [Download](https://www.roboti.us/index.html) and copy mujoco files to rllab path:
  If you're running on OSX, download https://www.roboti.us/download/mjpro131_osx.zip instead, and copy the `.dylib` files instead of `.so` files.
```
mkdir -p /tmp/mujoco_tmp && cd /tmp/mujoco_tmp
wget -P . https://www.roboti.us/download/mjpro131_linux.zip
unzip mjpro131_linux.zip
mkdir <installation_path_of_your_choice>/rllab/vendor/mujoco
cp ./mjpro131/bin/libmujoco131.so <installation_path_of_your_choice>/rllab/vendor/mujoco
cp ./mjpro131/bin/libglfw.so.3 <installation_path_of_your_choice>/rllab/vendor/mujoco
cd ..
rm -rf /tmp/mujoco_tmp
```

3. Copy your Mujoco license key (mjkey.txt) to rllab path:
```
cp <mujoco_key_folder>/mjkey.txt <installation_path_of_your_choice>/rllab/vendor/mujoco
```

4. Clone sac
```
cd <installation_path_of_your_choice>
git clone https://github.com/haarnoja/sac.git
cd sac
```

5. Create and activate conda environment
```
cd sac
conda env create -f environment.yml
source activate sac
```

The environment should be ready to run. See examples section for examples of how to train and simulate the agents.

Finally, to deactivate and remove the conda environment:
```
source deactivate
conda remove --name sac --all
```

## Examples
### Training and simulating an agent
1. To train the agent
```
python ./examples/mujoco_all_sac.py --env=swimmer --log_dir="/root/sac/data/swimmer-experiment"
```

2. To simulate the agent (*NOTE*: This step currently fails with the Docker installation, due to missing display.)
```
python ./scripts/sim_policy.py /root/sac/data/swimmer-experiment/itr_<iteration>.pkl
```

`mujoco_all_sac.py` contains several different environments and there are more example scripts available in the  `/examples` folder. For more information about the agents and configurations, run the scripts with `--help` flag. For example:
```
python ./examples/mujoco_all_sac.py --help
usage: mujoco_all_sac.py [-h]
                         [--env {ant,walker,swimmer,half-cheetah,humanoid,hopper}]
                         [--exp_name EXP_NAME] [--mode MODE]
                         [--log_dir LOG_DIR]
```

`mujoco_all_sac.py` contains several different environments and there are more example scripts available in the  `/examples` folder. For more information about the agents and configurations, run the scripts with `--help` flag. For example:
```
python ./examples/mujoco_all_sac.py --help
usage: mujoco_all_sac.py [-h]
                         [--env {ant,walker,swimmer,half-cheetah,humanoid,hopper}]
                         [--exp_name EXP_NAME] [--mode MODE]
                         [--log_dir LOG_DIR]
```

# Benchmark Results
Benchmark results for some of the OpenAI Gym v2 environments can be found [here](https://drive.google.com/open?id=1I0NUrAzU7wwJQiX_MSmr1LvshjDZ4gSh).

# Credits
The soft actor-critic algorithm was developed by Tuomas Haarnoja under the supervision of Prof. [Sergey Levine](https://people.eecs.berkeley.edu/~svlevine/) and Prof. [Pieter Abbeel](https://people.eecs.berkeley.edu/~pabbeel/) at UC Berkeley. Special thanks to [Vitchyr Pong](https://github.com/vitchyr), who wrote some parts of the code, and [Kristian Hartikainen](https://github.com/hartikainen) who helped testing, documenting, and polishing the code and streamlining the installation process. The work was supported by [Berkeley Deep Drive](https://deepdrive.berkeley.edu/).

# Reference
```
@article{haarnoja2017soft,
  title={Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor},
  author={Haarnoja, Tuomas and Zhou, Aurick and Abbeel, Pieter and Levine, Sergey},
  booktitle={Deep Reinforcement Learning Symposium},
  year={2017}
}
```
