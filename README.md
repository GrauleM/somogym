# SoMoGym

SoMoGym (**So**ft**Mo**tion **Gym**) is an open-source framework that builds on [SoMo](https://github.com/GrauleM/somo). SoMoGym facilitates the training and testing of soft robot control policies on complex tasks in which robots interact with their environments, frequently making and breaking contact. We provide a set of standard benchmark tasks, spanning challenges of manipulation, locomotion, and navigation in environments both with and without obstacles. A detailed description of the SoMoGym benchmark tasks and the baseline control policies we developed for these tasks can be found in the accompanying [RA-L paper](https://ieeexplore.ieee.org/abstract/document/9707663) ([PDF](/publications/somogym.pdf)). Interested users are also invited to watch our [video summary](/media/somogym_supplemental_video.mp4) to learn more.

## Codebase Overview

We define each task in its own SoMoGym "environment". Each SoMoGym environment is built upon the SomoEnv class, found in /environments/SomoEnv.py. SomoEnv inherits OpenAI Gym's base environment class, and is structured in the same manner. As a high level, it handles environment functionality that is general to every task: setting up the physics engine, making the OpenAI gym environment, observing state, stepping, and resetting. Each task's SoMoGym environment inherits SomoEnv, providing it with required task-specific parameters (i.e. action space) and augmenting it with any task-specific functionality (i.e. handling a box object).

There are a few instances of tasks that are only slightly different from another task (i.e. InHandManipulation and InHandManipulationInverted). In these cases, one of the tasks SoMoGym environments inherits from the other.

Each task has its own SoMoGym environment directory in the /environments folder. This directory includes definitions for entities in the physics engine, an init file for registering the environment (using OpenAI gym), the SoMoGym environment file (defining the environment class discussed above), and any additional util files required by the task. The Somogym environment directory also includes a benchmark_run_config.yaml file, which defines the standard settings for testing control policies on the environment.

We define all the SoMoGym environments we have released in the /environments folder. These environments are as follows:

- antipodal gripper
- in-hand manipulation
- inverted in-hand manipulation
- pen spinning
- pen spinning to a farther target
- planar block pushing
- planar reaching
- discrete snake locomotion


In /sample_policies, we define trajectories for each task. These trajectories can be run using /sample_policies/run_traj.py.

## Usage 

Run tasks with defined control trajectories using /sample_policies/run_traj.py. Not every trajectory will work on every environment, as action spaces are not constant across tasks. 

To train and evaluate RL policies on SoMoGym environments, check out [SoMo-RL](https://github.com/GrauleM/somo-rl/) (still in private beta).


## Installation
### Requirements
- [Python 3.8](https://www.python.org/downloads/release/python-3810/) +
- Tested on:
	- Ubuntu 20.04 with Python 3.8.10
- Recommended: pip (`sudo apt-get install python3-pip`) 
- Recommended (for Ubuntu): [venv](https://docs.python.org/3/library/venv.html) (`sudo apt-get install python3-venv`)

### Setup
0. Make sure your system meets the requirements
1. Clone this repository
2. Set up a dedicated virtual environment using `venv`
3. Activate virtual environment 
4. Upgrade pip if needed
5. Install requirements from this repo: `$ pip install -r requirements.txt`
6. Install this module:
    - either by cloning this repo to your machine and using `$  pip install -e .` from the repo root, or with
    - `$ pip install git+https://github.com/graulem/somogym`
7. To upgrade to the newest version: `$ pip install git+https://github.com/graulem/somogym --upgrade`


## Contributing

We're always happy for contributions, bug reports, feature requests, or other suggestions on how we can improve SoMoGym!

If you want to contribute, please keep the following in mind to make our lives easier:
- only through a new branch and reviewed PR (no pushes to master!)
- always use [Black](https://pypi.org/project/black/) for code formatting

### Testing
SoMoGym uses pytest for testing. In most cases, it will be better to ignore the tests that rely on the GUI -test coverage will be identical. You can run all tests with `$ pytest` from the repository's root and ignore the tests involving the GUI with `$ pytest -m "not gui"`

## Related Projects
Below is a list of papers and projects that make use of either SoMo or SoMoGym. Contact us to have your project added to this list.

- [An Active Palm Enhances Dexterity of Soft Robotic In-Hand Manipulation](https://ieeexplore.ieee.org/abstract/document/9562049)
- [The Role of Digit Arrangement in Soft Robotic In-Hand Manipulation](https://ieeexplore.ieee.org/abstract/document/9636188)


## Citation
If you find SoMoGym helpful in advancing your research, please cite our paper: 

> Graule, Moritz A., Thomas Mccarthy, Clark Teeple, Justin Werfel, and Robert Wood. "SoMoGym: A toolkit for developing and evaluating controllers and reinforcement learning algorithms for soft robots." IEEE Robotics and Automation Letters (2022).

## License
Code is provided "as is", without warranty of any kind. 

Copyright (c) 2021 Moritz A. Graule and Tom P. McCarthy 