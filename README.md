# NRC Diffusion Policy

This is an imitation learning pipeline for training Diffusion Policies, a simplified adaptation of the code from this [paper](https://diffusion-policy.cs.columbia.edu/) for NRC's robot setup and teleoperation devices. The code has support to collect data with both the Haply Inverse3 (Haptic Device) and Meta Quest (VR) in all 3 action representations (Delta, Absolute, and Relative). This document gives a brief overview of how to use the code.

## Environment Setup

The conda environment and everything is already set up on the lab computer so you can skip most of this section if you are able to activate the environment with `conda activate robodiff`. If you are setting this up for training on a new GPU server, you will need to set the environment up so clone the repository on the server and run these commands:

`curl -L -O "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh"` <br>
`bash Miniforge3-$9uname)-$(uname -m).sh` <br>
`export PATH="$HOME/miniforge3/bin:$PATH"`

We use Miniforge to speed up the installation process for all the libraries. It should be installed after these commands and you can confirm that with `mamba --version`. Then you can setup the environment:

`cd Environment` <br>
`mamba env create -f conda_environment_real.yaml` <br>
`source ~/miniforge3/etc/profile.d/conda.sh` <br>
`conda activate robodiff`

To run training eventually, you will have to login to wandb. Make sure to create an account and use `wandb login ACCESS_CODE`. Replace ACCESS_CODE with your own. 

### Data Collection

Whichever part of the pipeline you are working on, make sure to `cd` into the appropriate folder. There are two teleoperation modalities and each has it's own script. We primarily use the haptic device so I will focus on that but the VR script is very similar.

* To collect data, run `python collect_haply.py -o Collected_Data/taskName_numOfEpisodes`. Once the robot is calibrated and in it's home position, press C to start an episode. The robot will start moving once you enable it with the haptic device. When you have completed you're task, return to the proximity of the start position and press S to save the episode. Repeat this for all your episodes.
* Once you have collected your data, it should all show up under the Collected_Data folder. You can view the data by running `python read_zarr.py episodeNumber`, replacing episodeNumber with the one you want to view.

### Policy Training

Write Here

* Write Steps Here

### Inference

Write Here

* Write Steps Here
  
```
code blocks for commands
```

## Other Notes

Write Here

```
command to run if program contains helper info
```
