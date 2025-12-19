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

This is where the code starts to branch across in a bunch of different places. The current setup has all the main things configured as needed with comments on where to make chanegs specific to your data. I will show an example for the light_switch task of the typical changes you would make for each new dataset you collect.

* Choose Dataset - in the `config/task/light_switch.yaml` make sure to set the `dataset_path` appropriately. If you are using a different action representation, change the `_target_` variable under `dataset`
* Configure Training Setup - the transformer policy needs to be adapted to include flow-based conversion for RTC so start with UNet before move on to that. Make sure the `task` variable under `defaults` matches the name of your task.yaml file. This is also the file where you set training hyperparameters like action horizon, batch size, number of epochs, number of action steps, etc.

Once you have configured all the variable appropriately, run this command to start training: `nohup python train.py --config-name=unet_workspace.yaml > train1.log 2>&1 &`
* Note that the use of nohup allows the command to run even while logged out.
* View the training metrics and graphs on wandb

### Inference

Once you're training is done, assess the graphs to identify the best checkpoint. Then, use scp to move the ckpt file over to the lab computer and save it under `Inference/Checkpoints'. There are two options you have to run inference:

* `inference.py` - The regular implementation of inference. Run it with `python inference.py -i Checkpoints/FILENAME -o Logs/LOGNAME`
* `inference_RTC.py` - Runs the same way but implements Real-Time Chunking. See this **document** for a detailed explanation on how it works.

To get the best results out of the policy, you will likely need to try out different values for these variables through trial and error:
* **num_inference_steps** -
* **n_action_steps** - reliable portion of the action horizon. For inference.py, this dictates how many actions from the prediction we actually end up sending to the inference script (happens in the policy file).
* **steps_per_inference** - determines how much of the action horizon you actually execute from every prediction (inference.py). This typically gets overwritten by RTC.
  * For example, if your horizon is set to 16 in training, the policy plans out a 16 step action trajectory. n_action_steps then truncates that to reliable actions (8-10), and steps_per_inference dictates how many we actually end up executing. All of this applies to inference.py because with RTC, we care about the whole horizon.
* **rtc_max_guidance** - 
* **OUTPUT_GAIN** - sometimes the trained policy can output appropriate actions but they need to be scaled up slightly to actually accomplish the task consistently. 
