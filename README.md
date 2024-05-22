# Physics-Informed Diffusion Models

## Description

This repository contains the scripts to reproduce the numerical experiments presented in the submitted preprint titled 'Physics-Informed Diffusion Models'.

It consists of three main scripts:

1) main_toy.py 
This script reproduces the toy study presented in Appendix F.1. It is helpful to understand the implications of the PIDM loss and several variants. Simply change the config file and run the script to reproduce the results or experiment with different parameters.

For the other scripts, you will first have to download the data and potentially the pretrained models. To do so, download and unzip the data provided at

and model checkpoints provided at 

and place them as follows:
```
.
├── data
│   ├── darcy
│   └── mechanics
│       └── ...
└── trained_models
    ├── darcy
    │   └── ...
    └── mechanics
        └── ...
```

After this, you can run the following scripts:

2) main.py
This script reproduces the Darcy flow and topology optimization study presented in Section 4. Simply adjust the parameters and governing equations in model.yaml and run the script to train the models. Note that the name of the run and logging parameters can be directly adjusted in main.py, if necessary.

3) sample.py
This script evaluates trained models. Provide the directory_path, name, and load_model_step of the model to evaluate and run the script. We  Note that the full evaluation of the in- and out-of-distribution test sets for the topology optimization study may take some time.

## Dependencies

The framework was developed and tested on Python 3.11 using CUDA 12.0.
To run the toy model, the following packages are required:
Package | Version (>=)

`pytorch`                   | `2.0.1`
`tqdm`                      | `4.65.0`
`matplotlib`                | `3.7.2`s
`imageio`                   | `2.28.1`
`einops`                    | `0.6.1`
`wandb` (optional)          | `0.15.2`

To run the Darcy flow and topology optimization study, the following additional packages are required:
Package | Version (>=)

`findiff`                   | `0.10.0`
`solidspy`                  | `1.0.16`
`pandas`                    | `2.1.3`
`einops-exts`               | `0.0.4`
`rotary_embedding_torch`    | `0.2.3`
`torchvision`               | `0.15.2`
`opencv`                    | `4.9.0.80`