# nnUNet baseline model

We chose nnUNet as a model baseline for KiTS 2021 Challenge as it's well known as a framework for fast and effective
development of segmentation methods. Users with various background and expertise level can use nnUNet out of the box for
their custom 3D segmentation problem without much need for manual intervention. It's publicly available and can be
accessed via [MIC-DKFZ/nnUNet](https://github.com/MIC-DKFZ/nnUNet).

We do not expect the participants to use it for model development but strongly encourage to compare the performance of
developed model to nnUNet baseline.

A documentation on how to run nnUNet on a new dataset is
given [here](https://github.com/MIC-DKFZ/nnUNet#how-to-run-nnu-net-on-a-new-dataset). To simplify a number of the steps
for the participants of KiTS 2021 Challenge, here we present a number of steps one needs to perform to train nnUNet
baseline model specifically on KiTS 2021 dataset:

### nnUNet setup

Please follow the instruction to set up nnUNet and some environment variables following the
instructions [here](https://github.com/MIC-DKFZ/nnUNet#installation).

### Dataset preparation

As nnUNet expects datasets in a structured format, one needs to convert the dataset to be compatible with nnUNet format.
For that one should
run [Task135_KiTS2021.py](https://github.com/MIC-DKFZ/nnUNet/blob/master/nnunet/dataset_conversion/Task135_KiTS2021.py)
file locally.

### Experiment planning and preprocessing

For nnUNet to extract a fingerprint of the dataset, one should run nnUNet_plan_and_preprocess function that would
estimate a dataset specific properties. This information would be used to create all possible configurations of nnUNet:

- 2d (2d UNet)
- 3d_fullres (operates on full resolution images)
- 3d_lowres (operates on downsampled images)
- 3d_cascade_fullres (creates a coarse segmentation map in downsampled images which is then refined by 3d_fullres)

You can run this step with a following command (number 135 is a task name here and would be set to this value for 
following commands as well:
```console
nnUNet_plan_and_preprocess -t 135 --verify_dataset_integrity
```

### Training of the model

nnUNet trains all possible configurations in 5-fold cross-validation. This enables nnU-Net to determine the 
postprocessing and ensembling on the training dataset. Training models is done with the nnUNet_train command. For 
3d_fullres training, one needs to run following command with FOLD being [0, 1, 2, 3, 4]:

```console
nnUNet_train 3d_fullres nnUNetTrainerV2 Task135_KiTS2021 FOLD
```
For 3d_lowres training, one needs to run following command with FOLD being [0, 1, 2, 3, 4]:

```console
nnUNet_train 3d_lowres nnUNetTrainerV2 Task135_KiTS2021 FOLD
```

For 3d_cascade_fullres training, one needs to run following command with FOLD being [0, 1, 2, 3, 4]:

```console
nnUNet_train 3d_cascade_fullres nnUNetTrainerV2CascadeFullRes Task135_KiTS2021 FOLD
```

The trained models will we writen to the RESULTS_FOLDER/nnUNet folder. Each training obtains an automatically generated 
output folder name:

### Best configuration

Once the models are trained, one can run nnUNet_find_best_configuration command to determine what configuration to use 
for test set prediction
```console
nnUNet_find_best_configuration -m 3d_fullres 3d_lowres 3d_cascade_fullres -t 135 
```

### Inference 