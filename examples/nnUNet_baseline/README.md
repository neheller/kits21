# nnUNet baseline model

We chose nnUNet as a model baseline for KiTS 2021 Challenge as it's well known as a framework for fast and effective
development of segmentation methods. Users with various background and expertise can use nnUNet out-of-the-box for
their custom 3D segmentation problem without much need for manual intervention. It's publicly available and can be
accessed via [MIC-DKFZ/nnUNet](https://github.com/MIC-DKFZ/nnUNet).

We do not expect the participants to use nnUNet for model development but strongly encourage to compare the performance of
their developed model to the nnUNet baseline.

A documentation on how to run nnUNet on a new dataset is
given [here](https://github.com/MIC-DKFZ/nnUNet#how-to-run-nnu-net-on-a-new-dataset). To simplify a number of the steps
for the participants of KiTS 2021 Challenge, here we highlight the steps needed to train nnUNet on the KiTS 2021 dataset:

**IMPORTANT: nnU-Net only works on Linux-based operating systems!**

### nnUNet setup

Please follow the installation instructions [here](https://github.com/MIC-DKFZ/nnUNet#installation). Please follow the 
instructions for installing nnU-Net as an integrative framework (not via `pip install nnunet`).

### Dataset preparation

This section requires you to have downloaded the KiTS2021 dataset already.

As nnUNet expects datasets in a structured format, you need to convert the dataset to be compatible with nnUNet. We
provide a script to to this: [Task135_KiTS2021.py](https://github.com/MIC-DKFZ/nnUNet/blob/master/nnunet/dataset_conversion/Task135_KiTS2021.py)

Please adapt this script to your system and simply execute it with python. This will convert the KiTS dataset into 
nnU-Net's data format.

### Experiment planning and preprocessing
In order to train the nnU-Net models all you need to do is run the standard nnU-Net steps:

The following command will extract the dataset fingerprint and based on that configure nnU-Net's configurations.
```console
nnUNet_plan_and_preprocess -t 135 -pl2d None -tl 4 -tf 2
```

`-pl2d None` makes nnU-Net ignore the 2D configuration which is unlikely to perform well on the KiTS task. You can 
omit this part if you would like to use it.

Setting `-tf 2` and `-tl 4` is necessary to keep RAM utilization low during preprocessing. The provided numbers work 
well with 64GB RAM. If you find yourself running out of memory or if the preprocessing gets stuck, consider setting 
these lower. If you have more RAM (and CPU cores), set them higher.

Running preprocessing will take a while - so sit back and relax!

### Training of the model
Once preprocessing is completed you can run the nnU-net configurations you would like to use as baselines. Note that 
we will be providing pretrained model weights shortly after the dataset freeze so that you don't have to train nnU-Net yourself!

In nnU-Net, the default is to train each configuration via cross-validation. This is the setting we recommend you use 
as well, regardless of whether you use nnU-Net for your submission or not. Running cross-validation gives you the most 
stable estimate of model performance on the training set. To run training, use the following command:

```console
nnUNet_train CONFIGURATION nnUNetTrainerV2 137 FOLD
```

`CONFIGURATION` is hereby the nnU-Net configuration you would like to use (`2d`, `3d_lowres`, `3d_fullres`, 
`3d_cascade_fullres`; remember that we do nto have preprocessed data for `2d` by default). Run this command 5 times 
for `FOLD` 0, 1, 2, 3, 4. If have multiple GPUs you can run these simultaneously BUT you need to start one of the folds 
first and wait till it utilizes the GPU before starting the others (this has to do with unpacking the data for training).

The trained models will be writen to the RESULTS_FOLDER/nnUNet folder. Each training obtains an automatically generated
output folder name. Here we give an example of output folder for 3d_fullres:

    RESULTS_FOLDER/nnUNet/
    ├── 3d_cascade_fullres
    ├── 3d_fullres
    │   └── Task135_KiTS2021
    │       └── nnUNetTrainerV2__nnUNetPlansv2.1
    │           ├── fold_0
    │           │   ├── debug.json
    │           │   ├── model_best.model
    │           │   ├── model_best.model.pkl
    │           │   ├── model_final_checkpoint.model
    │           │   ├── model_final_checkpoint.model.pkl
    │           │   ├── progress.png
    │           │   └── validation_raw
    │           │       ├── case_00002.nii.gz
    │           │       ├── case_00008.nii.gz
    │           │       ├── case_00012.nii.gz
    │           │       ├── case_00021.nii.gz
    │           │       ├── case_00022.nii.gz
    │           │       ├── case_00031.nii.gz
    │           │       ├── case_00034.nii.gz
    │           │       ├── case_00036.nii.gz
    │           │       ├── summary.json
    │           │       └── validation_args.json
    │           ├── fold_1
    │           ├── fold_2
    │           ├── fold_3
    │           └── fold_4
    └── 3d_lowres

Exactly this structure of those three folders (3d_fullres, 3d_lowres and 3d_cascade_fullres) is required for running 
inference script presented in the example
of [nnUNet docker submission](../submission/nnUNet_submission).

### Best configuration

Once the models are trained, you can either choose manually which one you would like to use, or use the 
`nnUNet_find_best_configuration` command to automatically determine the best configuration. Since this command does not 
understand the HECs, we recommend to evaluate the different configurations manually with the evaluation scripts 
provided in the kits21 repository and selecting the best performing model based on that.

Should you still with to use `nnUNet_find_best_configuration`, this is how you do it:
```console
nnUNet_find_best_configuration -m 3d_fullres 3d_lowres 3d_cascade_fullres -t 135 
```

Note: adapt the `-m` part to the configurations that you actually have trained!

### Inference

For running the predictions on specific folder you can either make use of the scripts
prepared for docker submission or run `nnUNet_predict` command: 
```console
nnUNet_predict -i INPUT_FOLDER -o OUTPUT_FOLDER -t 135 -m 3d_fullres 
```

IMPORTANT: nnU-Net expects the filenames in the input folder to end with _XXXX.nii.gz where _XXXX is a modality 
identifier. For KiTS there is just one modality (CT) so the files need to end with _0000.nii.gz

## Updating the KiTS21 dataset within nnU-Net

The datset will be finalized by July 15th 2021. In order to upate the dataset within nnU-Net you HAVE TO delete not 
only the content of `${nnUNet_raw_data_base}/nnUNet_raw_data` but also `${nnUNet_raw_data_base}/nnUNet_cropped_data` 
and `${nnUNet_preprocessed}/Task137_KiTS2021`. Then rerun the conversion script again, followed by
[experiment planning and preprocessing](#experiment-planning-and-preprocessing).

# Extending nnU-Net for KiTS2021

[Here](https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/extending_nnunet.md) are instructions on how to 
change and adapt nnU-Net. In order to keep things fair between participants **WE WILL NOT PROVIDE SUPPORT FOR IMPROVING 
nnU-Net FOR KITS2021**. You are on your own! 