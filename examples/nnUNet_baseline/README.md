# nnUNet baseline model

We chose [nnUNet](https://www.nature.com/articles/s41592-020-01008-z) as a model baseline for KiTS 2021 Challenge since 
it is well known as a framework for fast and effective
development of segmentation methods. Users with various backgrounds and expertise can use nnUNet out-of-the-box for
their custom 3D segmentation problem without much need for manual intervention. It's publicly available and can be
accessed via [MIC-DKFZ/nnUNet](https://github.com/MIC-DKFZ/nnUNet).

We do not expect the participants to use nnUNet for model development but strongly encourage to compare the performance of
their developed model to the nnUNet baseline.

A documentation on how to run nnUNet on a new dataset is
given [here](https://github.com/MIC-DKFZ/nnUNet#how-to-run-nnu-net-on-a-new-dataset). To simplify a number of the steps
for the participants of KiTS 2021 Challenge, here we highlight the steps needed to train nnUNet on the KiTS 2021 dataset.

**IMPORTANT: nnU-Net only works on Linux-based operating systems!**

Note that our nnU-Net baseline uses the majority voted segmentations as ground truth for training and does not make
use of the sampled segmentations. 

### nnUNet setup

Please follow the installation instructions [here](https://github.com/MIC-DKFZ/nnUNet#installation). Please install 
nnU-Net as an integrative framework (not via `pip install nnunet`).

Remember that all nnU-Net commands support the `-h` argument for displaying usage instructions!

### Dataset preparation

This section requires you to have downloaded the KiTS2021 dataset.

As nnUNet expects datasets in a structured format, you need to convert the dataset to be compatible with nnUNet. We
provide a script to do this as part of the nnU-Net repository: [Task135_KiTS2021.py](https://github.com/MIC-DKFZ/nnUNet/blob/master/nnunet/dataset_conversion/Task135_KiTS2021.py)

Please adapt this script to your system and simply execute it with python. This will convert the KiTS dataset into 
nnU-Net's data format.


### Experiment planning and preprocessing
In order to train the nnU-Net models all you need to do is run the standard nnU-Net steps:

The following command will extract the dataset fingerprint and based on that configure nnU-Net.
```console
nnUNet_plan_and_preprocess -t 135 -pl2d None -tl 4 -tf 2
```

`-pl2d None` makes nnU-Net ignore the 2D configuration which is unlikely to perform well on the KiTS task. You can 
remove this part if you would like to use the 2D model.

Setting `-tf 2` and `-tl 4` is necessary to keep RAM utilization low during preprocessing. The provided numbers work 
well with 64GB RAM. If you find yourself running out of memory or if the preprocessing gets stuck, consider setting 
these lower. If you have more RAM (and CPU cores), set them higher.

Running preprocessing will take a while - so sit back and relax!

### Model training
Once preprocessing is completed you can run the nnU-net configurations you would like to use as baselines. Note that 
we will be providing pretrained model weights shortly after the dataset freeze so that you don't have to train nnU-Net 
yourself! (TODO)

In nnU-Net, the default is to train each configuration via cross-validation. This is the setting we recommend you use 
as well, regardless of whether you use nnU-Net for your submission or not. Running cross-validation gives you the most 
stable estimate of model performance on the training set. To run training with nnU-Net, use the following command:

```console
nnUNet_train CONFIGURATION nnUNetTrainerV2 135 FOLD
```

`CONFIGURATION` is hereby the nnU-Net configuration you would like to use (`2d`, `3d_lowres`, `3d_fullres`, 
`3d_cascade_fullres`; remember that we do not have preprocessed data for `2d` because we used `-pl2d None` in 
`nnUNet_plan_and_preprocess`). Run this command 5 times for `FOLD` 0, 1, 2, 3 and 4. If have multiple GPUs you can run 
these simultaneously BUT you need to start one of the folds first and wait till it utilizes the GPU before starting 
the others (this has to do with unpacking the data for training).

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

### Choosing the best configuration

Once the models are trained, you can either choose manually which one you would like to use, or use the 
`nnUNet_find_best_configuration` command to automatically determine the best configuration. Since this command does not 
understand the KiTS2021 HECs, we recommend evaluating the different configurations manually with the 
evaluation scripts provided in the kits21 repository and selecting the best performing model based on that.

In order to evaluate a nnU-Net model with the kits21 repository you first need to gather the validation set 
predictions from the five folds into a single folder. These are located here:
`${RESULTS_FOLDER}/nnUNet/CONFIGURATION/Task135_KiTS21/TRAINERCLASS__PLANSIDENTIFIER/fold_X/validation_raw`
Note that we are using the `validation_raw` and not the `validation_raw_postprocessed` folder. That is because 
a) nnU-Net prostprocessing needs to be executed for the entire cross validation using `nnUNet_determine_postprocessing` 
(`validation_raw_postprocessed` is for development purposes only) and b) the nnU-Net postprocessing is not useful for 
KiTS2021 anyways so it can safely be omitted.

Once you have all validation set predictions of the desired nnU-Net run in one folder, double check that all 300 KiTS21 
training cases are present. Then run 

`python kits21/evaluation/evaluate_predictions.py FOLDER -num_processes XX`

(note that you need to have generated the sampled segmentations first, see [here](../../kits21/evaluation))

Once that is completed there will be a file in `FOLDER` with the kits metrics. 

### Inference

For running inference on all images in a specific folder you can either make use of the scripts
prepared for docker submission or run `nnUNet_predict` command: 
```console
nnUNet_predict -i INPUT_FOLDER -o OUTPUT_FOLDER -t 135 -m 3d_fullres 
```

IMPORTANT: When using `nnUNet_predict`, nnU-Net expects the filenames in the input folder to end with _XXXX.nii.gz 
where _XXXX is a modality 
identifier. For KiTS there is just one modality (CT) so the files need to end with _0000.nii.gz 
(example: case_00036_0000.nii.gz). This is not needed when using the scripts in the nnU-Net docker examples!

## Updating the KiTS21 dataset within nnU-Net

The datset will be finalized by July 15th 2021. In order to update the dataset within nnU-Net you HAVE TO delete not 
only the content of `${nnUNet_raw_data_base}/nnUNet_raw_data` but also `${nnUNet_raw_data_base}/nnUNet_cropped_data` 
and `${nnUNet_preprocessed}/Task135_KiTS2021`. Then rerun the conversion script again, followed by
[experiment planning and preprocessing](#experiment-planning-and-preprocessing).

# nnU-Net baseline results
Pretrained model weights and predicted segmentation masks from the training set are provided here: https://zenodo.org/record/5126443
If you would like to use the pretrained weights, download the [Task135_KiTS21.zip](https://zenodo.org/record/5126443/files/Task135_KiTS2021.zip?download=1) 
file and import it with `nnUNet_install_pretrained_model_from_zip Task135_KiTS21.zip`.


Here are the results obtained with our nnU-Net baseline on the 300 training cases (5-fold cross-validation):

|                    | Dice_kidney | Dice_masses | Dice_tumor | Dice_average |   | SurfDice_kidney | SurfDice_masses | SurfDice_tumor | SurfDice_average |
|--------------------|-------------|-------------|------------|--------------|---|-----------------|-----------------|----------------|------------------|
| 3d_fullres         | 0.9666      | 0.8618      | 0.8493     | 0.8926       |   | 0.9336          | 0.7532          | 0.7371         | 0.8080           |
| 3d_lowres          | 0.9683      | 0.8702      | 0.8508     | 0.8964       |   | 0.9272          | 0.7507          | 0.7347         | 0.8042           |
| 3d_cascade_fullres | 0.9747      | 0.8799      | 0.8491     | 0.9012       |   | 0.9453          | 0.7714          | 0.7393         | 0.8187           |

As you can see, the `3d_cascade_fullres` configuration performed best, both in thers of average Dice score and average Surface Dice.

# Extending nnU-Net for KiTS2021

[Here](https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/extending_nnunet.md) are instructions on how to 
change and adapt nnU-Net. In order to keep things fair between participants **WE WILL NOT PROVIDE SUPPORT FOR IMPROVING 
nnU-Net FOR KITS2021**. You are on your own! 