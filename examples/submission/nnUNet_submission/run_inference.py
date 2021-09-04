from pathlib import Path
import os

import numpy as np
import nibabel as nib
import SimpleITK as sitk


INPUT_NIFTI = '/input_nifti'
OUTPUT_NIFTI = '/output_nifti'
if not os.path.exists(INPUT_NIFTI):
    os.mkdir(INPUT_NIFTI)
if not os.path.exists(OUTPUT_NIFTI):
    os.mkdir(OUTPUT_NIFTI)


def _load_mha_as_nifti(filepath):
    reader = sitk.ImageFileReader()
    reader.SetImageIO("MetaImageIO")
    reader.SetFileName(str(filepath))
    image = reader.Execute()
    nda = np.moveaxis(sitk.GetArrayFromImage(image), -1, 0)
    mha_meta = {
        "origin": image.GetOrigin(),
        "spacing": image.GetSpacing(),
        "direction": image.GetDirection(),
        "filename": filepath.name
    }

    affine = np.array(
       [[0.0, 0.0, -1*mha_meta["spacing"][2], 0.0],
        [0.0, -1*mha_meta["spacing"][1], 0.0, 0.0],
        [-1*mha_meta["spacing"][0], 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0]]
    )

    return nib.Nifti1Image(nda, affine), mha_meta


def _save_mha(segmentation_nib, mha_meta):
    output_mha = '/output/images/kidney-tumor-and-cyst/'
    if not Path(output_mha).exists():
        Path(output_mha).mkdir(parents=True)

    channels_last = np.moveaxis(np.asanyarray(segmentation_nib.dataobj), 0, -1)

    dummy_segmentation = sitk.GetImageFromArray(channels_last)
    dummy_segmentation.SetOrigin(mha_meta["origin"])
    dummy_segmentation.SetSpacing(mha_meta["spacing"])
    dummy_segmentation.SetDirection(mha_meta["direction"])

    writer = sitk.ImageFileWriter()
    writer.SetFileName(os.path.join(output_mha, mha_meta["filename"]))
    writer.Execute(dummy_segmentation)


def convert_imaging_to_nifti():
    input_mha = Path('/input/images/ct/')
    meta = {}
    for x in input_mha.glob("*.mha"):
        x_nii, x_meta = _load_mha_as_nifti(x)
        meta[x.stem] = x_meta
        nib.save(x_nii, str(Path(INPUT_NIFTI) / "{}.nii.gz".format(x.stem)))

    return meta


def convert_predictions_to_mha(conversion_meta):
    for uid in conversion_meta:
        pred_nii_pth = Path(OUTPUT_NIFTI) / "{}.nii.gz".format(uid)
        if not pred_nii_pth.exists():
            raise ValueError("No prediction found for file {}.mha".format(uid))
        pred_nii = nib.load(str(pred_nii_pth))
        _save_mha(pred_nii, conversion_meta[uid])


if __name__ == '__main__':
    """
    This inference script is intended to be used within a Docker container as part of the KiTS Test set submission. It
    expects to find input files (.nii.gz) in /input and will write the segmentation output to /output
    
    For testing purposes we set the paths to something local, but once we pack it in a docker we need to adapt them of 
    course
    
    IMPORTANT: This script performs inference using one nnU-net configuration (3d_lowres, 3d_fullres, 2d OR 
    3d_cascade_fullres). Within the /parameter folder, nnU-Net expects to find fold_X subfolders where X is the fold ID 
    (typically [0-4]). These folds CANNOT originate from different configurations. There also needs to be the plans.pkl 
    file that you find along with these fold_X folders in the 
    corresponding nnunet training output directory.
    
    /parameters/
    ├── fold_0
    │    ├── model_final_checkpoint.model
    │    └── model_final_checkpoint.model.pkl
    ├── fold_1
    ├── ...
    ├── plans.pkl
    
    Note: nnU-Net will read the correct nnU-Net trainer class from the plans.pkl file. Thus there is no need to 
    specify it here.
    """

    # This converts the mha files to nifti just like the official GitHub
    conversion_meta = convert_imaging_to_nifti()

    # this will be changed to /input for the docker
    input_folder = INPUT_NIFTI

    # this will be changed to /output for the docker
    output_folder = OUTPUT_NIFTI
    outpth = Path(output_folder)
    outpth.mkdir(parents=True, exist_ok=True)

    # this will be changed to /parameters for the docker
    parameter_folder = '/parameters'

    from nnunet.inference.predict import predict_cases
    from batchgenerators.utilities.file_and_folder_operations import subfiles, join

    input_files = subfiles(input_folder, suffix='.mha', join=False)

    output_files = [join(output_folder, i) for i in input_files]
    input_files = [join(input_folder, i) for i in input_files]

    # in the parameters folder are five models (fold_X) traines as a cross-validation. We use them as an ensemble for
    # prediction
    folds = (0, 1, 2, 3, 4)

    # setting this to True will make nnU-Net use test time augmentation in the form of mirroring along all axes. This
    # will increase inference time a lot at small gain, so you can turn that off
    do_tta = True

    # does inference with mixed precision. Same output, twice the speed on Turing and newer. It's free lunch!
    mixed_precision = True

    predict_cases(parameter_folder, [[i] for i in input_files], output_files, folds, save_npz=False,
                  num_threads_preprocessing=2, num_threads_nifti_save=2, segs_from_prev_stage=None, do_tta=do_tta,
                  mixed_precision=mixed_precision, overwrite_existing=True, all_in_gpu=False, step_size=0.5)

    # This converts the nifti predictions to the expected output
    convert_predictions_to_mha(conversion_meta)

    # done!
    # (ignore the postprocessing warning!)
