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
    ├── fold_1
    ├── ...
    ├── plans.pkl
    
    Note: nnU-Net will read the correct nnU-Net trainer class from the plans.pkl file. Thus there is no need to 
    specify it here.
    """

    # this will be changed to /input for the docker
    input_folder = '/home/fabian/drives/E132-Projekte/Projects/2021_Isensee_Trofimova_KiTS_Challenge/input'

    # this will be changed to /output for the docker
    output_folder = '/home/fabian/drives/E132-Projekte/Projects/2021_Isensee_Trofimova_KiTS_Challenge/output'

    # this will be changed to /parameters for the docker
    parameter_folder = '/home/fabian/drives/E132-Projekte/Projects/2021_Isensee_Trofimova_KiTS_Challenge/parameters'

    from nnunet.inference.predict import predict_cases
    from batchgenerators.utilities.file_and_folder_operations import subfiles, join

    input_files = subfiles(input_folder, suffix='.nii.gz', join=False)

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

    # done!
    # (ignore the postprocessing warning!)
