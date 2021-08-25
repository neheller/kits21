import shutil

if __name__ == '__main__':
    # this will be changed to /input for the docker
    input_folder = '/input'

    # this will be changed to /output for the docker
    output_folder = '/output'

    # this will be changed to /parameters/X for the docker
    parameter_folder_cascade_fullres = '/parameters/3d_cascade_fullres'
    parameter_folder_lowres = '/parameters/3d_lowres'

    from nnunet.inference.predict import predict_cases
    from batchgenerators.utilities.file_and_folder_operations import subfiles, join, maybe_mkdir_p

    input_files = subfiles(input_folder, suffix='.nii.gz', join=False)

    # in the parameters folder are five models (fold_X) traines as a cross-validation. We use them as an ensemble for
    # prediction
    folds_cascade_fullres = (0, 1, 2, 3, 4)
    folds_lowres = (0, 1, 2, 3, 4)

    # setting this to True will make nnU-Net use test time augmentation in the form of mirroring along all axes. This
    # will increase inference time a lot at small gain, so we turn that off here (you do whatever you want)
    do_tta = False

    # does inference with mixed precision. Same output, twice the speed on Turing and newer. It's free lunch!
    mixed_precision = True

    # This will make nnU-Net save the softmax probabilities. We need them for ensembling the configurations. Note
    # that ensembling the 5 folds of each configurationis done BEFORE saving the softmax probabilities
    save_npz = False  # no ensembling here

    # predict with 3d_lowres
    output_folder_lowres = join(output_folder, '3d_lowres')
    maybe_mkdir_p(output_folder_lowres)
    output_files_lowres = [join(output_folder_lowres, i) for i in input_files]

    predict_cases(parameter_folder_lowres, [[join(input_folder, i)] for i in input_files], output_files_lowres, folds_lowres,
                  save_npz=save_npz, num_threads_preprocessing=2, num_threads_nifti_save=2, segs_from_prev_stage=None,
                  do_tta=do_tta, mixed_precision=mixed_precision, overwrite_existing=True, all_in_gpu=False,
                  step_size=0.5)

    # predict with 3d_fullres
    output_folder_cascade_fullres = output_folder
    maybe_mkdir_p(output_folder_cascade_fullres)
    output_files_cascade_fullres = [join(output_folder_cascade_fullres, i) for i in input_files]

    # CAREFUL! I set all_in_gpu=True and step_size=0.75. These are not the defaults!
    predict_cases(parameter_folder_cascade_fullres, [[join(input_folder, i)] for i in input_files],
                  output_files_cascade_fullres, folds_cascade_fullres,
                  save_npz=save_npz, num_threads_preprocessing=2, num_threads_nifti_save=2,
                  segs_from_prev_stage=[join(output_folder_lowres, i) for i in input_files],
                  do_tta=do_tta, mixed_precision=mixed_precision, overwrite_existing=True, all_in_gpu=True,
                  step_size=0.75)

    # cleanup
    shutil.rmtree(output_folder_lowres)

    # done!

