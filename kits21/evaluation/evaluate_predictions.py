from kits21.evaluation.metrics import evaluate_predictions_on_samples, evaluate_predictions

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Runs the KiTS evaluation. You can use this code to evaluate your "
                                                 "own trainings. We will use this exact code to evaluate the test set "
                                                 "as well, so it's a good idea to use it for method development ;-)\n"
                                                 "The output of this command is going to be a csv file located in your "
                                                 "folder_with_predictions.")
    parser.add_argument('folder_with_predictions', type=str,
                        help='folder containing the predicted segmentations. The evaluation will not check whether all '
                             'predictions are present and just evaluate what is in that folder. It is your '
                             'responsibility to verify that. Predicted segmentations MUST be named case_XXXXX.nii.gz '
                             'where XXXXX is the case id, for example case_00005.nii.gz.')
    parser.add_argument('-num_processes', required=False, default=4, type=int,
                        help="Number of CPU cores to be used for evaluation. We recommend to use as many as your "
                             "System supports. Default: 4")
    parser.add_argument('--use_samples', required=False, action='store_true',
                        help="Set this flag to evaluate against sampled ground truth segmentations rather than the "
                             "majority voted segmentations (aggregated_MAJ_seg.nii.gz). Experimental. Will take much "
                             "longer to compute. You need to run kits21/annotation/sample_segmentations.py "
                             "first before you can set this flag. IMPORTANT: In case there was a dataset update you "
                             "need to rerun sample_segmentations!")
    args = parser.parse_args()
    if args.use_samples:
        evaluate_predictions_on_samples(args.folder_with_predictions, args.num_processes)
    else:
        evaluate_predictions(args.folder_with_predictions, args.num_processes)
