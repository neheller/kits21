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
    parser.add_argument('-num_processes', required=False, default=12, type=int,
                        help="Number of CPU cores to be used for evaluation. We recommend to use as many as your "
                             "System supports. Default: 12")
    parser.add_argument('--use_maj_voting_as_gt', required=False, action='store_true',
                        help="Set this flag to evaluate against the "
                             "majority voted segmentations (aggregated_MAJ_seg.nii.gz). This will be faster than "
                             "evaluating against the sampled segmentations, but keep in mind that this is not the way "
                             "the test set will be evaluated. For the test set evaluation we will be using the "
                             "samples as reference annotations.")
    args = parser.parse_args()
    if args.use_maj_voting_as_gt:
        evaluate_predictions(args.folder_with_predictions, args.num_processes)
    else:
        evaluate_predictions_on_samples(args.folder_with_predictions, args.num_processes)
