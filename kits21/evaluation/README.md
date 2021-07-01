# KiTS21 Evaluation
We refer to the [KiTS21](https://kits21.kits-challenge.org/) homepage for a detailed description of the metrics and 
ranking scheme used in the competition. This document only provides a brief overview.

## Evaluation metrics
KiTS21 uses two metrics for evaluation, the volumetric 
[Dice coefficient](https://en.wikipedia.org/wiki/S%C3%B8rensen%E2%80%93Dice_coefficient) and the 
[Surface Dice](https://arxiv.org/pdf/1809.04430.pdf). 

### Rationale
We refrain from using surface distance-based metrics such as the average symmetric surface distance or the 
Hausdorff distance because these measures penalize false positive/negative predictions by their distance, not their 
existence. While this property is useful for organ segmentation, it is not desired for tumor segmentation where 
lesions can be spread throughout the affected organ and thus do not have an expected spatial location. In addition, 
both Dice and SD have a predefined range (from 0 to 1 where 1 is best) which makes aggregation of these metrics through 
for example averaging more robust (we hereby refer to computing the average Dice or SD over cases, we are not averaging 
Dice and SD into a single number). ASSD and HD in contrast can have outlier values that, even if capped, can 
cause a single test case to dominate the performance of an algorithm (imagine having ASSD of ~1mm in most cases and 
a single case with very large ASSD (>100))).

## Ranking procedure
Compute Dice and Surface Dice for each test case and HEC, average over all test cases. This will give three Dice and three 
Surface Dice values for each algorithm (one per HEC). Then, all algorithms are ranked for these 6 values independently.
The winner will be determined as the algorithm with the lowest average rank.

## Code for metric computation
We provide a full implementation of the metric implementation in `kits21/evaluation/evaluate_predictions.py`. Please run

`python kits21/evaluation/evaluate_predictions.py -h`

to see usage instructions. Since we will be using this code to evaluate the test cases as well, you are 
encouraged to use it for evaluating your own train:validation splits during model development (we recommend running 
5-fold CV on the provided training cases).

## Computing inter-rater variability as a reference measure
Inter-rater varibility gives us an estimate of how close model performance is to human accuracy. We can use the 
multiple annotations per instance and label that we have to generate fictive plausible complete 
annotations which in turn allow us to estimate the inter-rater variability. Note that annotators are different for each 
instance and label, so annotator 1 of kidney instance 1 is not the same person as annotator 1 of kidney instance 2. 

When generating sampled segmentations with the intent of computing the inter-rater variability we cannot compare 
samples segmentations that have an overlap between their instance annotations. To illustrate this, we use a simple 
example that only has a kindey label (no tumor and cyst). We use `kidney_i1a1` as abbreviation for kidney instance 1 
annotation 1.

- computing the inter-rater variability between `kidney_i1a1_i2a1` and `kidney_i1a2_i2a2` is valid because for none of the 
  instances there are shared annotations
- computing the inter-rater variability between `kidney_i1a1_i2a1` and `kidney_i1a2_i2a1` is not valid because i2a1 was 
  used to construct both segmentations. This would result in an underestimation of the inter-rater variability because 
  parts of the segmentations perfectly overlap

To prevent underestimation of the inter-rater variability we generate 'groups' of sampled segmentations. Within each 
group, none of the annotations are shared and members of each group can be evaluated against each other (therefore 
each group has as many samples as there are annotations per instance). To get a more 
robust estimate of the inter-rater variability of a case, we generate multiple groups of sampled segmentations and 
average the inter-rater disagreement across groups. 

You can generate the groups yourself by running `python kits21/annotation/sample_segmentations.py`. Sampling is 
seeded to ensure that everyone uses the same samples.

## Finding the tolerance for SD
We follow the procedure described by the paper that introduces the Surface Dice.

[https://arxiv.org/pdf/1809.04430.pdf](https://arxiv.org/pdf/1809.04430.pdf) page 5:

> We defined the organ-specific tolerance as the 95th percentile of the distances collected across multiple 
> segmentations from a subset of seven TCIA scans, where each segmentation was performed by a radiographer and then 
> arbitrated by an oncologist, neither of whom had seen the scan previously.

We use the same groups of sampled segmentations as are used to compute the inter-rater variability for computing the 
tolerance. The tolerance is computed for each HEC individually and is averaged over all cases. The values are 
precomputed by us - you do not have to rerun the computation. You can find the values in `kits21/configuration/labels.py`. 
These values will automatically be used for metric computation.

If you still desire to rerun the computation of the tolerances, you can do so by running 
`python kits21/evaluation/compute_tolerances.py`. Note that this requires you to have generated the sampled 
segmentations first (see above).