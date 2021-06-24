# Changelog

This file is meant to keep track of which annotations were changed in each dataset version. Broadly, the *major* version will be incremented with any change to the postprocessing code that causes every region to be regenerated (starting at `v1.x.x`). The *minor* version will be incremented with the addition of new **imaging**, and the *patch* version will be incremented with new **annotations** for the existing imaging.

## [2.0.4] - June 24, 2021

- Full annotations for cases 226 - 249
- Complete revisions for
  - `case_00041`
  - `case_00075`
  - `case_00076`
  - `case_00077`
  - `case_00078`
  - `case_00079`
  - `case_00080`
  - `case_00081`

## [2.0.3] - June 23, 2021

- Full annotations for cases 200 - 225
- Incomplete revisions for
  - `case_00017`
  - `case_00019`
  - `case_00021`
  - `case_00023`
  - `case_00026`
  - `case_00033`
  - `case_00051`
  - `case_00056`
  - `case_00057`
  - `case_00058`
  - `case_00060`
  - `case_00065`
  - `case_00066`
  - `case_00068`
  - `case_00071`
  - `case_00150`
- Complete revisions for
  - `case_00016`
  - `case_00018`
  - `case_00020`
  - `case_00022`
  - `case_00024`
  - `case_00025`
  - `case_00027`
  - `case_00028`
  - `case_00029`
  - `case_00030`
  - `case_00031`
  - `case_00032`
  - `case_00034`
  - `case_00035`
  - `case_00036`
  - `case_00038`
  - `case_00039`
  - `case_00042`
  - `case_00050`
  - `case_00052`
  - `case_00053`
  - `case_00054`
  - `case_00055`
  - `case_00059`
  - `case_00061`
  - `case_00062`
  - `case_00064`
  - `case_00067`
  - `case_00069`
  - `case_00070`
  - `case_00072`
  - `case_00073`
  - `case_00074`

## [2.0.2] - June 22, 2021

- Full annotations for cases 150 - 199
- Incomplete revisions for
  - `case_00000`
  - `case_00001`
  - `case_00015`
  - `case_00100`
- Complete revisions for
  - cases 2 - 14
  - `case_00101`
  - `case_00102`

## [2.0.1] - June 18, 2021

- Full annotations for cases 93 - 149

## [2.0.0] - June 17, 2021

- Changed import and aggregation code to ignore ureter, artery, and vein regions
- Deleted existing ureter, artery, and vein instances that were segmented
- New annotations for cases 50 - 92

## [1.0.3] - May 21, 2021

- Full annotations for
  - `case_00036`
  - `case_00037`
  - `case_00038`
  - `case_00039`
  - `case_00041`
  - `case_00042`
  - `case_00044`
  - `case_00046`
- Added cleanup function to import script which deletes unused save files

## [1.0.2] - May 7, 2021

- Full annotations for
  - `case_00034`
  - `case_00035`

## [1.0.1] - May 6, 2021

- Full annotations for
  - `case_00030`
  - `case_00031`
  - `case_00032`

## [1.0.0] - May 5, 2021

- Full annotations for
  - `case_00008`
  - `case_00025`
  - `case_00026`
  - `case_00027`
  - `case_00028`
  - `case_00029`
- Two new methods for aggregation ("and" and "majority voting") and their associated files

## [0.0.8] - April 29, 2021

- Full annotations for
  - `case_00005`
  - `case_00023`
  - `case_00024`

## [0.0.7] - April 15, 2021

- Full annotations for
  - `case_00020`
  - `case_00021`
  - `case_00022`

## [0.0.6] - April 14, 2021

- Full annotations for
  - `case_00017`
  - `case_00018`
  - `case_00019`

## [0.0.5] - April 13, 2021

- Full annotations for
  - `case_00014`
  - `case_00015` sans one artery segmentation -- will include next time
  - `case_00016`

## [0.0.4] - April 12, 2021

- Full annotations for
  - `case_00011` - sans one kidney annotation -- will get on next round
  - `case_00012`
  - `case_00013`

## [0.0.3] - April 9, 2021

- Full annotations for
  - `case_00006`
  - `case_00007`
  - `case_00009`
  - `case_00010`

## [0.0.2] - April 8, 2021

- Full annotations for:
  - `case_00002`
  - `case_00003`
  - `case_00004`
- Added `pull_request_template.md`

## [0.0.1] - April 7, 2021

- Includes all imaging from the KiTS19 Challenge
- Preliminary postprocessing and aggregation -- still subject to change
- Full annotations for:
  - `case_00000`
  - `case_00001`
  - `case_00006`
- Partial annotations for
  - `case_00002`
