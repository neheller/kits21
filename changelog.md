# Changelog

This file is meant to keep track of which annotations were changed in each dataset version. Broadly, the *major* version will be incremented with any change to the postprocessing code that causes every region to be regenerated (starting at `v1.x.x`). The *minor* version will be incremented with the addition of new **imaging**, and the *patch* version will be incremented with new **annotations** for the existing imaging.

## [1.0.3] - May 21, 2021

- kits21 is now a python package. Install with `pip install -e .`
- metric computation
- delta value computation
- sampling of valid segmentations from the raw annotations (WIP)

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
