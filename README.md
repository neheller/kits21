# KiTS21

The official repository of the 2021 Kidney and Kidney Tumor Segmentation Challenge

[Challenge Homepage](https://kits21.kits-challenge.org/) (draft)

## News

- **Mar 9, 2021**: A preliminary challenge homepage has been published at kits21.kits-challenge.org. You can keep tabs on the data annotation process there.
- **Mar 29, 2020**: A second edition of KiTS was accepted to be held in conjunction with MICCAI 2021 in Strasbourg! More information will be posted here and on the [discussion forum](https://discourse.kits-challenge.org/) when it becomes available.

## Usage

### Download

Start by cloning this repository, but note that **the imaging is not stored here**, it must be downloaded using one of the `get_imaging` scripts in the `starter_code` directory. Currently there are implementations in:

- **python3**: `python3 starter_code/get_imaging.py`
- **MATLAB**: `matlab starter_code/get_imaging.m`
- **bash**: `bash starter_code/get_imaging.sh`

If you would like to request another implementation of `get_imaging`, please [submit an issue](https://github.com/neheller/kits21/issues/new).

## Folder Structure

### `data/`

**NOTE** at present, no data has been imported yet, but the imaging for the first 300 training cases can still be retrieved using any of the `get_imaging` scripts.

```
data/
├── case_00000/
|   ├── raw/
|   ├── segmentations/
|   ├── aggregated_seg.nii.gz
|   └── imaging.nii.gz
├── case_00001/
|   ├── raw/
|   ├── segmentations/
|   ├── aggregated_seg.nii.gz
|   └── imaging.nii.gz
...
├── case_00209/
|   ├── raw/
|   ├── segmentations/
|   ├── aggregated_seg.nii.gz
|   └── imaging.nii.gz
└── clinical_data.json
```

This is different from [KiTS19](https://github.com/neheller/kits19) because unlike 2019, we now have multiple annotations per "instance" and multiple instances per region.

Consider the "kidney" label in a scan: most patients have two kidneys (i.e., two "instances" of kidney), and each instance was annotated by three independent people. That case's `segmentations/` we would thus have

- `kidney_instance-1_annotation-1.nii.gz`
- `kidney_instance-1_annotation-2.nii.gz`
- `kidney_instance-1_annotation-3.nii.gz`
- `kidney_instance-2_annotation-1.nii.gz`
- `kidney_instance-2_annotation-2.nii.gz`
- `kidney_instance-2_annotation-3.nii.gz`

along with similar collections for `ureter`, `artery`, `vein`, `cyst`, and `tumor` regions. The `aggregated_seg.nii.gz` file is a result of combining all of these files via simple voxelwise majority voting.

### `starter_code/`

This folder holds code snippets for viewing and manipulating the data. See [Usage](#Usage) for more information.

### `annotation/`

This folder contains code used to process and import data from the annotation platform. As a participant, there's no reason you should need to run this code, it's only meant to serve as a reference.

## Challenge Information

This challenge will feature significantly more data, several annotations per case, and a number of additional annotated regions. The accepted proposal can be found [on Zenodo](https://doi.org/10.5281/zenodo.3714971), but the most up-to-date information about the challenge can be found on [the KiTS21 homepage](https://kits21.kits-challenge.org/).
