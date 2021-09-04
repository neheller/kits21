"""
This python script is a dummy example of the inference script that populates output/ folder. For this example, the
loading of the model from the directory /model is not taking place and the output/ folder is populated with arrays
filled with zeros of the same size as the images in the input/ folder.
"""

import os
import numpy as np
from pathlib import Path

import SimpleITK as sitk
import nibabel as nib

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



# =========================================================================
# Replace this function with your inference code!
def predict(image_nib, model):
    # As a dummy submission, just predict random voxels
    width, height, queue = image_nib.shape
    data = np.round(np.random.uniform(low=-0.49, high=2.49, size=(width, height, queue))).astype(np.uint8)

    # Must return a Nifti1Image object
    return nib.Nifti1Image(data, image_nib.affine)
# =========================================================================


def main():
    # This converts the mha files to nifti just like the official GitHub
    conversion_meta = convert_imaging_to_nifti()

    # =========================================================================
    # Load model from /model folder!
    # This part is skipped for simplicity reasons
    model = None
    # =========================================================================

    for filename in os.listdir(INPUT_NIFTI):
        if filename.endswith(".nii.gz"):
            # Load mha as nifti using provided function
            image_nib = nib.load(os.path.join(INPUT_NIFTI, filename))

            # Run your prediction function
            segmentation_nib = predict(image_nib, model)

            # Save nifti just as you otherwise would
            nib.save(segmentation_nib, Path(OUTPUT_NIFTI) / "{}".format(filename))

    # This converts the nifti predictions to the expected output
    convert_predictions_to_mha(conversion_meta)


if __name__ == '__main__':
    main()
