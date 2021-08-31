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

INPUT_FOLDER = '/input/images/ct/'
OUTPUT_FOLDER = '/output/images/kidney-tumor-and-cyst/'


def prep_output_folder():
    outpth = Path(OUTPUT_FOLDER)
    outpth.mkdir(parents=True, exist_ok=True)


def load_file_as_nifti(filename):
    reader = sitk.ImageFileReader()
    reader.SetImageIO("MetaImageIO")
    reader.SetFileName(os.path.join(INPUT_FOLDER, filename))
    image = reader.Execute()
    nda = sitk.GetArrayFromImage(image)
    mha_meta = {
        "origin": image.GetOrigin(),
        "spacing": image.GetSpacing(),
        "direction": image.GetDirection(),
        "filename": filename
    }

    affine = np.array(
       [[0.0, 0.0, -1*mha_meta["spacing"][2], 0.0],
        [0.0, -1*mha_meta["spacing"][1], 0.0, 0.0],
        [-1*mha_meta["spacing"][0], 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0]]
    )

    return nib.Nifti1Image(nda, affine), mha_meta


def save_nifti(segmentation_nib, mha_meta):
    dummy_segmentation = sitk.GetImageFromArray(np.asanyarray(segmentation_nib.dataobj))
    dummy_segmentation.SetOrigin(mha_meta["origin"])
    dummy_segmentation.SetSpacing(mha_meta["spacing"])
    dummy_segmentation.SetDirection(mha_meta["direction"])

    writer = sitk.ImageFileWriter()
    writer.SetFileName(os.path.join(OUTPUT_FOLDER, mha_meta["filename"]))
    writer.Execute(dummy_segmentation)


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
    prep_output_folder()

    # =========================================================================
    # Load model from /model folder!
    # This part is skipped for simplicity reasons
    model = None
    # =========================================================================

    for filename in os.listdir(INPUT_FOLDER):
        if filename.endswith(".mha"):
            # Load mha as nifti using provided function
            image_nib, mha_meta = load_file_as_nifti(filename)

            # Run your prediction function
            segmentation_nib = predict(image_nib, model)

            # Save nifti as mha using provided function
            save_nifti(segmentation_nib, mha_meta)


if __name__ == '__main__':
    main()
