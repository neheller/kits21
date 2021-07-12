"""
This python script should be able to load your pretrained model from the directory /model, access the test data from
/home/input/ path and save the predictions to the folder /home/output/
"""

import os
import numpy as np
import nibabel as nib

if __name__ == '__main__':

    if not os.path.exists('/home/output'):
        os.mkdir('/home/output')

    # Load model from /model folder. Here this part is skipped for simplicity reasons.

    for filename in os.listdir('/home/input'):
        if filename.endswith(".nii.gz"):
            img = nib.load(os.path.join('/home/input', filename))
            width, height, queue = img.dataobj.shape
            data = np.arange(width * height * queue).reshape(width, height, queue)
            img = nib.Nifti1Image(data, affine=np.eye(4))
            nib.save(img, os.path.join('/home/output', filename))
