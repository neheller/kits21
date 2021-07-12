"""
This python script should be able to load your pretrained model from the directory /model, access the test data from
/home/input/ path and save the predictions to the folder /home/output/
"""

import os
import numpy as np

if not os.path.exists('/home/output'):
    os.mkdir('/home/output')

for filename in os.listdir('/home/input'):
    if filename.endswith(".nii.gz"):
        with open(os.path.join('/home/output/', filename.split('.')[0] + '.npy'), 'wb') as f:
            np.save(f, np.array([1, 2]))
