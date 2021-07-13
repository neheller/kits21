# Submission examples

## Submission guidelines

As the participants of KiTS 2021 challenge won't have access to the test dataset, the submission system will be managed
through docker. The primary reasoning for that is to eliminate an occurrence of cheating e.g. designing the model
specifically for test set. This year the submission takes place by uploading a saved docker image (single file), that
would be loaded and run by organizers on a private servers. Uploaded docker images has to be able to access the folder
with test samples called /input, the folder /output for writing out the predictions, your trained model as well as the
inference python script.

- **input and output folders**:
  Within the docker container the images from the test dataset would be placed in the folder /input without additional
  nested folders. The file extension of the files in /input folder is ".nii.gz". The /output folder should be populated
  with computed segmentations saved to the filenames identical to the /input folder filenames. The structure of those
  folders is shown below with the example of two cases: \
  ── input\
  ├── case00000.nii.gz\
  └── case00001.nii.gz\
  ── output\
  ├── case00000.nii.gz\
  └── case00001.nii.gz \
  Those folders will be mounted as volumes during docker run command (see Step 4 of Installation and running guidelines).
- **trained model**:
  As your trained model has to be a part of submitted docker image, it has to be added at the stage of building a docker
  image. This is done by copying your local folder with the model weights to a specified folder within the container.
  For more information see the examples of the dockerfiles we prepared.
- **python inference script**:
  The inference script for computing segmentation of test images has to be a part of the submitted docker image as well.
  Hence, it has to be added at the stage of building a docker image. We ask to name this script *run_inference.py* as it
  will be executed by organizers while running a docker image. In this script, one can access the test images from the
  /input folder and predicted segmentation should be saved to the /output folder. The model should be loaded from the
  folder that was specified for copying of your model weights at the stage of a docker image build.

## Folder structure

This folder consist of 2 examples that can be used as a base for docker submission of the KiTS challenge 2021.

- dummy_submission folder includes
  vanilla [dockerfile](https://github.com/trofimova/kits21/blob/master/examples/submission/dummy_submission/Dockerfile)
  and simplistic inference
  script [run_inference.py](https://github.com/trofimova/kits21/blob/master/examples/submission/dummy_submission/run_inference.py)
  for computing dummy output segmentation (in current case its arrays filled with zeros).

- nnU-Net_baseline folder has
  a [dockerfile](https://github.com/trofimova/kits21/blob/master/examples/submission/nnU-Net_baseline/Dockerfile) for
  running nnUnet baseline model along with 2 options: single model
  submission ([run_inference.py](https://github.com/trofimova/kits21/blob/master/examples/submission/nnU-Net_baseline/run_inference.py))
  and ensemble of the
  models ([run_inference_ensemble.py](https://github.com/trofimova/kits21/blob/master/examples/submission/nnU-Net_baseline/run_inference_ensembling.py))
  . Please note here, that to run the ensemble script locally, one need to change the naming of the parameters folder as
  well as script to run (as it's given in the comments of
  the [dockerfile](https://github.com/trofimova/kits21/blob/master/examples/submission/nnU-Net_baseline/Dockerfile)).
  Your docker run command has to be adapted accordingly. For final submission, however, your inference script should be
  always called *run_inference.py*.

## Installation and running guidelines

We recognize that not all participants will have had experience with Docker, so we've prepared a quick guidelines for
setting up a docker and using submission examples. Here are the steps to follow to install docker, build a docker image,
run a container, save and load a docker image created.

### Step 1. Install Docker

To install docker use following instructions https://docs.docker.com/engine/install/ depending on your OS.

### Step 2. Creating Dockerfile

A good practice when using docker is to create a dockerfile with all needed requirements and needed operations. One can
find a simple example of the dockerfile in
the [dummy_submission/](https://github.com/trofimova/kits21/tree/master/examples/submission/dummy_submission) folder.
More complicated example of a dockerfile can be found
in [nnU-Net_baseline/](https://github.com/trofimova/kits21/tree/master/examples/submission/nnU-Net_baseline) folder,
where we specified additional requirements needed for running nnUnet baseline model. Please make sure that your
dockerfile is placed to the same folder as your python script to run inference on the test data
(*run_inference.py*) and directory that contains your training weights (model/ folder for dummy example and parameters/
folder for nnUnet baseline example).

Please double check that the naming of your folder with a trained model is correctly specified in a dockerfile as well
as in the inference script.

### Step 3. Build a docker image from a dockerfile.

Navigate to the directory with the dockerfile and run following command:

```console
docker build -t YOUR_DOCKER_IMAGE_NAME .
```

### Step 4. Run a container from a created docker image.

A container is a process which runs on a host. To run this process "docker run" command has to be executed with
specifying docker image to derive the container from. For this, one needs to run the following:

```console
docker run --rm --runtime=nvidia --ipc=host -v LOCAL_PATH_INPUT:/input:ro -v LOCAL_PATH_OUTPUT:/output YOUR_DOCKER_IMAGE_NAME python run_inference.py
```

-v flag mounts the directories between your local host and the container. :ro addition to the -v flag for input folder
specifies, that the input folder has read-only permissions. Make sure that LOCAL_PATH_INPUT contains your test samples,
and LOCAL_PATH_OUTPUT is an output folder for saving the predictions. This command will be run on a private server
managed by the organizers with mounting to the folders with final test data.

<!---
### (Optional) Step 5. Running script within the container
To run any additional scripts, one can execute following line **within the container**:
```console
python run_inference.py
```
"""
-->

### Step 5. Save docker image container

To save docker image to test_docker.tar file on your local machine, one should run following command in a terminal:

```console
docker save -o test_docker.tar YOUR_DOCKER_IMAGE_NAME
```

### Step 6. Load the image

To double check your saved image, one can load it with running:

```console
docker load -i test_docker.tar
```

And run loaded docker as previously with following command identical to the Step 4:

```console
docker run --rm --runtime=nvidia --ipc=host -v LOCAL_PATH_INPUT:/input:ro -v LOCAL_PATH_OUTPUT:/output YOUR_DOCKER_IMAGE_NAME python run_inference.py
```