# Submission examples

Please direct any questions or concerns about these instructions or the submission process generally to [the KiTS21 Discourse Forum](https://discourse.kits-challenge.org/).

## Submission guidelines

As the participants of KiTS 2021 challenge won't have access to the test dataset, the submission system will be managed through [Docker](https://www.docker.com/). The primary reason for that is to eliminate any possibility of cheating e.g. designing the model specifically for test dataset, or manually correcting test set predictions.

This year, the submission takes place by uploading a saved docker image (single file) containing your inference code to [our grand-challenge.org site](https://kits21.grand-challenge.org/). This image will be loaded on the evaluation system and executed on private servers. Naturally, these docker images **will NOT have access to the internet**, so please make sure everything you need it included in the image you upload.

- **input and output folders**:
  On our evaluation server, the docker will be executed such that an `/input/` (read only) and an `/output/` folder will be available at
  run time. The `/input/` folder contains the test set imaging. There are no subfolders -
  merely a bunch of `*.nii.gz` files containing the test images. Your docker is expected to produce equivalently
  named segmentation files (also ending with .nii.gz) in the `/output/` folder. The structure of those folders is shown
  below with the example of two cases:
  
      ├── input/
      │   └── case_00000.nii.gz
      │   └── case_00001.nii.gz
      └── output/
          └── case_00000.nii.gz
          └── case_00001.nii.gz

  Those folders will be mounted as volumes during docker run command (see [Step 4](#step-4-run-a-container-from-a-created-docker-image)).
- **Trained model**:
  Your trained model has to be part of the docker image and needs to have been added to the docker at the stage of building the image. Transferring parameter files is simply done by copying them to a specified folder within the container using the `ADD` command in the dockerfile.
  For more information, see the examples of the dockerfiles we prepared.
- **Python Inference Script**:
  The inference script for computing segmentation of test images must be a part of the submitted docker image as well. Hence, it has to be added at the stage of building a docker image. We ask you to name this script `run_inference.py`. This script is expected to read the test images from `/input/` and generate a matching segmentation (with matching filename) in `/output/`. How you achieve this is up to you - maximum freedom!

## Docker examples

This folder consist of 2 examples that can be used as a base for docker submission of the KiTS challenge 2021.

- The `dummy_submission` folder includes
  a simple [dockerfile](dummy_submission/Dockerfile)
  and simplistic inference
  script [run_inference.py](dummy_submission/run_inference.py)
  for computing dummy output segmentation (in current case its arrays filled with zeros).

- The `nnUNet_submission` folder has
  a [dockerfile](nnU-Net_baseline/Dockerfile) for
  running nnUNet baseline model along with 2 options: single model
  submission ([run_inference.py](nnUNet_submission/run_inference.py))
  and ensemble of the
  models ([run_inference_ensemble.py](nnUNet_submission/run_inference_ensembling.py))
  . Please note here, that to run the ensemble script locally, you need to change the naming of the parameters folder as
  well as the script to run (as outlines in the comments of
  the [dockerfile](nnUNet_submission/Dockerfile)).
  Your docker run command has to be adapted accordingly. For final submission, your inference script should be
  always called `run_inference.py`.

## Installation and running guidelines

We recognize that not all participants will have had experience with Docker, so we've prepared quick guidelines for
setting up a docker and using the submission examples. Here are the steps to follow to:

- Install docker
- Build a docker image
- Run a container
- Save and load a docker image created

### Step 1. Install Docker

To install docker use following instructions [https://docs.docker.com/engine/install/](https://docs.docker.com/engine/install/) depending on your OS.

### Step 2. Creating Dockerfile

A good practice when using docker is to create a dockerfile with all needed requirements and needed operations. You can
find a simple example of the dockerfile in
the [`dummy_submission/`](dummy_submission) folder.
More complicated example of a dockerfile can be found
in [`nnUNet_submission/`](nnUNet_submission) folder,
where we specified additional requirements needed for running the nnUNet baseline model. Please make sure that your
dockerfile is placed in the same folder as your python script to run inference on the test data
(*run_inference.py*) and directory that contains your training weights (`model/` folder for dummy example and `parameters/`
folder for nnUNet baseline example).

Please double check that the naming of your folder with a trained model is correctly specified in a dockerfile as well
as in the inference script.

### Step 3. Build a docker image from a dockerfile

Navigate to the directory with the dockerfile and run following command:

```console
docker build -t YOUR_DOCKER_IMAGE_NAME .
```

### Step 4. Run a container from a created docker image

To run a container the `docker run` command is used:

```console
docker run --rm --runtime=nvidia --ipc=host -v LOCAL_PATH_INPUT:/input:ro -v LOCAL_PATH_OUTPUT:/output YOUR_DOCKER_IMAGE_NAME python run_inference.py
```

`-v` flag mounts the directories between your local host and the container. `:ro` specifies that the folder mounted
with `-v` has read-only permissions. Make sure that `LOCAL_PATH_INPUT` contains your test samples,
and `LOCAL_PATH_OUTPUT` is an output folder for saving the predictions. During test set submission this command will
be run on a private server managed by the organizers with mounting to the folders with final test data. Please test
the docker on your local computer using the command above before uploading!

<!---
### (Optional) Step 5. Running script within the container
To run any additional scripts, you can execute the following line **within the container**:
```console
python run_inference.py
```
"""
-->

### Step 5. Save docker image container

To save your docker image to a file on your local machine, you can run the following command in a terminal:

```console
docker save -o test_docker.tar YOUR_DOCKER_IMAGE_NAME
```

This will create a file named `test_docker.tar` containing your image.

### Step 6. Load the image

To double check your saved image, you can load it with:

```console
docker load -i test_docker.tar
```

and run the loaded docker as outlined above with the following command (see Step 4):

```console
docker run --rm --runtime=nvidia --ipc=host -v LOCAL_PATH_INPUT:/input:ro -v LOCAL_PATH_OUTPUT:/output YOUR_DOCKER_IMAGE_NAME python run_inference.py
```
