# Submission examples

This folder consist of 2 examples that can be used as a base for docker submission of the KiTS challenge 2021.

- dummy_submission folder includes
  vanilla [dockerfile](https://github.com/trofimova/kits21/blob/master/examples/submission/dummy_submission/Dockerfile)
  and simplistic inference
  script [run_inference.py](https://github.com/trofimova/kits21/blob/master/examples/submission/dummy_submission/run_inference.py)
  for computing dummy output segmentation (in current case its arrays filled with zeros)
  .
- nnU-Net_baseline folder has
  a [dockerfile](https://github.com/trofimova/kits21/blob/master/examples/submission/nnU-Net_baseline/Dockerfile) for
  running nnUnet baseline model along with 2 options: single model
  submission ([run_inference.py](https://github.com/trofimova/kits21/blob/master/examples/submission/nnU-Net_baseline/run_inference.py))
  and ensemble of the
  models ([run_inference_ensemble.py](https://github.com/trofimova/kits21/blob/master/examples/submission/nnU-Net_baseline/run_inference_ensembling.py))
  . Please note here, that to run the ensemble script locally, one need to change the naming of the parameters folder as
  well as script to run (as it's given in the comments of
  the [dockerfile](https://github.com/trofimova/kits21/blob/master/examples/submission/nnU-Net_baseline/Dockerfile)).
  Your docker run commant has to be adapted accordingly. For final submission, however, your inference script should be
  always called *run_inference.py*.

We also prepared a quick guidelines for setting up a docker and using submission examples. Here are the steps to follow
to install docker, build a docker image, run a container, save and load a docker image created.

### Step 1. Install Docker

To install docker use following instructions https://docs.docker.com/engine/install/ depending on your OS.

### Step 2. Creating Dockerfile

A good practice when using docker is to create a dockerfile with all needed requirements and needed operations. One can
find a simple example of the dockerfile in the dummy_submission/ folder. More complicated example of a dockerfile can be
found in nnU-Net_baseline/ folder, where we specified additional requirements needed for running nnUnet baseline model.
Please make sure your dockerfile is placed to the same folder as your python script to run inference on the test data
(*run_inference.py* file as a dummy example) and directory that contains your training weights (model/ folder for dummy
example and parameters/ folder for nnUnet baseline example).

Please make sure that the naming of your folder with a trained model is correctly specified in Dockerfile as well as in
the inference script.

### Step 3. Build a docker image from a dockerfile.

Navigate to the directory with the dockerfile and run following command:

```console
docker build -t YOUR_DOCKER_IMAGE_NAME .
```

### Step 4. Run a container from a created docker image.

A container is a process which runs on a host. To run this process "docker run" command has to be executed with
specifying docker image to derive the container from. For this, one needs to run the following:

```console
docker run --rm -it --runtime=nvidia --ipc=host -v LOCAL_PATH_INPUT:/input:ro -v LOCAL_PATH_OUTPUT:/output YOUR_DOCKER_IMAGE_NAME python run_inference.py
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
docker run --rm -it --runtime=nvidia --ipc=host -v LOCAL_PATH_INPUT:/input:ro -v LOCAL_PATH_OUTPUT:/output YOUR_DOCKER_IMAGE_NAME python run_inference.py
```