# Guidelines for setting up and running docker

Here is a quick guide on how to install docker, build a docker image and run a container with mounting your local data
for testing to the container folder.

### Step 1. Install Docker

To install docker use following instructions https://docs.docker.com/engine/install/ depending on your OS.

### Step 2. Creating Dockerfile

A good practice when using docker is to create a dockerfile with all needed requirements and needed operations. One can
find a simple example of the dockerfile in the dummy_submission/ folder. More complicated example of a dockerfile can be
found in nnU-Net_baseline/ folder, where we specified additional requirements needed for running nnUnet baseline model.
Please make sure your dockerfile is placed to the same folder as your python script to run inference on the test data
(inference_script.py file as a dummy example) and directory that contains your training weights (model/ folder for dummy
example and parameters/ folder for nnUnet baseline example).

### Step 3. Build a docker image from a dockerfile.

Navigate to the directory with the dockerfile and run following command:

```console
docker build -t YOUR_DOCKER_IMAGE_NAME .
```

### Step 4. Run a container from a created docker image.

A container is a process which runs on a host. To run this process "docker run" command has to be executed with
specifying docker image to derive the container from. For this, one needs to run the following:

```console
docker run --rm -it --ipc=host -v LOCAL_PATH_INPUT:/input -v LOCAL_PATH_OUTPUT:/output YOUR_DOCKER_IMAGE_NAME python inference_script.py
```

-v flag mounts the directories between your local host and the container. Make sure that LOCAL_PATH folder has /input
and /output folders in place, with /input folder consisting of test cases you would want to test on yourself. After the
submission, this command will be run on a private server managed by the organizers with mounting to the local folder
that has test data in the folder /input.

In case you want to have it running on gpu, please specify --runtime=nvidia:

```console  
docker run --rm -it --runtime=nvidia --ipc=host -v LOCAL_PATH_INPUT:/input -v LOCAL_PATH_OUTPUT:/output YOUR_DOCKER_IMAGE_NAME python nnUNet_inference_script.py      
```

<!---
### (Optional) Step 5. Running script within the container
To run any additional scripts, one can execute following line **within the container**:
```console
python inference_script.py
```
"""
-->

### Step 5. Save docker image container

To import or save docker image to test_docker.tar file on your local machine, one should run following command in a
terminal:

```console
docker save -o test_docker.tar YOUR_DOCKER_IMAGE_NAME
```

### Step 6. Load the container

To double check your saved container, one can load it with running:

```console
docker load -i test_docker.tar
```

And run loaded docker as previously with following command:

```console
docker run --rm -it --ipc=host -v LOCAL_PATH_INPUT:/input -v LOCAL_PATH_OUTPUT:/output YOUR_DOCKER_IMAGE_NAME python inference_script.py
```