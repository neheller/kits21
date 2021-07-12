# Guidelines for setting up and running docker

Here is BLABLABLA

###Step 1. Install Docker
To install docker use following instructions https://docs.docker.com/engine/install/ depending on your OS.

###Step 2. Creating Dockerfile
A good practice when using docker is to create a dockerfile with all needed requirements and needed operations. One can find an example of the docker file here in the dummy_submission folder. Please make sure this file is placed to the same folder as your python script to run inference on the test data INFERENCE_SCRIPT.py file and directory MODEL/ that contains your training weights.

###Step 3. Build a docker image from a dockerfile.
Navigate to the directory with the dockerfile and run following command:

```console
docker build -t YOUR_DOCKER_IMAGE_NAME .
```

###Step 4. Run a container from a created docker image.
A container is a process which runs on a host. To run this process "docker run" command has to be executed with specifying docker image to derive the container from. For this, one needs to run the following:
```console
nvidia-docker run --rm -it --runtime=nvidia -v LOCAL_PATH:/home --name YOUR_DOCKER_CONTAINER_NAME YOUR_DOCKER_IMAGE_NAME
```
-v flag mounts the directories between your local host and the container. Make sure that LOCAL_PATH folder has /input and /output folders in place, with /input folder constisting of test cases you would want to test on yourself. After the submission, this command will be run on a private server managed by the organizers with mounting to the local folder that has test data in the folder /input. 

###Step 5. Running inference script within the container
To run the script, one should execute following line **within the container**:
```console
python inference_script.py
```
###Spet 6. Save running container
To import or save running container to test_container.tar file on your local machine, one should run following command in a terminal:
```console
docker save -o test_container.tar YOUR_DOCKER_IMAGE_NAME
```
### (Optional) Step 7. Load the container 

To double check your saved container, one can load it with running:
```console
docker load -i test_container.tar
```
and run it again with command:
```console
sudo docker run --rm -it -v LOCAL_PATH:/home --name YOUR_DOCKER_CONTAINER_NAME YOUR_DOCKER_IMAGE_NAME
```