# Here is an example of a Dockerfile to use. Please make sure this file is placed to the same folder as run_inference.py file and directory model/ that contains your training weights.

FROM ubuntu:latest

# Install some basic utilities and python
RUN apt-get update \
  && apt-get install -y python3-pip python3-dev \
  && cd /usr/local/bin \
  && ln -s /usr/bin/python3 python \
  && pip3 install --upgrade pip

RUN pip3 install numpy simpleitk nibabel

# Copy the folder with your pretrained model here to /model folder within the container. This part is skipped here due to simplicity reasons
# ADD model /model/

ADD run_inference.py ./

RUN groupadd -r myuser -g 433 && \
    useradd -u 431 -r -g myuser -s /sbin/nologin -c "Docker image user" myuser

RUN mkdir /input_nifti && mkdir /output_nifti && chown -R myuser /input_nifti && chown -R myuser /output_nifti

USER myuser

CMD python3 ./run_inference.py
