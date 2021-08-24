FROM nvcr.io/nvidia/pytorch:21.07-py3

# Install some basic utilities and python
RUN apt-get update \
  && apt-get install -y python3-pip python3-dev \
  && cd /usr/local/bin \
  && ln -s /usr/bin/python3 python \
  && pip3 install --upgrade pip

# install nnunet
RUN pip install nnunet

# for single model inference
ADD parameters /parameters/
ADD run_inference.py ./

# for ensemble model inference
# ADD parameters_ensembling /parameters_ensembling/
# ADD run_inference_ensembling.py ./run_inference.py

ENV OMP_NUM_THREADS=1
