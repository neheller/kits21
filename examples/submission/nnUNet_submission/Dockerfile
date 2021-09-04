FROM nvcr.io/nvidia/pytorch:20.08-py3

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
# ADD run_inference_ensembling.py ./

RUN groupadd -r myuser -g 433 && \
    useradd -u 431 -r -g myuser -s /sbin/nologin -c "Docker image user" myuser

RUN mkdir /input_nifti && mkdir /output_nifti && chown -R myuser /input_nifti && chown -R myuser /output_nifti

USER myuser

CMD python3 ./run_inference.py
# or CMD python3 ./run_inference_ensembling.py