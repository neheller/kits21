# Here is an example of a Dockerfile to use. Please make sure this file is placed to the same folder as inference_script.py file and directory model/ that contains your training weights.

FROM ubuntu:latest

# Install some basic utilities and python
RUN apt-get update \
  && apt-get install -y python3-pip python3-dev \
  && cd /usr/local/bin \
  && ln -s /usr/bin/python3 python \
  && pip3 install --upgrade pip

RUN pip3 install numpy nibabel

ADD model /model/
ADD inference_script.py ./

# run inference script that will populate /output folder
CMD ["inference_script.py"]
ENTRYPOINT ["python3"]

