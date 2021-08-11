FROM ubuntu:20.04
RUN apt-get update
RUN apt-get upgrade -y
RUN apt-get install -y python3
RUN apt-get install -y python3-pip

SHELL [ "/bin/bash", "--login", "-c" ]

# Keeps Python from generating .pyc files in the container
ENV PYTHONDONTWRITEBYTECODE=1

# Turns off buffering for easier container logging
ENV PYTHONUNBUFFERED=1

# Install pip requirements
COPY requirements.txt .
RUN python3 -m pip install -r requirements.txt

#
COPY benchmark benchmark

# WORKDIR /

# CMD [ "python", "benchmark.py" ]