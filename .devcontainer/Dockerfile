FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    python3-dev \
    git \
    wget \
    curl \
    vim \
    graphviz \
&& rm -rf /var/lib/apt/lists/*

RUN ln -sf /usr/bin/python3.10 /usr/bin/python && \
    ln -sf /usr/bin/python3.10 /usr/bin/python3

COPY .devcontainer/requirements.txt /tmp/requirements.txt
RUN pip install --upgrade pip && pip install -r /tmp/requirements.txt

# Matplotlib defaults (container-wide)
ENV MATPLOTLIBRC=/etc/matplotlib
ENV MPLCONFIGDIR=/tmp/matplotlib

RUN mkdir -p /etc/matplotlib /tmp/matplotlib && chmod -R 777 /tmp/matplotlib

COPY .devcontainer/matplotlibrc /etc/matplotlib/matplotlibrc


WORKDIR /workspace

CMD ["/bin/bash"]
