FROM pytorch/pytorch:2.1.1-cuda12.1-cudnn8-runtime
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    git wget curl nano ffmpeg libsm6 libxext6 && \
    rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip
RUN pip install ultralytics

WORKDIR /workspace

CMD ["/bin/bash"]
