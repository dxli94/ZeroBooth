FROM nvcr.io/nvidia/pytorch:22.12-py3 

ENV DEBIAN_FRONTEND=noninteractive 

RUN apt-get update && apt-get install -y --no-install-recommends \
    locales \
    wget \
    build-essential \
    vim \
    htop \
    curl \
    git less ssh cmake \
    zip unzip gzip bzip2 \
    python3-tk gcc g++ libpq-dev

RUN apt -y install openssh-server openssh-client
RUN pip install salesforce-lavis

RUN pip install streamlit
RUN pip install streamlit-image-select

RUN pip install accelerate==0.12.0
RUN pip install diffusers

# COPY blip2_backend /app
RUN git clone https://github.com/dxli94/ZeroBooth.git /BLIP-Diffusion-Demo

WORKDIR /BLIP-Diffusion-Demo/BLIP2/CLIP
RUN pip install -e .

WORKDIR /BLIP-Diffusion-Demo/BLIP2/open_clip
RUN pip install -e .

WORKDIR /BLIP-Diffusion-Demo/deployment

ENV PYTHONPATH="${PYTHONPATH}:./:/BLIP-Diffusion-Demo:/BLIP-Diffusion-Demo/deployment"

RUN chmod +x /BLIP-Diffusion-Demo/deployment/entrypoint.sh
