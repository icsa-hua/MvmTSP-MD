From python:3.12-slim as python-base 

ENV DEBIAN_FRONTEND=noninteractive \ 
    VENV_PATH=/opt/venv

# Prevent Python from writing .pyc files and enable unbuffered logging
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install --no-install-recommends -y \
    glpk-utils \ 
    libglpk-dev \ 
    build-essential \ 
    libgdal-dev \ 
    gdal-bin \ 
    wget \ 
    curl \ 
    ffmpeg \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* \
    && rm -rf /tmp/*

ENV PATH="${VENV_PATH}/bin:${PATH}"

WORKDIR /workspace

CMD ["bash"]


