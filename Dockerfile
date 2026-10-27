# EMUstack Modernized Docker Environment (Python 3.11+, NGSolve/Netgen, Zero Fortran)
FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Install system dependencies including Gmsh for Path II geometry import
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    python3-venv \
    gmsh \
    libgl1 \
    libglu1-mesa \
    libxcursor1 \
    libxinerama1 \
    libxrandr2 \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /home/EMUstack
COPY ./ /home/EMUstack/

# Install Python packages and emustack-fem
RUN pip install --break-system-packages --no-cache-dir \
    numpy>=1.26 \
    scipy>=1.11 \
    matplotlib>=3.8 \
    ngsolve>=6.2.2600 \
    netgen-mesher>=6.2.2600 \
    xarray>=2023.1 \
    pyyaml>=6.0 \
    pytest>=8 \
    pytest-xdist \
    jupyterlab

RUN pip install --break-system-packages --no-build-isolation -e /home/EMUstack

# Concurrency limits for deterministic FEM eigenvalue solving
ENV OPENBLAS_NUM_THREADS=1
ENV OMP_NUM_THREADS=1
ENV MKL_NUM_THREADS=1
ENV NGS_NUM_THREADS=1

WORKDIR /home/EMUstack
EXPOSE 8888

CMD ["pytest", "tests/", "-v", "--tb=short"]
