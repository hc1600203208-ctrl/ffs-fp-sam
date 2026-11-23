FROM openeuler/openeuler:22.03-lts-sp3
RUN dnf install -y --setopt=install_weak_deps=False \
    gcc-c++ make cmake eigen3-devel libpng-devel zlib-devel python3-pyyaml \
    && dnf clean all
WORKDIR /src
