FROM ubuntu:20.04
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential cmake libeigen3-dev libpng-dev zlib1g-dev python3 python3-yaml \
    ca-certificates && rm -rf /var/lib/apt/lists/*
WORKDIR /src
