The below documentation explains how to create and run docker containers.

# Prerequisites:
Software Installation
1. Docker

# Build

This command will create the image using the file `docker/notebook.dockerfile` and will name the name `notebook`

```sh
docker build -f docker/notebook.dockerfile -t notebook .
```

# Run

This command will create a container called `collage_radiomics` that will start the image `notebook`

```sh
docker rm collage_radiomics
docker run -d -p 3000:3000 --name collage_radiomics notebook
```