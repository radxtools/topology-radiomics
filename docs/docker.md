The below documentation explains how to create and run docker containers.

# Prerequisites:
Software Installation
1. Docker

# Build

This command will create the image using the file `docker/notebook.dockerfile` and will name the name image `notebook`

```sh
docker build -f docker/notebook.dockerfile -t notebook .
docker tag notebook neshdev/topology_radiomics_notebook
```

# Run

This command will create a container called `topology_radiomics_notebook` that will start the image `notebook`

```sh
docker rm topology_radiomics_notebook
docker run -d -p 3000:3000 --name topology_radiomics_notebook notebook
```

# Compose

```sh
cd docker
docker-compose up
```

# Pushing to Docker Repo

The docker repo can be found [here](https://hub.docker.com/repository/docker/neshdev/bric_radiomics)

```sh
docker login
>>username
>>password
docker tag notebook neshdev/topology_radiomics_notebook
docker push neshdev/topology_radiomics_notebook:latest
```

# Debugging the docker build

We will open a shell of the last successful installation

```
docker exec -it bric_notebook /bin/bash
```

This will create a shell. You can use this shell to troubleshoot other things like installed packages, file system paths and etc...