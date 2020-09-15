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

This command will create a container called `bric_notebook` that will start the image `notebook`

```sh
docker rm bric_radiomics
docker run -d -p 3000:3000 --name bric_notebook notebook
```

# Pushing to Docker Repo

The docker repo can be found [here](https://hub.docker.com/repository/docker/neshdev/bric_radiomics)

```sh
docker login
>>username
>>password
docker tag notebook neshdev/notebook
docker push neshdev/bric_radiomics:tagname
```

# Debugging the docker build

We will open a shell of the last successful installation

```
docker exec -it bric_notebook /bin/bash
```

This will create a shell. You can use this shell to troubleshoot other things like installed packages, file system paths and etc...