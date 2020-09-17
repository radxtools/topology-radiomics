# Getting started with topology radiomics

## Installing using pip

You can find our package on pypi

Run the below command to install the package:

```
pip install topology_radiomics
```

## Running with docker

First clone this repository

```
git clone https://github.com/Toth-Technology/bric-morphology.git
```

There are multiple ways to get started.

1. docker-compose
2. docker run

### docker-compose

Run the following commands to start the docker container

```
cd docker
docker-compose up
```

### docker run

```
docker rm neshdev/topology_radiomics_notebook
docker pull neshdev/topology_radiomics_notebook
docker run -d -p 3000:3000 --name topology_radiomics_notebook neshdev/topology_radiomics_notebook
```

## Tutorials

Once the docker image is up and running. You can view our notebooks. You can get started with the notebook to learn how to use the package. You should start with `Tutorial - Getting started with topology_radiomics.ipynb`

Tutorial Notebooks:

1. Tutorial - Getting started with topoplogy_radiomics.ipynb
2. Tutorial - Using topology_radiomics to visualize features.ipynb
3. Tutorial - Working with medpy and topology_radiomics.ipynb


# Contribution Guide:

Please follow google style formatting for [docstrings](https://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings)

## Bugs and Feature Request

Please submit bugs and features to our github page.


## Pull Requests
Create a issue on our board.
Create a pull request with your changes. Tag your changes with the issue number (commit message should have issue number).
Someone from the team will review your request and merge your changes for the next release.