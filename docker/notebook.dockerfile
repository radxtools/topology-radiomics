# set base image (host OS)
FROM python:3.8

# install graphics libs
RUN apt-get update && apt-get install -y libglu1-mesa-dev freeglut3-dev mesa-common-dev

# set the working directory in the container
WORKDIR /notebooks

# copy the dependencies file to the working directory
COPY requirements.txt .
COPY .env-scripts/dev_requirements.txt .
COPY notebooks ./notebooks


# install dpendencies
RUN PIP_INSTALL="python -m pip install --upgrade --no-cache-dir --retries 10 --timeout 60" && \
    $PIP_INSTALL -r dev_requirements.txt && \
    $PIP_INSTALL install -r requirements.txt

COPY src ./src

# copy the content of the local src directory to the working directory
RUN cd src && python setup.py develop

# # command to run on container start
CMD [ "jupyter", "notebook", "--port=3000", "--no-browser", "--ip=0.0.0.0", "--allow-root", "--NotebookApp.token=" ]

EXPOSE 3000