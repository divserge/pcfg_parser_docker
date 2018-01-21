# Base image
FROM continuumio/anaconda

# Author information
MAINTAINER Sergei.Divakov@skoltech.ru

# Set a working directory
WORKDIR parser

# Install latex.
RUN apt-get update && apt-get install -y texlive-full

# Install dependencies
RUN conda install numpy cython
RUN apt-get install -y gfortran

# Install necessary libraries
RUN pip install numpy scipy matplotlib nltk ttpy

# Add necessary files. Good practice to do it at the end
# in order to avoid reinstallation of dependencies when files change
ADD code ./code
ADD data ./data
ADD tex ./tex
ADD run.sh ./

# Make run.sh executable
RUN chmod +x run.sh

VOLUME parser/results

CMD ./run.sh