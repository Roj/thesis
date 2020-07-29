FROM tensorflow/tensorflow:latest-gpu

WORKDIR /thesis

ENV http_proxy=http://proxy.fcen.uba.ar:8080
ENV https_proxy=http://proxy.fcen.uba.ar:8080

ADD requirements.txt /thesis

RUN pip install -r requirements.txt
RUN apt-get update && apt-get -y install git
RUN pip install git+https://github.com/MachinLeninIC/biograph.git@master

ADD src/ /thesis
ADD names_groups.pkl /thesis
ADD experiments /thesis/experiments

CMD python experimentexecutor.py
