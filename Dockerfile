FROM tensorflow/tensorflow:latest-gpu

WORKDIR /thesis

ADD requirements.txt /thesis

RUN pip install -r requirements.txt
RUN apt-get update && apt-get -y install git
RUN pip install git+https://github.com/MachinLeninIC/biograph.git@master

ADD src/ /thesis
ADD names_groups.pkl /thesis
ADD experiments /thesis/experiments

CMD bash
