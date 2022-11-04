FROM ubuntu:18.04
ENV DEBIAN_FRONTEND noninteractive
RUN useradd -ms /bin/bash --uid 1000 jupyter\
 && apt update\
 && apt install -y software-properties-common\
 && add-apt-repository -y ppa:deadsnakes/ppa\
 && apt install -y python3.9-dev python3.9-distutils build-essential\
 && ln -s /usr/bin/python3.9 /usr/local/bin/python3\
 && apt install -y libxml2-dev libxslt-dev libpq-dev\
 && apt install -y curl zip unzip git\
 && curl https://bootstrap.pypa.io/get-pip.py | python3
RUN curl https://raw.githubusercontent.com/rusmux/dls-simpsons/main/requirements.txt > requirements.txt\
 && pip install -r requirements.txt\
 && pip cache purge\
 && rm -rf requirements.txt
