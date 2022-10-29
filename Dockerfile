FROM nvidia/cuda:11.0.3-devel-ubuntu18.04

RUN mkdir /app
WORKDIR /app
RUN apt-get update && apt-get install -y python3 python3-pip git libgl1-mesa-dev

COPY script/bashrc /etc/bash.bashrc
RUN chmod a+rwx /etc/bash.bashrc

COPY requirements.txt .
RUN python3 -m pip install --upgrade pip && \
    python3 -m pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu113

RUN mkdir /.local && chmod 777 /.local
RUN mkdir -p /root/.vscode-server/bin && chmod 777 /root/.vscode-server/bin
ENV PYTHONIOENCODING utf-8
ENV LC_ALL C.UTF-8
ENV LANG C.UTF-8

ARG USER_UID=1000
ARG USER_GID=$USER_UID
ARG USERNAME=vgg16
RUN groupadd --gid $USER_GID $USERNAME && \
 useradd -s /bin/bash --uid $USER_UID --gid $USER_GID -m $USERNAME

CMD ["python3", "-B", "main.py"]
