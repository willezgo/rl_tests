FROM pytorch/pytorch:1.9.1-cuda11.1-cudnn8-runtime
# FROM tensorflow/tensorflow:1.14.0-gpu-py3

# For some reason, SB3 wants 0.19
RUN pip3 install gym==0.19
RUN pip3 install gym[box2d]==0.19
RUN pip3 install gym[atari]==0.19

RUN pip3 install stable-baselines3[extra]

RUN apt-get update
RUN apt-get install -y python-opengl 
RUN apt-get install -y python-opencv
RUN apt-get install -y ffmpeg
RUN apt-get install -y xvfb

RUN pip3 install highway-env

COPY . . 

#CMD ["ls"]
ENTRYPOINT ["sh", "entrypoint.sh"]
