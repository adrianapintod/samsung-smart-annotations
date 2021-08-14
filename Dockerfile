FROM python:3.6-slim

RUN apt-get update && apt-get upgrade -y

RUN apt-get install python3-opencv -y

RUN python3.6 -m pip install pipenv

WORKDIR /app/

COPY Pipfile Pipfile.lock ./

RUN pipenv install --system --deploy --ignore-pipfile

ADD https://pjreddie.com/media/files/yolov3.weights /app/

ADD http://www.cs.toronto.edu/polyrnn/models/checkpoints_cityscapes.tar.gz /app/models/

RUN cd /app/models/ && tar -xzf checkpoints_cityscapes.tar.gz && rm checkpoints_cityscapes.tar.gz

COPY . .

CMD ["/bin/bash"]
