FROM python:3.6-slim

ENV PYTHONPATH=$PYTHONPATH:/app/src

RUN apt-get update && apt-get upgrade -y

RUN apt-get install python3-opencv -y

RUN python3.6 -m pip install pipenv

WORKDIR /app/

COPY Pipfile Pipfile.lock ./

RUN pipenv install --system --deploy --ignore-pipfile

ADD https://pjreddie.com/media/files/yolov3.weights ./src/checkpoints/yolo/

ADD https://www.cs.toronto.edu/polyrnn/models/checkpoints_cityscapes.tar.gz ./src/checkpoints/polyrnn/

RUN cd ./src/checkpoints/polyrnn/ && tar -xzf checkpoints_cityscapes.tar.gz && rm checkpoints_cityscapes.tar.gz && mv ./models/* ./ && rm -rf ./models 

COPY . ./

CMD ["python", "src/app.py"]
