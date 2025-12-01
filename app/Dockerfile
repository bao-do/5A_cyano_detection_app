FROM pytorch/pytorch:2.1.2-cuda12.1-cudnn8-devel
WORKDIR /app

COPY app/requirements.txt .
RUN pip install -r requirements.txt


COPY app /app
COPY exp /app/exp
COPY training /app/training
COPY dataset /app/dataset
COPY utils /app/utils
COPY NNModels /app/NNModels

EXPOSE 5000

CMD ["python", "app.py"]