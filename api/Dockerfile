FROM pytorch/pytorch:2.1.2-cuda12.1-cudnn8-devel
WORKDIR /app

COPY api/requirements.txt .
RUN pip install -r requirements.txt


COPY api /app
COPY utils /app/utils
COPY NNModels /app/NNModels
COPY training /app/training
COPY dataset /app/dataset

EXPOSE 5000

CMD ["python", "api.py"]