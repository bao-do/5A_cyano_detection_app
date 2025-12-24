FROM pytorch/pytorch:2.1.2-cuda12.1-cudnn8-devel
WORKDIR /app

COPY app/requirements.txt .
RUN pip install -r requirements.txt


COPY app /app

EXPOSE 5000

CMD ["python", "app.py"]