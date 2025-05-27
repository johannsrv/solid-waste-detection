FROM pytorch/pytorch:2.2.0-cuda11.8-cudnn8-runtime

RUN pip install --upgrade pip

WORKDIR /app

COPY app .
COPY requirements.txt .

RUN pip install -r requirements.txt

CMD ["python", "__main__.py"]
