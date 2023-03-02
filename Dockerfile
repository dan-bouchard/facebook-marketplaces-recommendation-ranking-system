FROM python:3.10

WORKDIR /app

RUN apt-get update && \
    apt-get install \
    'ffmpeg'\
    'libsm6'\
    'libxext6'  -y && \
    pip install --upgrade pip

RUN pip install torch torchvision --extra-index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .

RUN pip install -r requirements.txt

COPY . .

EXPOSE 8080

CMD ["python", "api.py"]
