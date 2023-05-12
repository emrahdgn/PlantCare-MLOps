# Base image
FROM python:3.10-slim

# Install dependencies
WORKDIR /plantcare

# Set build arguments for AWS credentials
ARG AWS_ACCESS_KEY_ID
ARG AWS_SECRET_ACCESS_KEY

COPY setup.py setup.py
COPY requirements.txt requirements.txt
RUN apt-get update \
    && apt-get install -y --no-install-recommends git \
    && apt-get install -y --no-install-recommends gcc build-essential \
    && rm -rf /var/lib/apt/lists/* \
    && python3 -m pip install --upgrade pip setuptools wheel \
    && python3 -m pip install -e . --no-cache-dir \
    && python3 -m pip install protobuf==3.20.1 --no-cache-dir \
    && apt-get purge -y --auto-remove gcc build-essential

COPY plantcare plantcare
COPY app app
COPY config config
COPY outputs/model outputs/model


# Pull assets from S3
RUN dvc init --no-scm \ 
    && dvc remote add s3_model s3://plantcare-mlops/model_registry \
    && dvc pull -r s3_model \ 
    && find outputs/model/Experiments -name optimizer.pth -delete 


# Export ports
EXPOSE 8000


# Start app
ENTRYPOINT ["gunicorn", "-c", "app/gunicorn.py", "-k", "uvicorn.workers.UvicornWorker", "app.api:app"]


