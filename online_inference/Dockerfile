FROM python:3.9-slim


RUN apt-get update \
    && apt-get install -y git \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* \
    && pip install --no-cache-dir fastapi uvicorn \
    && pip install --no-cache-dir "git+https://github.com/made-ml-in-prod-2022/FiliVan@homework1_last#egg=ml_project&subdirectory=ml_project" \
    && mkdir /home/app/

WORKDIR /home/app/
COPY app.py .


ENV PATH_TO_MODEL="https://drive.google.com/file/d/1oGfWKZVbueiiMZ1L2f_BqDv0oiIE3uH8/view?usp=sharing"

RUN pip install pytest
COPY test_app.py .

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]