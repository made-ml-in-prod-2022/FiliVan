import os
from datetime import timedelta

from airflow import DAG
from airflow.providers.docker.operators.docker import DockerOperator
from airflow.utils.dates import days_ago
from docker.types import Mount


SAVE_DIR_GENERATE = "/data/raw/{{ ds }}"
VOLUME = "C:/Users/Filivan/Desktop/DS/airflow_ml_dags/data"

default_args = {
    "owner": "airflow",
    "email": ["airflow@example.com"],
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}


with DAG(
    "data_generator",
    default_args=default_args,
    schedule_interval="@daily",
    start_date=days_ago(0),
) as dag:
    generate = DockerOperator(
        image="airflow-data-generate",
        command="--out_dir {}".format(SAVE_DIR_GENERATE),
        task_id="docker-airflow-generate",
        do_xcom_push=False,
        network_mode="bridge",
        mounts=[
            Mount(
                source=VOLUME,
                target="/data",
                type="bind",
            )
        ],
    )

    generate
