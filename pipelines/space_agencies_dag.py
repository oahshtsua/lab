"""Extracts, stores and visualizes agencies data from TheSpaceDevs' Launch Library API."""

import json
import sqlite3

import airflow
import pandas as pd
from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator


def _transform_and_load():
    conn = sqlite3.connect("/tmp/agencies.db")

    with open("/tmp/agencies.json") as f:
        agencies = json.load(f)

    cur = conn.cursor()
    cur.execute(
        """CREATE TABLE IF NOT EXISTS space_agencies (
        id INTEGER,
        name TEXT,
        type TEXT,
        country_code TEXT)
        """
    )

    with conn:
        for agency in agencies["results"]:
            for country in agency["country_code"].split(","):
                cur.execute(
                    "INSERT INTO space_agencies VALUES (?, ?, ?, ?)",
                    (agency["id"], agency["name"], agency["type"], country),
                )
    conn.close()


def _visualize_share_by_agency_type():
    with sqlite3.connect("/tmp/agencies.db") as conn:
        agencies_df = pd.read_sql("SELECT * FROM space_agencies", conn)

    plot = (
        agencies_df["type"]
        .value_counts()
        .plot.pie(
            figsize=(10, 10),
            title="Types of Space Agencies",
            autopct="%1.0f%%",
            legend=True,
            ylabel="",
            labels=None,
        )
    )
    plot.get_figure().savefig("/tmp/share_by_agency_type.png")


def _visualize_share_by_country():
    with sqlite3.connect("/tmp/agencies.db") as conn:
        agencies_df = pd.read_sql("SELECT * FROM space_agencies", conn)

    plot = (
        agencies_df["country_code"]
        .value_counts()
        .plot.barh(
            figsize=(50, 50),
            title="Agencies Share by Country",
            ylabel="Country",
            xlabel="Number of Agencies",
        )
    )
    plot.get_figure().savefig("/tmp/share_by_country.png")


dag = DAG(
    dag_id="space_agencies",
    start_date=airflow.utils.dates.days_ago(0),
    schedule_interval=None,
)

download_agencies_data = BashOperator(
    task_id="download_agencies_data",
    bash_command="curl -o /tmp/agencies.json -L 'https://lldev.thespacedevs.com/2.2.0/agencies/?limit=300'",
    dag=dag,
)


transform_and_load = PythonOperator(
    task_id="transform_and_load",
    python_callable=_transform_and_load,
    dag=dag,
)

visualize_share_by_agency_type = PythonOperator(
    task_id="visualize_share_by_agency_type",
    python_callable=_visualize_share_by_agency_type,
    dag=dag,
)

visualize_share_by_country = PythonOperator(
    task_id="visualize_share_by_country",
    python_callable=_visualize_share_by_country,
    dag=dag,
)

tasks = [
    download_agencies_data,
    transform_and_load,
    [visualize_share_by_agency_type, visualize_share_by_country],
]
airflow.models.baseoperator.chain(*tasks)
