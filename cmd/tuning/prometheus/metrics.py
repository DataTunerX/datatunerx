from typing import List, Dict
from datetime import datetime
from urllib.parse import urljoin
from .prometheus_pb2 import (
    WriteRequest,
    TimeSeries
)
import calendar
import logging
import requests
import snappy


def dt2ts(dt):
    """Converts a datetime object to UTC timestamp
    naive datetime will be considered UTC.
    """
    return calendar.timegm(dt.utctimetuple())


def write(address: str, series: List[TimeSeries]):
    write_request = WriteRequest()
    write_request.timeseries.extend(series)

    uncompressed = write_request.SerializeToString()
    compressed = snappy.compress(uncompressed)

    url = urljoin(address, "/api/v1/write")
    headers = {
        "Content-Encoding": "snappy",
        "Content-Type": "application/x-protobuf",
        "X-Prometheus-Remote-Write-Version": "0.1.0",
        "User-Agent": "metrics-worker"
    }
    try:
        response = requests.post(url, headers=headers, data=compressed)
        print(response)
    except Exception as e:
        print(e)


def export_train_metrics(address: str, metrics: Dict):
    series = TimeSeries()
    label = series.labels.add()
    label.name = "__name__"
    label.value = "train_metrics"

    label = series.labels.add()
    label.name = "uid"
    label.value = str(metrics["uid"])

    label = series.labels.add()
    label.name = "total_steps"
    label.value = str(metrics.get("total_steps", ""))

    label = series.labels.add()
    label.name = "current_steps"
    label.value = str(metrics.get("current_steps", ""))

    label = series.labels.add()
    label.name = "loss"
    label.value = str(metrics.get("loss", ""))

    label = series.labels.add()
    label.name = "learning_rate"
    label.value = str(metrics.get("learning_rate", ""))

    label = series.labels.add()
    label.name = "epoch"
    label.value = str(metrics.get("epoch", ""))

    sample = series.samples.add()
    sample.value = 1
    sample.timestamp = dt2ts(datetime.utcnow()) * 1000

    write(address, [series])


def export_eval_metrics(address: str, metrics: Dict):
    series = TimeSeries()
    label = series.labels.add()
    label.name = "__name__"
    label.value = "eval_metrics"

    label = series.labels.add()
    label.name = "uid"
    label.value = str(metrics["uid"])

    label = series.labels.add()
    label.name = "total_steps"
    label.value = str(metrics.get("total_steps", ""))

    label = series.labels.add()
    label.name = "current_steps"
    label.value = str(metrics.get("current_steps", ""))

    label = series.labels.add()
    label.name = "eval_loss"
    label.value = str(metrics.get("eval_loss", ""))

    label = series.labels.add()
    label.name = "eval_perplexity"
    label.value = str(metrics.get("eval_perplexity", ""))

    label = series.labels.add()
    label.name = "epoch"
    label.value = str(metrics.get("epoch", ""))

    sample = series.samples.add()
    sample.value = 1
    sample.timestamp = dt2ts(datetime.utcnow()) * 1000

    write(address, [series])


if __name__ == '__main__':
    train_metrics = {
        "uid": "1",
        "current_steps": 10,
        "total_steps": 84,
        "loss": 3.088,
        "learning_rate": 4.404761904761905e-05,
        "epoch": 0.71
    }
    export_train_metrics("http://10.33.1.10:30722", train_metrics)

    eval_metrics = {
        "uid": "1",
        "current_steps": 10,
        "total_steps": 84,
        "eval_loss": 3.088,
        "eval_perplexity": 4.404761904761905e-05,
        "epoch": 0.71
    }
    export_eval_metrics("http://10.33.1.10:30722", eval_metrics)
