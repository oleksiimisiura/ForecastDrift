"""Shared HTTP retry helper - Open-Meteo occasionally times out on individual
requests (observed in GitHub Actions runs), and without a retry a single
transient timeout aborts the whole daily collection run."""

import time

import requests


def get_with_retry(url, params, timeout, retries=3, backoff=10):
    last_exc = None
    for attempt in range(retries):
        try:
            return requests.get(url, params=params, timeout=timeout)
        except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as e:
            last_exc = e
            if attempt < retries - 1:
                time.sleep(backoff * (attempt + 1))
    raise last_exc
