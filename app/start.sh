#!/bin/bash

cd "$(dirname "$0")"
#echo "$(pwd)"

sudo service nginx restart

uvicorn app:app --reload --host 0.0.0.0 --log-level "info" -uds "/tmp/uvicorn.sock"
#python3 -m uvicorn app:app --reload --host 0.0.0.0 --log-level "info"