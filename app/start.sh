#!/bin/bash

cd "$(dirname "$0")"
#echo "$(pwd)"

uvicorn app:app --reload --host 0.0.0.0 --log-level "info"