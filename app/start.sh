#!/bin/bash

#cd "$(dirname "$0")"
#echo "$(pwd)"

sudo service nginx restart

#python3 -m uvicorn app:app --reload --host 0.0.0.0 --log-level "info"
#uvicorn app:app --reload --host 0.0.0.0 --log-level info --forwarded-allow-ips='*' --uds='/tmp/uvicorn.sock' --proxy-headers
python3 -m uvicorn app.app:app --reload --host 0.0.0.0 --log-level info --forwarded-allow-ips='*' --proxy-headers