#!/bin/bash

URL="http://localhost:8000/predict"

for i in {1..3}
do
  curl -s -X POST "$URL" \
    -H "Content-Type: application/json" \
    -d '{"features":[1,63,"Male","Cleveland","typical angina",145,233,"TRUE","lv hypertrophy",150,"FALSE",2.3,"downsloping",0,"fixed defect",0]}'
  echo "Request $i sent"
  sleep 5
done
