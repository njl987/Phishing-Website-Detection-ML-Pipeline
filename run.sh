#!/bin/bash
set -e
echo "--- Starting AIAP 22 Pipeline Execution via bash ---"

# 1. Download database using curl (more portable than wget)
echo "Downloading database from source: https://techassessment.blob.core.windows.net/aiap22-assessment-data/phishing.db"
mkdir -p data
curl -L -o data/phishing.db "https://techassessment.blob.core.windows.net/aiap22-assessment-data/phishing.db"

# 2. Execute Python pipeline
echo "Executing pipeline: Preprocessing, Training, and Evaluation..."
PYTHONPATH=. python src/main_pipeline.py

echo "--- Pipeline Execution Complete ---"
