#!/usr/bin/env bash
set -e

echo "[DEBUG] SELECTED_MODE: $SELECTED_MODE"

if [ "$SELECTED_MODE" = "remote_model" ]; then
    echo "[ENTRYPOINT] Initial Mode: Training remote model"
    cd /train_model
    exec python train_model/local_model.py
else
    echo "[ENTRYPOINT] Initial Mode: Training local model"
    cd /app
    exec python train_model/local_model.py
fi