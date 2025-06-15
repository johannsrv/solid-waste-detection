#!/usr/bin/env bash
set -e

sleep 5

echo "[INFO] List of video devices:"
v4l2-ctl --list-devices

echo "[INFO] Setting devices to 640x480 MJPG/30fps"
for dev in /dev/video*; do
    if v4l2-ctl -d "$dev" --get-fmt-video >/dev/null 2>&1; then
        echo "[CONFIG] $dev"
        v4l2-ctl -d "$dev" \
          --set-fmt-video=width=640,height=480,pixelformat=MJPG \
          --set-parm=30 \
          || echo "[WARN] Partial config on $dev"
    else
        echo "[SKIP] $dev no es dispositivo de captura"
    fi
done

echo "[DEBUG] SELECTED_MODE: $SELECTED_MODE"

if [ "$SELECTED_MODE" = "train" ]; then
    echo "[ENTRYPOINT] Iniciando MODO TRAINING"
    cd /train_model
    exec python __main__.py
else
    echo "[ENTRYPOINT] Iniciando MODO INFERENCE (Flask streaming)"
    cd /app
    exec python __main__.py 
fi