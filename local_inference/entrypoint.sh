#!/usr/bin/env bash
# Wait for USB cameras to initialize
sleep 5

echo "[INFO] List of video devices:"
v4l2-ctl --list-devices

echo "[INFO] Setting devices to 640x480"
for dev in /dev/video*; do
    if v4l2-ctl -d $dev --get-fmt-video >/dev/null 2>&1; then
        echo "[CONFIG] Setting $dev a MJPG/640x480/30fps"
        v4l2-ctl -d $dev \
            --set-fmt-video=width=640,height=480,pixelformat=MJPG \
            --set-parm=30 || echo "[WARN] Partial configuration in $dev"
    else
        echo "[SKIP] $dev it is not a capture device"
    fi
done

exec "$@"