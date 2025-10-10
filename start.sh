#!/usr/bin/env sh
set -e

MODEL_PATH=${MODEL_PATH:-cnn_model.pth}

if [ ! -f "$MODEL_PATH" ] && [ -n "$MODEL_URL" ]; then
  echo "Downloading model from $MODEL_URL to $MODEL_PATH ..."
  curl -L "$MODEL_URL" -o "$MODEL_PATH"
fi

exec gunicorn --bind 0.0.0.0:${PORT:-5000} --workers ${WEB_CONCURRENCY:-2} --threads ${GTHREADS:-8} app:app
