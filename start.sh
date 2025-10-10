#!/usr/bin/env sh
# Robust start script: optionally download model, but never fail the service

MODEL_PATH=${MODEL_PATH:-cnn_model.pth}

maybe_download() {
  URL="$1"
  DEST="$2"
  if [ -z "$URL" ]; then
    echo "MODEL_URL not provided; skipping download"
    return 0
  fi
  echo "Attempting to download model from $URL to $DEST ..."
  if command -v curl >/dev/null 2>&1; then
    curl -fL "$URL" -o "$DEST" || return 1
    return 0
  fi
  if command -v wget >/dev/null 2>&1; then
    wget -O "$DEST" "$URL" || return 1
    return 0
  fi
  # Fallback to Python stdlib
  python - <<PY
import sys, urllib.request
url = sys.argv[1]
dest = sys.argv[2]
try:
    with urllib.request.urlopen(url) as r, open(dest, 'wb') as f:
        f.write(r.read())
    print('Downloaded model to', dest)
except Exception as e:
    print('WARN: python download failed:', e)
    sys.exit(1)
PY
  if [ $? -ne 0 ]; then
    return 1
  fi
  return 0
}

if [ ! -f "$MODEL_PATH" ] && [ -n "$MODEL_URL" ]; then
  maybe_download "$MODEL_URL" "$MODEL_PATH" || echo "WARN: could not download model; starting app without it"
fi

exec gunicorn --bind 0.0.0.0:${PORT:-5000} --workers ${WEB_CONCURRENCY:-2} --threads ${GTHREADS:-8} app:app
