FROM rocm/pytorch:rocm7.2_ubuntu24.04_py3.12_pytorch_release_2.9.1

ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /workspace

# Minimal runtime utilities + tini
RUN apt-get update && apt-get install -y --no-install-recommends \
      git \
      ca-certificates \
      curl \
      tini \
    && rm -rf /var/lib/apt/lists/*

# ---- Install Python requirements at BUILD time (fast boots) ----
# We only copy the requirement file(s) needed for dependency install,
# NOT the full repo (models/etc stay on host via bind mount).
COPY requirements.txt /tmp/requirements.txt

RUN python -m pip install --upgrade pip \
 && pip install --no-cache-dir -r /tmp/requirements.txt

# Copy entrypoint (small, stable)
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

EXPOSE 7860

ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["/entrypoint.sh"]
