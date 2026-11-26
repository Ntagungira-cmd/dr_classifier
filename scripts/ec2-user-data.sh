#!/bin/bash
# Bootstrap: Docker + Compose, clone repo, run the stack.
set -euxo pipefail
exec > >(tee -a /var/log/app-bootstrap.log) 2>&1

# 4 GB swap so TensorFlow builds/retrains don't OOM on 4 GB RAM
if [ ! -f /swapfile ]; then
  dd if=/dev/zero of=/swapfile bs=1M count=4096
  chmod 600 /swapfile && mkswap /swapfile && swapon /swapfile
  echo '/swapfile none swap defaults 0 0' >> /etc/fstab
fi

dnf install -y docker git
systemctl enable --now docker

PLUG=/usr/local/lib/docker/cli-plugins
mkdir -p "$PLUG"
curl -fsSL -o "$PLUG/docker-compose" https://github.com/docker/compose/releases/download/v2.29.7/docker-compose-linux-x86_64
curl -fsSL -o "$PLUG/docker-buildx" https://github.com/docker/buildx/releases/download/v0.17.1/buildx-v0.17.1.linux-amd64
chmod +x "$PLUG/docker-compose" "$PLUG/docker-buildx"

if [ ! -d /opt/app/.git ]; then
  git clone --depth 1 https://github.com/Ntagungira-cmd/dr_classifier.git /opt/app
fi
cd /opt/app

# Instance-only override: survive reboots / stop-start
cat > docker-compose.override.yml <<'EOF'
services:
  api:
    restart: unless-stopped
  nginx:
    restart: unless-stopped
  ui:
    restart: unless-stopped
EOF

docker compose build
docker compose up -d
echo "BOOTSTRAP_DONE"
