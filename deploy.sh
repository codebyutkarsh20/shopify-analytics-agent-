#!/usr/bin/env bash
# ============================================
# Shopify Analytics Agent - Deploy Script
# Pushes code to Azure server and restarts
# ============================================
set -euo pipefail

# ---- Configuration ----
REMOTE_HOST="sagar-azure"          # SSH alias from ~/.ssh/config
REMOTE_DIR="/home/azureuser/shopify-analytics-agent"
APP_NAME="shopify-analytics-bot"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

log()  { echo -e "${GREEN}[DEPLOY]${NC} $1"; }
warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
err()  { echo -e "${RED}[ERROR]${NC} $1"; exit 1; }

# ---- Pre-flight checks ----
log "Running pre-flight checks..."

# Check SSH connectivity
ssh -o ConnectTimeout=5 "$REMOTE_HOST" "echo ok" > /dev/null 2>&1 \
    || err "Cannot reach $REMOTE_HOST. Check your SSH config (~/.ssh/config)."

# Check .env exists locally (don't deploy without it)
if [ ! -f .env ]; then
    warn "No local .env file found. Make sure .env exists on the server at $REMOTE_DIR/.env"
fi

log "Pre-flight checks passed."

# ---- Sync code to server ----
log "Syncing code to $REMOTE_HOST:$REMOTE_DIR ..."

rsync -avz --progress \
    --exclude '.git' \
    --exclude '.env' \
    --exclude '.env.local' \
    --exclude '.env.production' \
    --exclude 'venv/' \
    --exclude '.venv/' \
    --exclude '__pycache__' \
    --exclude '*.pyc' \
    --exclude 'data/' \
    --exclude 'logs/' \
    --exclude '.pytest_cache' \
    --exclude 'htmlcov/' \
    --exclude '.coverage' \
    --exclude '.idea/' \
    --exclude '.vscode/' \
    ./ "$REMOTE_HOST:$REMOTE_DIR/"

log "Code synced successfully."

# ---- Build and restart on server ----
log "Building and restarting on server..."

ssh "$REMOTE_HOST" bash -s <<'REMOTE_SCRIPT'
set -euo pipefail

cd /home/azureuser/shopify-analytics-agent

echo "[SERVER] Checking Docker..."
if ! command -v docker &> /dev/null; then
    echo "[SERVER] Docker not found. Installing..."
    curl -fsSL https://get.docker.com | sh
    sudo usermod -aG docker $USER
    echo "[SERVER] Docker installed. You may need to log out and back in, then re-run this script."
    exit 1
fi

# Ensure docker compose is available (v2 plugin or standalone)
if docker compose version &> /dev/null; then
    COMPOSE="docker compose"
elif command -v docker-compose &> /dev/null; then
    COMPOSE="docker-compose"
else
    echo "[SERVER] docker compose not found. Installing plugin..."
    sudo apt-get update && sudo apt-get install -y docker-compose-plugin
    COMPOSE="docker compose"
fi

echo "[SERVER] Checking for .env file..."
if [ ! -f .env ]; then
    echo "[SERVER] ERROR: .env file not found at $(pwd)/.env"
    echo "[SERVER] Create it from .env.example: cp .env.example .env && nano .env"
    exit 1
fi

echo "[SERVER] Building Docker image..."
$COMPOSE build --no-cache

echo "[SERVER] Stopping old container (if running)..."
$COMPOSE down --remove-orphans 2>/dev/null || true

echo "[SERVER] Starting bot..."
$COMPOSE up -d

echo "[SERVER] Waiting 5 seconds for startup..."
sleep 5

echo "[SERVER] Container status:"
$COMPOSE ps

echo "[SERVER] Recent logs:"
$COMPOSE logs --tail=20

echo "[SERVER] Deploy complete!"
REMOTE_SCRIPT

log "Deployment finished successfully!"
log "Monitor logs:  ssh $REMOTE_HOST 'cd $REMOTE_DIR && docker compose logs -f'"
log "Stop bot:      ssh $REMOTE_HOST 'cd $REMOTE_DIR && docker compose down'"
log "Restart bot:   ssh $REMOTE_HOST 'cd $REMOTE_DIR && docker compose restart'"
