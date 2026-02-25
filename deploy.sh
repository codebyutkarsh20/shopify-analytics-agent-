#!/usr/bin/env bash
# ============================================
# Shopify Analytics Agent - Deploy Script
# Pushes code to Azure server and restarts
#
# Usage:
#   ./deploy.sh              Auto-detect what changed and deploy accordingly
#   ./deploy.sh --restart    Just restart container (for .env changes on server)
#   ./deploy.sh --fresh      Full rebuild from scratch (nuclear option)
#   ./deploy.sh --fresh-db   Rebuild + wipe database & logs (clean slate)
#   ./deploy.sh --logs       Just show recent logs, no deploy
#   ./deploy.sh --status     Show container status and disk usage
#   ./deploy.sh --cleanup    Free disk space (prune all unused Docker data)
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
CYAN='\033[0;36m'
NC='\033[0m' # No Color

log()  { echo -e "${GREEN}[DEPLOY]${NC} $1"; }
warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
err()  { echo -e "${RED}[ERROR]${NC} $1"; exit 1; }
info() { echo -e "${CYAN}[INFO]${NC} $1"; }

MODE="${1:-deploy}"

# ---- Quick commands that don't need a full deploy ----

if [ "$MODE" = "--logs" ]; then
    log "Fetching recent logs..."
    ssh "$REMOTE_HOST" "cd $REMOTE_DIR && docker compose logs --tail=50"
    exit 0
fi

if [ "$MODE" = "--status" ]; then
    log "Fetching server status..."
    ssh "$REMOTE_HOST" bash -s <<'STATUS_SCRIPT'
echo "=== Container Status ==="
cd /home/azureuser/shopify-analytics-agent
docker compose ps
echo ""
echo "=== Container Health ==="
docker inspect --format='{{.State.Health.Status}}' shopify-analytics-bot 2>/dev/null || echo "No health status available"
echo ""
echo "=== Resource Usage ==="
docker stats shopify-analytics-bot --no-stream --format "CPU: {{.CPUPerc}}  Memory: {{.MemUsage}}  Net I/O: {{.NetIO}}" 2>/dev/null || echo "Container not running"
echo ""
echo "=== Disk Usage ==="
df -h / | tail -1 | awk '{print "Disk: " $3 " used / " $2 " total (" $5 " full)"}'
echo ""
echo "=== Docker Disk Usage ==="
docker system df
STATUS_SCRIPT
    exit 0
fi

if [ "$MODE" = "--cleanup" ]; then
    warn "This will remove ALL unused Docker images, containers, and build cache."
    read -p "Continue? (y/N) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        log "Cleaning up Docker resources on server..."
        ssh "$REMOTE_HOST" "docker system prune -a -f && docker builder prune -a -f"
        log "Cleanup complete!"
        ssh "$REMOTE_HOST" "df -h / | tail -1 | awk '{print \"Disk: \" \$3 \" used / \" \$2 \" total (\" \$5 \" full)\"}'"
    fi
    exit 0
fi

if [ "$MODE" = "--restart" ]; then
    log "Restarting container (no rebuild)..."
    ssh "$REMOTE_HOST" bash -s <<'RESTART_SCRIPT'
cd /home/azureuser/shopify-analytics-agent
docker compose down --remove-orphans 2>/dev/null || true
docker compose up -d
sleep 10
echo "[SERVER] Container status:"
docker compose ps
echo "[SERVER] Recent logs:"
docker compose logs --tail=20
RESTART_SCRIPT
    log "Restart complete!"
    exit 0
fi

if [ "$MODE" = "--fresh-db" ]; then
    warn "This will DELETE the bot database and logs, then rebuild and restart."
    warn "All learned templates, conversation history, and sessions will be lost."
    read -p "Continue? (y/N) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        log "Syncing latest code first..."
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

        log "Stopping container, wiping DB volumes, rebuilding..."
        ssh "$REMOTE_HOST" bash -s <<'FRESHDB_SCRIPT'
set -euo pipefail
cd /home/azureuser/shopify-analytics-agent

echo "[SERVER] Stopping container..."
docker compose down --remove-orphans 2>/dev/null || true

echo "[SERVER] Removing data and log volumes..."
docker volume rm shopify-bot-data 2>/dev/null || true
docker volume rm shopify-bot-logs 2>/dev/null || true

echo "[SERVER] Rebuilding image (fresh, no cache)..."
docker compose build --no-cache

echo "[SERVER] Starting bot with clean database..."
docker compose up -d

echo "[SERVER] Waiting 15 seconds for startup + seed templates..."
sleep 15

echo "[SERVER] Container status:"
docker compose ps

echo "[SERVER] Recent logs:"
docker compose logs --tail=30
FRESHDB_SCRIPT
        log "Fresh DB deploy complete! Seed templates will regenerate on first startup."
    fi
    exit 0
fi

# ---- Validate mode ----
if [ "$MODE" != "deploy" ] && [ "$MODE" != "--fresh" ]; then
    err "Unknown option: $MODE\nUsage: ./deploy.sh [--restart|--fresh|--fresh-db|--logs|--status|--cleanup]"
fi

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

# ---- Auto-detect what changed (for smart deploy) ----
if [ "$MODE" = "deploy" ]; then
    # Check what files changed in the last commit vs what's on server
    CHANGED_FILES=$(git diff --name-only HEAD~1 HEAD 2>/dev/null || echo "")

    if echo "$CHANGED_FILES" | grep -q "requirements.txt"; then
        info "Detected requirements.txt change → dependencies will be reinstalled"
        warn "This deploy will take longer (~5-10 min) to install new packages"
    fi

    if echo "$CHANGED_FILES" | grep -q "Dockerfile\|docker-compose.yml"; then
        info "Detected Dockerfile/docker-compose change → switching to fresh build"
        MODE="--fresh"
    fi

    if echo "$CHANGED_FILES" | grep -qE "^src/|^main.py$"; then
        info "Detected code changes → quick cached build"
    fi
fi

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

ssh "$REMOTE_HOST" bash -s -- "${MODE}" <<'REMOTE_SCRIPT'
set -euo pipefail

BUILD_MODE="${1:-deploy}"

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

# ---- Auto-prune: clean up dangling images only (preserves build cache) ----
echo "[SERVER] Cleaning up dangling Docker resources..."
docker image prune -f 2>/dev/null || true
docker builder prune -f --keep-storage=5GB 2>/dev/null || true

# ---- Check disk space before build ----
AVAILABLE_KB=$(df / | tail -1 | awk '{print $4}')
AVAILABLE_GB=$((AVAILABLE_KB / 1048576))
echo "[SERVER] Available disk space: ${AVAILABLE_GB}GB"

if [ "$AVAILABLE_KB" -lt 5000000 ]; then
    echo "[SERVER] WARNING: Less than 5GB disk space available!"
    echo "[SERVER] Pruning old build cache (keeping current image cache)..."
    # Only prune build cache, NOT images — this preserves the pip install layer
    docker builder prune -a -f 2>/dev/null || true
    AVAILABLE_KB=$(df / | tail -1 | awk '{print $4}')

    if [ "$AVAILABLE_KB" -lt 3000000 ]; then
        echo "[SERVER] Still low on space. Removing unused images..."
        # Remove images not used by running containers, but keep the build cache
        docker image prune -a -f 2>/dev/null || true
        AVAILABLE_KB=$(df / | tail -1 | awk '{print $4}')
    fi

    if [ "$AVAILABLE_KB" -lt 2000000 ]; then
        echo "[SERVER] ERROR: Less than 2GB available after cleanup. Cannot build safely."
        echo "[SERVER] Run './deploy.sh --cleanup' from local to free more space."
        df -h /
        exit 1
    fi
fi

# ---- Build ----
if [ "$BUILD_MODE" = "--fresh" ]; then
    echo "[SERVER] Building Docker image (FRESH — no cache, full rebuild)..."
    $COMPOSE build --no-cache
else
    echo "[SERVER] Building Docker image (cached — fast for code-only changes)..."
    $COMPOSE build
fi

echo "[SERVER] Stopping old container (if running)..."
$COMPOSE down --remove-orphans 2>/dev/null || true

echo "[SERVER] Starting bot..."
$COMPOSE up -d

echo "[SERVER] Waiting 10 seconds for startup..."
sleep 10

echo "[SERVER] Container status:"
$COMPOSE ps

echo "[SERVER] Recent logs:"
$COMPOSE logs --tail=30

echo ""
echo "[SERVER] Disk usage:"
df -h / | tail -1 | awk '{print "  " $3 " used / " $2 " total (" $5 " full)"}'

echo "[SERVER] Deploy complete!"
REMOTE_SCRIPT

log "Deployment finished successfully!"
echo ""
info "Available commands:"
info "  ./deploy.sh              Smart deploy (auto-detects changes)"
info "  ./deploy.sh --restart    Restart only (after .env change on server)"
info "  ./deploy.sh --fresh      Full rebuild (nuclear option)"
info "  ./deploy.sh --fresh-db   Rebuild + wipe database & logs (clean slate)"
info "  ./deploy.sh --logs       View recent logs"
info "  ./deploy.sh --status     Container + disk status"
info "  ./deploy.sh --cleanup    Free disk space"
