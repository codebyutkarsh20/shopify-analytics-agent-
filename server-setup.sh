#!/usr/bin/env bash
# ============================================
# First-time Azure server setup
# Run this ONCE on a fresh server before deploy
# ============================================
set -euo pipefail

echo "=== Shopify Analytics Agent - Server Setup ==="

# Update system
echo "[1/4] Updating system packages..."
sudo apt-get update && sudo apt-get upgrade -y

# Install Docker
if command -v docker &> /dev/null; then
    echo "[2/4] Docker already installed: $(docker --version)"
else
    echo "[2/4] Installing Docker..."
    curl -fsSL https://get.docker.com | sh
    sudo usermod -aG docker "$USER"
    echo "Docker installed. You may need to log out and back in for group changes."
fi

# Install Docker Compose plugin
if docker compose version &> /dev/null 2>&1; then
    echo "[3/4] Docker Compose already installed: $(docker compose version)"
else
    echo "[3/4] Installing Docker Compose plugin..."
    sudo apt-get install -y docker-compose-plugin
fi

# Create project directory
PROJECT_DIR="/home/azureuser/shopify-analytics-agent"
echo "[4/4] Setting up project directory at $PROJECT_DIR ..."
mkdir -p "$PROJECT_DIR"
cd "$PROJECT_DIR"

# Create .env from example if it doesn't exist
if [ ! -f .env ]; then
    if [ -f .env.example ]; then
        cp .env.example .env
        echo ""
        echo "=== ACTION REQUIRED ==="
        echo "Edit your .env file with real credentials:"
        echo "  nano $PROJECT_DIR/.env"
        echo ""
        echo "Required variables:"
        echo "  - TELEGRAM_BOT_TOKEN"
        echo "  - ANTHROPIC_API_KEY (or OPENAI_API_KEY)"
        echo "  - SHOPIFY_ACCESS_TOKEN"
        echo "  - SHOPIFY_SHOP_DOMAIN"
    else
        echo ".env.example not found yet. Run deploy.sh first, then re-run this script."
    fi
else
    echo ".env already exists. Skipping."
fi

echo ""
echo "=== Setup Complete ==="
echo "Next steps:"
echo "  1. Edit .env with your credentials: nano $PROJECT_DIR/.env"
echo "  2. From your local machine, run: ./deploy.sh"
