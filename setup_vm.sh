#!/bin/bash
################################################################################
# Polymarket System - Vultr VM Setup Script
# Email: 00@kpi-ai.com
#
# This script will:
# 1. Install all dependencies (Python, packages, etc.)
# 2. Set up systemd services for auto-restart
# 3. Configure daily email status reports
# 4. Start all processes
################################################################################

set -e  # Exit on error

echo "=================================="
echo "Polymarket VM Setup"
echo "=================================="
echo ""

# Configuration
EMAIL="00@kpi-ai.com"
PROJECT_DIR="$HOME/moon-dev-ai-agents"
PYTHON_CMD="python3"

echo "[1/7] Updating system packages..."
sudo apt-get update
sudo apt-get upgrade -y

echo "[2/7] Installing Python 3.11 and dependencies..."
sudo apt-get install -y python3 python3-pip python3-venv git curl wget
sudo apt-get install -y mailutils postfix  # For email notifications

echo "[3/7] Cloning repository..."
if [ ! -d "$PROJECT_DIR" ]; then
    cd $HOME
    git clone https://github.com/your-username/moon-dev-ai-agents.git
    cd $PROJECT_DIR
else
    cd $PROJECT_DIR
    git pull
fi

# Configure git
git config user.email "00@kpi-ai.com"
git config user.name "oo"

echo "[4/7] Installing Python dependencies..."
pip3 install --upgrade pip
pip3 install -r requirements.txt

echo "[5/7] Setting up environment file..."
if [ ! -f .env ]; then
    echo "ERROR: Please create .env file with your API keys first!"
    echo "Copy .env_example to .env and fill in your keys"
    exit 1
fi

echo "[6/7] Creating systemd services..."

# Data Collector Service
sudo tee /etc/systemd/system/polymarket-collector.service > /dev/null <<EOF
[Unit]
Description=Polymarket Data Collector
After=network.target

[Service]
Type=simple
User=$USER
WorkingDirectory=$PROJECT_DIR
Environment="PYTHONIOENCODING=utf-8"
Environment="PATH=/usr/local/bin:/usr/bin:/bin"
ExecStart=$PYTHON_CMD $PROJECT_DIR/src/agents/polymarket_data_collector.py
Restart=always
RestartSec=30
StandardOutput=append:$PROJECT_DIR/logs/collector.log
StandardError=append:$PROJECT_DIR/logs/collector_error.log

[Install]
WantedBy=multi-user.target
EOF

# Scanner Service
sudo tee /etc/systemd/system/polymarket-scanner.service > /dev/null <<EOF
[Unit]
Description=Polymarket Scanner
After=network.target polymarket-collector.service

[Service]
Type=simple
User=$USER
WorkingDirectory=$PROJECT_DIR
Environment="PYTHONIOENCODING=utf-8"
Environment="PATH=/usr/local/bin:/usr/bin:/bin"
ExecStart=$PYTHON_CMD $PROJECT_DIR/src/agents/polymarket_scanner.py
Restart=always
RestartSec=30
StandardOutput=append:$PROJECT_DIR/logs/scanner.log
StandardError=append:$PROJECT_DIR/logs/scanner_error.log

[Install]
WantedBy=multi-user.target
EOF

# Create logs directory
mkdir -p $PROJECT_DIR/logs

echo "[7/7] Setting up daily status email..."

# Create monitoring script
tee $PROJECT_DIR/send_daily_status.sh > /dev/null <<'EOF'
#!/bin/bash
PROJECT_DIR="$HOME/moon-dev-ai-agents"
cd $PROJECT_DIR

# Run dashboard and capture output
DASHBOARD_OUTPUT=$(python3 polymarket_dashboard.py 2>&1)

# Get process status
COLLECTOR_STATUS=$(systemctl is-active polymarket-collector.service)
SCANNER_STATUS=$(systemctl is-active polymarket-scanner.service)

# Count data
SNAPSHOT_COUNT=$(wc -l < src/data/polymarket/training_data/market_snapshots.csv 2>/dev/null || echo "0")

# Create email body
EMAIL_BODY="Polymarket System Daily Status Report
Generated: $(date)

========================================
SERVICE STATUS
========================================
Data Collector: $COLLECTOR_STATUS
Scanner: $SCANNER_STATUS

========================================
DATA COLLECTION
========================================
Total Snapshots: $SNAPSHOT_COUNT

========================================
FULL DASHBOARD OUTPUT
========================================
$DASHBOARD_OUTPUT

========================================
RECENT LOGS (Last 50 lines)
========================================

=== Data Collector ===
$(tail -50 logs/collector.log 2>/dev/null || echo "No logs")

=== Scanner ===
$(tail -50 logs/scanner.log 2>/dev/null || echo "No logs")

========================================
This is an automated status report.
VM: $(hostname)
IP: $(curl -s ifconfig.me)
"

# Send email
echo "$EMAIL_BODY" | mail -s "Polymarket Daily Status - $(date +%Y-%m-%d)" 00@kpi-ai.com
EOF

chmod +x $PROJECT_DIR/send_daily_status.sh

# Set up daily cron job (8 AM UTC = 3 AM EST)
(crontab -l 2>/dev/null; echo "0 8 * * * $PROJECT_DIR/send_daily_status.sh") | crontab -

echo ""
echo "=================================="
echo "Setup Complete!"
echo "=================================="
echo ""
echo "Systemd services created:"
echo "  - polymarket-collector.service"
echo "  - polymarket-scanner.service"
echo ""
echo "To start services:"
echo "  sudo systemctl daemon-reload"
echo "  sudo systemctl enable polymarket-collector"
echo "  sudo systemctl enable polymarket-scanner"
echo "  sudo systemctl start polymarket-collector"
echo "  sudo systemctl start polymarket-scanner"
echo ""
echo "To check status:"
echo "  sudo systemctl status polymarket-collector"
echo "  sudo systemctl status polymarket-scanner"
echo ""
echo "To view logs:"
echo "  tail -f logs/collector.log"
echo "  tail -f logs/scanner.log"
echo ""
echo "Daily email reports will be sent to: $EMAIL"
echo "Test email: $PROJECT_DIR/send_daily_status.sh"
echo ""
