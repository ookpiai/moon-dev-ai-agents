#!/bin/bash
################################################################################
# Complete VM Setup - Run this on your Vultr VM
################################################################################

set -e

echo "=========================================="
echo "Completing Polymarket VM Setup"
echo "=========================================="
echo ""

# Create data collector service
echo "[1/6] Creating polymarket-collector service..."
sudo tee /etc/systemd/system/polymarket-collector.service > /dev/null <<'EOF'
[Unit]
Description=Polymarket Data Collector
After=network.target

[Service]
Type=simple
User=root
WorkingDirectory=/root/moon-dev-ai-agents
Environment="PYTHONIOENCODING=utf-8"
Environment="PATH=/usr/local/bin:/usr/bin:/bin"
ExecStart=/usr/bin/python3 /root/moon-dev-ai-agents/src/agents/polymarket_data_collector.py
Restart=always
RestartSec=30
StandardOutput=append:/root/moon-dev-ai-agents/logs/collector.log
StandardError=append:/root/moon-dev-ai-agents/logs/collector_error.log

[Install]
WantedBy=multi-user.target
EOF

# Create scanner service
echo "[2/6] Creating polymarket-scanner service..."
sudo tee /etc/systemd/system/polymarket-scanner.service > /dev/null <<'EOF'
[Unit]
Description=Polymarket Scanner
After=network.target polymarket-collector.service

[Service]
Type=simple
User=root
WorkingDirectory=/root/moon-dev-ai-agents
Environment="PYTHONIOENCODING=utf-8"
Environment="PATH=/usr/local/bin:/usr/bin:/bin"
ExecStart=/usr/bin/python3 /root/moon-dev-ai-agents/src/agents/polymarket_scanner.py
Restart=always
RestartSec=30
StandardOutput=append:/root/moon-dev-ai-agents/logs/scanner.log
StandardError=append:/root/moon-dev-ai-agents/logs/scanner_error.log

[Install]
WantedBy=multi-user.target
EOF

# Create logs directory
echo "[3/6] Creating logs directory..."
mkdir -p /root/moon-dev-ai-agents/logs

# Create daily email script
echo "[4/6] Creating daily status email script..."
tee /root/moon-dev-ai-agents/send_daily_status.sh > /dev/null <<'EOF'
#!/bin/bash
PROJECT_DIR="/root/moon-dev-ai-agents"
cd $PROJECT_DIR

COLLECTOR_STATUS=$(systemctl is-active polymarket-collector.service)
SCANNER_STATUS=$(systemctl is-active polymarket-scanner.service)
SNAPSHOT_COUNT=$(wc -l < src/data/polymarket/training_data/market_snapshots.csv 2>/dev/null || echo "0")

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
RECENT LOGS (Last 50 lines)
========================================

=== Data Collector ===
$(tail -50 logs/collector.log 2>/dev/null || echo "No logs yet")

=== Scanner ===
$(tail -50 logs/scanner.log 2>/dev/null || echo "No logs yet")

========================================
VM: $(hostname)
IP: $(curl -s ifconfig.me)
"

echo "$EMAIL_BODY" | mail -s "Polymarket Daily Status - $(date +%Y-%m-%d)" 00@kpi-ai.com
EOF

chmod +x /root/moon-dev-ai-agents/send_daily_status.sh

# Set up cron job
echo "[5/6] Setting up daily email cron job (8 AM UTC)..."
(crontab -l 2>/dev/null; echo "0 8 * * * /root/moon-dev-ai-agents/send_daily_status.sh") | crontab -

# Start services
echo "[6/6] Starting services..."
sudo systemctl daemon-reload
sudo systemctl enable polymarket-collector polymarket-scanner
sudo systemctl start polymarket-collector polymarket-scanner

echo ""
echo "=========================================="
echo "SETUP COMPLETE!"
echo "=========================================="
echo ""
echo "Service Status:"
echo ""
sudo systemctl status polymarket-collector --no-pager
echo ""
sudo systemctl status polymarket-scanner --no-pager
echo ""
echo "=========================================="
echo "Next Steps:"
echo "=========================================="
echo ""
echo "1. View collector logs:"
echo "   tail -f /root/moon-dev-ai-agents/logs/collector.log"
echo ""
echo "2. View scanner logs:"
echo "   tail -f /root/moon-dev-ai-agents/logs/scanner.log"
echo ""
echo "3. Test email (optional):"
echo "   /root/moon-dev-ai-agents/send_daily_status.sh"
echo ""
echo "4. Daily emails will be sent to: 00@kpi-ai.com at 8 AM UTC"
echo ""
echo "=========================================="
echo "System is now running 24/7!"
echo "=========================================="
