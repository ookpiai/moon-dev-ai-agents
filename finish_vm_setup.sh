#!/bin/bash
# Complete VM Setup - Run this script on the Vultr VM
set -e

echo "Creating systemd services..."

# Create collector service
cat > /tmp/polymarket-collector.service << 'ENDSERVICE'
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
ENDSERVICE

sudo mv /tmp/polymarket-collector.service /etc/systemd/system/
echo "✓ Collector service created"

# Create scanner service
cat > /tmp/polymarket-scanner.service << 'ENDSERVICE'
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
ENDSERVICE

sudo mv /tmp/polymarket-scanner.service /etc/systemd/system/
echo "✓ Scanner service created"

# Create logs directory
mkdir -p /root/moon-dev-ai-agents/logs
echo "✓ Logs directory created"

# Create email script
cat > /root/moon-dev-ai-agents/send_daily_status.sh << 'ENDSCRIPT'
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
ENDSCRIPT

chmod +x /root/moon-dev-ai-agents/send_daily_status.sh
echo "✓ Email script created"

# Set up cron job
(crontab -l 2>/dev/null; echo "0 8 * * * /root/moon-dev-ai-agents/send_daily_status.sh") | crontab -
echo "✓ Cron job configured"

# Reload and start services
sudo systemctl daemon-reload
sudo systemctl enable polymarket-collector polymarket-scanner
sudo systemctl start polymarket-collector polymarket-scanner
echo "✓ Services started"

echo ""
echo "=========================================="
echo "SETUP COMPLETE!"
echo "=========================================="
echo ""

# Show status
sudo systemctl status polymarket-collector --no-pager
echo ""
sudo systemctl status polymarket-scanner --no-pager
echo ""
echo "View logs with:"
echo "  tail -f /root/moon-dev-ai-agents/logs/collector.log"
echo "  tail -f /root/moon-dev-ai-agents/logs/scanner.log"
