# Vultr VM Setup Guide for Polymarket System

## Quick Start

SSH into your Vultr VM and run these commands:

```bash
# Clone your fork
git clone https://github.com/YOUR_USERNAME/moon-dev-ai-agents.git
cd moon-dev-ai-agents

# Create .env file with your API keys
nano .env
# Paste your API keys (ANTHROPIC_KEY, OPENAI_KEY, etc.)
# Save: Ctrl+X, Y, Enter

# Run setup script
chmod +x setup_vm.sh
./setup_vm.sh

# Start services
sudo systemctl daemon-reload
sudo systemctl enable polymarket-collector polymarket-scanner
sudo systemctl start polymarket-collector polymarket-scanner
```

Done! System will run 24/7 and email you daily reports.

## What Gets Installed

1. **Python 3.11** + pip + dependencies
2. **Git** configured with your email (00@kpi-ai.com)
3. **Postfix** mail server for sending emails
4. **Systemd services** for auto-restart:
   - `polymarket-collector.service` - Data collection
   - `polymarket-scanner.service` - Opportunity scanning
5. **Daily cron job** - Email report at 8 AM UTC (3 AM EST)

## Service Management

### Check Status
```bash
sudo systemctl status polymarket-collector
sudo systemctl status polymarket-scanner
```

### View Live Logs
```bash
tail -f logs/collector.log
tail -f logs/scanner.log
```

### Restart Services
```bash
sudo systemctl restart polymarket-collector
sudo systemctl restart polymarket-scanner
```

### Stop Services
```bash
sudo systemctl stop polymarket-collector
sudo systemctl stop polymarket-scanner
```

## Email Reports

Daily status emails sent to: **00@kpi-ai.com**

### Test Email Now
```bash
./send_daily_status.sh
```

### Change Email Schedule
```bash
crontab -e
# Current: 0 8 * * * (8 AM UTC daily)
# Change to your preference
```

## Claude Code on VM

To use Claude Code on the VM:

```bash
# Install Node.js (required for Claude Code)
curl -fsSL https://deb.nodesource.com/setup_20.x | sudo -E bash -
sudo apt-get install -y nodejs

# Install Claude Code CLI
npm install -g @anthropic-ai/claude-code

# Set up API key
export ANTHROPIC_API_KEY="your-key-here"
echo 'export ANTHROPIC_API_KEY="your-key-here"' >> ~/.bashrc

# Run Claude Code
claude-code
```

## Monitoring

### Dashboard (Manual Check)
```bash
cd ~/moon-dev-ai-agents
python3 polymarket_dashboard.py
```

### Data Collection Progress
```bash
wc -l src/data/polymarket/training_data/market_snapshots.csv
```

### Process Memory/CPU
```bash
htop
# Then press F4 and search for "polymarket"
```

## Troubleshooting

### Services Won't Start
```bash
# Check error logs
sudo journalctl -u polymarket-collector -n 50
sudo journalctl -u polymarket-scanner -n 50

# Check .env file exists
cat .env | grep -v "KEY"
```

### Email Not Sending
```bash
# Test postfix
echo "Test email" | mail -s "Test" 00@kpi-ai.com

# If fails, reconfigure postfix
sudo dpkg-reconfigure postfix
# Choose "Internet Site"
# System mail name: your-vm-hostname
```

### Out of Disk Space
```bash
# Check disk usage
df -h

# Clean up old logs (keeps last 1000 lines)
tail -1000 logs/collector.log > logs/collector.log.tmp
mv logs/collector.log.tmp logs/collector.log
```

### High Memory Usage
```bash
# Check memory
free -h

# If needed, add swap space
sudo fallocate -l 2G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab
```

## Data Backup

### Manual Backup to Local Machine
```bash
# On your local machine:
scp -r root@your-vm-ip:~/moon-dev-ai-agents/src/data/polymarket ./polymarket_backup_$(date +%Y%m%d)
```

### Automated Daily Backup (Optional)
Add to crontab on VM:
```bash
crontab -e

# Add this line (backup at midnight):
0 0 * * * cd ~/moon-dev-ai-agents && tar -czf backup_$(date +\%Y\%m\%d).tar.gz src/data/polymarket/training_data/ && find . -name "backup_*.tar.gz" -mtime +7 -delete
```

## Updating Code

```bash
cd ~/moon-dev-ai-agents
git pull
sudo systemctl restart polymarket-collector polymarket-scanner
```

## Cost Tracking

- **VM**: ~$12/month (2GB Vultr)
- **Bandwidth**: Free (generous limits)
- **API calls**: ~$10-20/month (only when opportunities found)
- **Total**: ~$22-32/month

## Security

### Firewall Setup
```bash
sudo ufw allow 22/tcp   # SSH
sudo ufw enable
```

### SSH Key Authentication (Recommended)
```bash
# On local machine, generate key if you don't have one:
ssh-keygen -t ed25519 -C "00@kpi-ai.com"

# Copy to VM:
ssh-copy-id root@your-vm-ip

# Disable password auth:
sudo nano /etc/ssh/sshd_config
# Set: PasswordAuthentication no
sudo systemctl restart sshd
```

## Expected Timeline

| Time | Expected Status |
|------|----------------|
| **Day 1** | Setup complete, data collection starts |
| **Week 1** | 10,000+ snapshots, meta-learner training |
| **Week 2-3** | First opportunities may appear |
| **Week 4-6** | 10-20 trades, preliminary edge assessment |
| **Week 8+** | 30+ trades, statistical confidence |

## Contact

Email: 00@kpi-ai.com
Daily reports will be sent automatically to this address.

---

**Setup Date**: $(date)
**VM Provider**: Vultr
**Auto-restart**: Enabled
**Email Reports**: Daily at 8 AM UTC
