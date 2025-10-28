"""
Polymarket System Dashboard
Quick status check for all components
"""

import os
import json
import pandas as pd
from datetime import datetime
from termcolor import cprint

def check_data_collection():
    """Check data collection status"""
    cprint("\n" + "="*80, "cyan")
    cprint("1. DATA COLLECTION", "cyan", attrs=["bold"])
    cprint("="*80, "cyan")

    snapshots_path = "src/data/polymarket/training_data/market_snapshots.csv"
    orderbook_path = "src/data/polymarket/training_data/orderbook_snapshots.csv"

    if os.path.exists(snapshots_path):
        df = pd.read_csv(snapshots_path)
        cprint(f"[OK] Market snapshots: {len(df):,} records", "green")

        if len(df) > 0:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            start = df['timestamp'].min()
            end = df['timestamp'].max()
            duration = (end - start).total_seconds() / 3600

            cprint(f"[OK] Time range: {start} to {end}", "green")
            cprint(f"[OK] Duration: {duration:.1f} hours ({duration/24:.1f} days)", "green")
            cprint(f"[OK] Unique markets: {df['market_id'].nunique()}", "green")
            cprint(f"[OK] File size: {os.path.getsize(snapshots_path) / 1024:.1f} KB", "green")
    else:
        cprint("[WARN] No market snapshots found", "yellow")

    if os.path.exists(orderbook_path):
        df_ob = pd.read_csv(orderbook_path)
        cprint(f"[OK] Order book snapshots: {len(df_ob):,} records", "green")
    else:
        cprint("[WARN] No order book snapshots found", "yellow")

def check_meta_learning():
    """Check meta-learning calibration"""
    cprint("\n" + "="*80, "cyan")
    cprint("2. META-LEARNING", "cyan", attrs=["bold"])
    cprint("="*80, "cyan")

    calibration_path = "src/data/polymarket/meta_learning/calibration.json"

    if os.path.exists(calibration_path):
        with open(calibration_path, 'r') as f:
            cal = json.load(f)

        cprint(f"[OK] Calibration file exists", "green")
        cprint(f"[OK] Version: {cal.get('version', 'unknown')}", "green")
        cprint(f"[OK] Model: {cal.get('meta_model', 'unknown')}", "green")
        cprint(f"[OK] Updated: {cal.get('updated_at', 'unknown')[:19]}", "green")

        segments = cal.get('segments', {})
        cprint(f"[OK] Market segments: {len(segments)}", "green")

        for seg_name, seg_data in segments.items():
            samples = seg_data.get('samples', 0)
            kelly = seg_data.get('kelly_multiplier', 1.0)
            cprint(f"     - {seg_name}: {samples} samples (Kelly={kelly}x)", "white")
    else:
        cprint("[WARN] No calibration file found", "yellow")
        cprint("[INFO] Run: python src/agents/polymarket_meta_learner.py", "white")

def check_positions():
    """Check open/closed positions"""
    cprint("\n" + "="*80, "cyan")
    cprint("3. POSITIONS", "cyan", attrs=["bold"])
    cprint("="*80, "cyan")

    open_path = "src/data/polymarket/positions/open_positions.csv"
    closed_path = "src/data/polymarket/positions/closed_positions.csv"

    if os.path.exists(open_path):
        df_open = pd.read_csv(open_path)
        cprint(f"[OK] Open positions: {len(df_open)}", "green")

        if len(df_open) > 0:
            total_invested = df_open['position_size'].sum()
            cprint(f"[OK] Total capital deployed: ${total_invested:,.2f}", "green")
    else:
        cprint("[INFO] No open positions", "white")

    if os.path.exists(closed_path):
        df_closed = pd.read_csv(closed_path)
        cprint(f"[OK] Closed positions: {len(df_closed)}", "green")

        if len(df_closed) > 0:
            total_pnl = df_closed['pnl'].sum()
            win_rate = (df_closed['pnl'] > 0).mean() * 100

            pnl_color = "green" if total_pnl > 0 else "red"
            cprint(f"[OK] Total PnL: ${total_pnl:,.2f}", pnl_color)
            cprint(f"[OK] Win rate: {win_rate:.1f}%", "green")
    else:
        cprint("[INFO] No closed positions yet", "white")

def check_scanner():
    """Check scanner configuration"""
    cprint("\n" + "="*80, "cyan")
    cprint("4. SCANNER", "cyan", attrs=["bold"])
    cprint("="*80, "cyan")

    scanner_path = "src/agents/polymarket_scanner.py"

    if os.path.exists(scanner_path):
        cprint(f"[OK] Scanner installed", "green")
        cprint(f"[INFO] Start with: python src/agents/polymarket_scanner.py", "white")

        # Check config
        try:
            from src.config import (
                POLYMARKET_SCAN_INTERVAL_MINUTES,
                POLYMARKET_MAX_SPREAD,
                POLYMARKET_MIN_LIQUIDITY
            )

            cprint(f"\n[CONFIG] Scan interval: {POLYMARKET_SCAN_INTERVAL_MINUTES} minutes", "white")
            cprint(f"[CONFIG] Max spread: {POLYMARKET_MAX_SPREAD:.2%}", "white")
            cprint(f"[CONFIG] Min liquidity: ${POLYMARKET_MIN_LIQUIDITY:,.0f}", "white")
        except:
            cprint("[WARN] Could not load config", "yellow")
    else:
        cprint("[ERROR] Scanner not found", "red")

def check_running_processes():
    """Check for running processes"""
    cprint("\n" + "="*80, "cyan")
    cprint("5. RUNNING PROCESSES", "cyan", attrs=["bold"])
    cprint("="*80, "cyan")

    # Note: This is a simplified check
    # In production, you'd use psutil or check process IDs

    collector_log = "src/data/polymarket/logs/data_collector.log"

    if os.path.exists(collector_log):
        # Check if log was recently updated
        mtime = os.path.getmtime(collector_log)
        last_update = datetime.fromtimestamp(mtime)
        now = datetime.now()
        seconds_ago = (now - last_update).total_seconds()

        if seconds_ago < 120:  # Updated in last 2 minutes
            cprint(f"[OK] Data collector appears to be running", "green")
            cprint(f"[OK] Last log update: {int(seconds_ago)}s ago", "green")
        else:
            cprint(f"[WARN] Data collector may not be running", "yellow")
            cprint(f"[WARN] Last log update: {int(seconds_ago/60)} minutes ago", "yellow")
    else:
        cprint("[INFO] No data collector log found", "white")

    cprint("\n[INFO] To check all processes, run:", "white")
    cprint("       ps aux | grep polymarket", "white")

def show_quick_actions():
    """Show quick action commands"""
    cprint("\n" + "="*80, "cyan")
    cprint("QUICK ACTIONS", "cyan", attrs=["bold"])
    cprint("="*80, "cyan")

    actions = [
        ("Start data collection", "python src/agents/polymarket_data_collector.py"),
        ("Train meta-learner", "python src/agents/polymarket_meta_learner.py"),
        ("Run scanner once", "python src/agents/polymarket_scanner.py --once"),
        ("Start continuous scanning", "python src/agents/polymarket_scanner.py"),
        ("Check data status", "python polymarket_status.py"),
    ]

    for i, (desc, cmd) in enumerate(actions, 1):
        cprint(f"\n{i}. {desc}:", "yellow")
        cprint(f"   {cmd}", "white")

def main():
    """Main dashboard"""
    cprint("\n" + "="*80, "cyan", attrs=["bold"])
    cprint("POLYMARKET SYSTEM DASHBOARD", "cyan", attrs=["bold"])
    cprint("="*80, "cyan", attrs=["bold"])
    cprint(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", "white")

    try:
        check_data_collection()
        check_meta_learning()
        check_positions()
        check_scanner()
        check_running_processes()
        show_quick_actions()

        cprint("\n" + "="*80, "cyan")
        cprint("[DASHBOARD] All checks complete", "green", attrs=["bold"])
        cprint("="*80 + "\n", "cyan")

    except Exception as e:
        cprint(f"\n[ERROR] Dashboard check failed: {e}", "red")

if __name__ == "__main__":
    main()
