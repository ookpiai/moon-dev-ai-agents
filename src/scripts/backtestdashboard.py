"""
🌙 Moon Dev's AI Agent Backtests Dashboard 🚀
FastAPI web interface for viewing backtest results from rbi_agent_pp_multi.py
Built with love by Moon Dev

================================================================================
📋 HOW TO USE THIS DASHBOARD:
================================================================================

1. RUN THE RBI AGENT to generate backtest results:
   ```bash
   python src/agents/rbi_agent_pp_multi.py
   ```
   This will create a CSV file with all your backtest stats at:
   src/data/rbi_pp_multi/backtest_stats.csv

2. CONFIGURE THE CSV PATH below (line 60) to point to your stats CSV

3. RUN THIS DASHBOARD:
   ```bash
   python src/scripts/backtestdashboard.py
   ```

4. OPEN YOUR BROWSER to: http://localhost:8001

================================================================================
⚙️ CONFIGURATION:
================================================================================
"""

from fastapi import FastAPI, Request, BackgroundTasks
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from typing import List, Dict, Any
import pandas as pd
import numpy as np
from pathlib import Path
import uvicorn
import shutil
import subprocess
import threading
from datetime import datetime
import sys
import asyncio
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.interval import IntervalTrigger
import traceback
import logging

# Add path to import MoonDevAPI
sys.path.append('/Users/md/Dropbox/dev/github/moon-dev-trading-bots')
from api import MoonDevAPI

# ============================================================================
# 🔧 CONFIGURATION - CHANGE THESE PATHS TO MATCH YOUR SETUP!
# ============================================================================

# 📊 Path to your backtest stats CSV file
# This CSV is created by rbi_agent_pp_multi.py after running backtests
# Default: src/data/rbi_pp_multi/backtest_stats.csv
STATS_CSV = Path("/Users/md/Dropbox/dev/github/moon-dev-ai-agents-for-trading/src/data/rbi_pp_multi/backtest_stats.csv")

# 📁 Directory for static files (CSS, JS) and templates (HTML)
# These files are located in: src/data/rbi_pp_multi/static and src/data/rbi_pp_multi/templates
TEMPLATE_BASE_DIR = Path("/Users/md/Dropbox/dev/github/moon-dev-ai-agents-for-trading/src/data/rbi_pp_multi")

# 🗂️ Directory to store user-created folders
# Folders allow you to organize and group your backtest results
USER_FOLDERS_DIR = TEMPLATE_BASE_DIR / "user_folders"

# 🎯 Target return percentage (must match rbi_agent_pp_multi.py TARGET_RETURN)
TARGET_RETURN = 50  # % - Optimization goal
SAVE_IF_OVER_RETURN = 1.0  # % - Minimum return to save to CSV

# 📊 Data Portal Configuration - Moon Dev
DATA_DIR = TEMPLATE_BASE_DIR / "downloads"
DATA_DIR.mkdir(parents=True, exist_ok=True)

# 📊 Test Data Sets Directory - Historical datasets for backtesting
TEST_DATA_DIR = Path("/Users/md/Dropbox/dev/github/moon-dev-ai-agents-for-trading/src/data/private_data")

# TEST MODE for data portal - Set to True for fast testing with sample data
TEST_MODE = True

# ============================================================================
# 🚀 FASTAPI APP INITIALIZATION
# ============================================================================

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Moon Dev's AI Agent Backtests")

# Create user_folders directory if it doesn't exist
USER_FOLDERS_DIR.mkdir(exist_ok=True)

# Track running backtests
running_backtests = {}

# 🌙 Moon Dev Data API Integration
moon_api = MoonDevAPI()

# Track data update status
data_status = {
    "liquidations": {"status": "pending", "last_updated": None, "file_size": None},
    "oi": {"status": "pending", "last_updated": None, "file_size": None}
}

# Mount static files and templates
app.mount("/static", StaticFiles(directory=str(TEMPLATE_BASE_DIR / "static")), name="static")
templates = Jinja2Templates(directory=str(TEMPLATE_BASE_DIR / "templates"))


# 🌙 Moon Dev: Request models for folder operations
class AddToFolderRequest(BaseModel):
    folder_name: str
    backtests: List[Dict[str, Any]]


class DeleteFolderRequest(BaseModel):
    folder_name: str


class BacktestRunRequest(BaseModel):
    ideas: str
    run_name: str


# ============================================================================
# 🌙 MOON DEV DATA API FUNCTIONS
# ============================================================================

def format_file_size(size_bytes):
    """Format file size in human readable format"""
    if size_bytes is None:
        return "N/A"
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.2f} TB"


async def fetch_liquidation_data():
    """Fetch liquidation data from Moon Dev API"""
    try:
        if TEST_MODE:
            logger.info("🧪 TEST MODE: Creating sample liquidation data...")
            data_status["liquidations"]["status"] = "fetching"

            # Create realistic sample data for testing (10,000 rows)
            num_rows = 10000
            symbols = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'BNBUSDT', 'ADAUSDT'] * (num_rows // 5)
            sample_data = pd.DataFrame({
                'symbol': symbols[:num_rows],
                'side': ['Buy', 'Sell'] * (num_rows // 2),
                'size': [100000 + i * 10000 for i in range(num_rows)],
                'price': [50000 + i * 100 for i in range(num_rows)],
                'timestamp': [datetime.now().timestamp() - i * 3600 for i in range(num_rows)]
            })

            file_path = DATA_DIR / "liquidations.csv"
            sample_data.to_csv(file_path, index=False)

            file_size = file_path.stat().st_size
            data_status["liquidations"]["status"] = "ready"
            data_status["liquidations"]["last_updated"] = datetime.now().strftime("%H:%M:%S")
            data_status["liquidations"]["file_size"] = file_size

            logger.info(f"✅ TEST MODE: Sample liquidation data created: {format_file_size(file_size)}")
        else:
            logger.info("🌙 Fetching liquidation data...")
            data_status["liquidations"]["status"] = "fetching"

            # Fetch ALL liquidation data (no limit for full dataset)
            df = moon_api.get_liquidation_data(limit=None)

            if df is not None:
                file_path = DATA_DIR / "liquidations.csv"
                df.to_csv(file_path, index=False)

                file_size = file_path.stat().st_size
                data_status["liquidations"]["status"] = "ready"
                data_status["liquidations"]["last_updated"] = datetime.now().strftime("%H:%M:%S")
                data_status["liquidations"]["file_size"] = file_size

                logger.info(f"✅ Liquidation data saved: {format_file_size(file_size)}")
            else:
                data_status["liquidations"]["status"] = "error"
                logger.error("❌ Failed to fetch liquidation data")

    except Exception as e:
        data_status["liquidations"]["status"] = "error"
        logger.error(f"💥 Error fetching liquidation data: {str(e)}")
        logger.error(traceback.format_exc())


async def fetch_oi_data():
    """Fetch open interest data from Moon Dev API"""
    try:
        if TEST_MODE:
            logger.info("🧪 TEST MODE: Creating sample OI data...")
            data_status["oi"]["status"] = "fetching"

            # Create realistic sample data for testing (10,000 rows)
            num_rows = 10000
            symbols = ['BTC', 'ETH', 'SOL', 'BNB', 'ADA'] * (num_rows // 5)
            exchanges = ['Binance', 'Bybit', 'OKX', 'Bitget', 'Deribit'] * (num_rows // 5)
            sample_data = pd.DataFrame({
                'symbol': symbols[:num_rows],
                'exchange': exchanges[:num_rows],
                'open_interest': [1000000 + i * 50000 for i in range(num_rows)],
                'timestamp': [datetime.now().timestamp() - i * 3600 for i in range(num_rows)]
            })

            file_path = DATA_DIR / "oi.csv"
            sample_data.to_csv(file_path, index=False)

            file_size = file_path.stat().st_size
            data_status["oi"]["status"] = "ready"
            data_status["oi"]["last_updated"] = datetime.now().strftime("%H:%M:%S")
            data_status["oi"]["file_size"] = file_size

            logger.info(f"✅ TEST MODE: Sample OI data created: {format_file_size(file_size)}")
        else:
            logger.info("📊 Fetching OI data...")
            data_status["oi"]["status"] = "fetching"

            df = moon_api.get_oi_data()

            if df is not None:
                file_path = DATA_DIR / "oi.csv"
                df.to_csv(file_path, index=False)

                file_size = file_path.stat().st_size
                data_status["oi"]["status"] = "ready"
                data_status["oi"]["last_updated"] = datetime.now().strftime("%H:%M:%S")
                data_status["oi"]["file_size"] = file_size

                logger.info(f"✅ OI data saved: {format_file_size(file_size)}")
            else:
                data_status["oi"]["status"] = "error"
                logger.error("❌ Failed to fetch OI data")

    except Exception as e:
        data_status["oi"]["status"] = "error"
        logger.error(f"💥 Error fetching OI data: {str(e)}")
        logger.error(traceback.format_exc())


async def fetch_all_data():
    """Fetch all data from Moon Dev API"""
    logger.info("🚀 Starting data fetch for all datasets...")

    try:
        # Run all fetches concurrently
        await asyncio.gather(
            fetch_liquidation_data(),
            fetch_oi_data()
        )
        logger.info("✨ Data fetch complete!")
    except Exception as e:
        logger.error(f"Error during data fetch: {str(e)}")
        # Don't crash, just log the error


async def background_data_fetch():
    """Background task to fetch data without blocking startup"""
    await asyncio.sleep(1)  # Small delay to let server fully start
    await fetch_all_data()


# ============================================================================
# 🌙 ROUTES
# ============================================================================

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Render the main dashboard page"""
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/api/backtests")
async def get_backtests():
    """API endpoint to fetch all backtest data"""
    try:
        if not STATS_CSV.exists():
            return JSONResponse({
                "data": [],
                "message": "No backtest data found yet. Run rbi_agent_pp_multi.py to generate results!"
            })

        # 🌙 Moon Dev: Read CSV with proper header handling
        # Check if header needs updating (old format without Exposure %)
        with open(STATS_CSV, 'r') as f:
            header_line = f.readline().strip()

        # If header is old format, read with names parameter to handle 13 columns
        if 'Exposure %' not in header_line:
            print("📊 Detected old CSV header format - reading with new column names")
            df = pd.read_csv(
                STATS_CSV,
                names=['Strategy Name', 'Thread ID', 'Return %', 'Buy & Hold %',
                       'Max Drawdown %', 'Sharpe Ratio', 'Sortino Ratio', 'Exposure %',
                       'EV %', 'Trades', 'File Path', 'Data', 'Time'],
                skiprows=1,  # Skip old header
                on_bad_lines='warn'
            )
        else:
            # New format - read normally
            df = pd.read_csv(STATS_CSV, on_bad_lines='warn')

        # Debug: Print columns
        print(f"📊 CSV Columns: {list(df.columns)}")
        print(f"📊 Row count: {len(df)}")

        # Convert numeric columns, replacing 'N/A' with NaN
        numeric_cols = ['Return %', 'Buy & Hold %', 'Max Drawdown %', 'Sharpe Ratio', 'Sortino Ratio', 'Exposure %', 'EV %', 'Trades']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        # Replace inf/-inf with None (can't JSON serialize infinity!)
        df = df.replace([np.inf, -np.inf], None)

        # Replace NaN with None for JSON serialization
        df = df.where(pd.notnull(df), None)

        # Convert to records and clean floats
        data = []
        for record in df.to_dict('records'):
            cleaned_record = {}
            for key, value in record.items():
                # Check if value is a problematic float
                if isinstance(value, (float, np.floating)):
                    if np.isnan(value) or np.isinf(value):
                        cleaned_record[key] = None
                    else:
                        cleaned_record[key] = float(value)
                else:
                    cleaned_record[key] = value
            data.append(cleaned_record)

        return JSONResponse({
            "data": data,
            "total": len(data),
            "message": f"Loaded {len(data)} backtest results"
        })

    except Exception as e:
        print(f"❌ Error in /api/backtests: {str(e)}")
        import traceback
        traceback.print_exc()
        return JSONResponse({
            "data": [],
            "error": str(e),
            "message": "Error loading backtest data"
        }, status_code=500)


@app.get("/api/stats")
async def get_stats():
    """API endpoint for summary statistics"""
    try:
        if not STATS_CSV.exists():
            return JSONResponse({
                "total_backtests": 0,
                "unique_strategies": 0,
                "unique_data_sources": 0,
                "avg_return": 0,
                "max_return": 0,
                "avg_sortino": 0,
                "message": "No data yet"
            })

        # 🌙 Moon Dev: Read CSV with proper header handling
        with open(STATS_CSV, 'r') as f:
            header_line = f.readline().strip()

        # If header is old format, read with names parameter to handle 13 columns
        if 'Exposure %' not in header_line:
            df = pd.read_csv(
                STATS_CSV,
                names=['Strategy Name', 'Thread ID', 'Return %', 'Buy & Hold %',
                       'Max Drawdown %', 'Sharpe Ratio', 'Sortino Ratio', 'Exposure %',
                       'EV %', 'Trades', 'File Path', 'Data', 'Time'],
                skiprows=1,
                on_bad_lines='warn'
            )
        else:
            df = pd.read_csv(STATS_CSV, on_bad_lines='warn')

        print(f"📊 Stats CSV Columns: {list(df.columns)}")

        # Convert numeric columns, replacing 'N/A' with NaN
        numeric_cols = ['Return %', 'Buy & Hold %', 'Max Drawdown %', 'Sharpe Ratio', 'Sortino Ratio', 'Exposure %', 'EV %', 'Trades']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        # Replace inf/-inf with NaN (can't calculate stats on infinity!)
        df = df.replace([float('inf'), float('-inf')], float('nan'))

        # Helper function to safely get numeric stat
        def safe_stat(series, func, default=0):
            try:
                val = func(series)
                if pd.isna(val) or np.isinf(val) or not isinstance(val, (int, float)):
                    return default
                # Ensure it's JSON-safe
                val = float(val)
                if np.isnan(val) or np.isinf(val):
                    return default
                return round(val, 2)
            except:
                return default

        stats = {
            "total_backtests": len(df),
            "unique_strategies": df['Strategy Name'].nunique() if 'Strategy Name' in df.columns else 0,
            "unique_data_sources": df['Data'].nunique() if 'Data' in df.columns else 0,
            "avg_return": safe_stat(df['Return %'], lambda s: s.mean()) if 'Return %' in df.columns else 0,
            "max_return": safe_stat(df['Return %'], lambda s: s.max()) if 'Return %' in df.columns else 0,
            "avg_sortino": safe_stat(df['Sortino Ratio'], lambda s: s.mean()) if 'Sortino Ratio' in df.columns else 0,
        }

        return JSONResponse(stats)

    except Exception as e:
        print(f"❌ Error in /api/stats: {str(e)}")
        import traceback
        traceback.print_exc()
        return JSONResponse({"error": str(e)}, status_code=500)


@app.get("/api/folders")
async def get_folders():
    """🌙 Moon Dev: Get list of all folder names"""
    try:
        folders = [f.name for f in USER_FOLDERS_DIR.iterdir() if f.is_dir()]
        return JSONResponse({"folders": sorted(folders)})
    except Exception as e:
        print(f"❌ Error in /api/folders: {str(e)}")
        return JSONResponse({"folders": [], "error": str(e)}, status_code=500)


@app.get("/api/folders/list")
async def list_folders_with_details():
    """🌙 Moon Dev: Get folders with backtest counts"""
    try:
        folders_info = []

        for folder_path in USER_FOLDERS_DIR.iterdir():
            if folder_path.is_dir():
                csv_path = folder_path / "backtest_stats.csv"
                count = 0

                if csv_path.exists():
                    df = pd.read_csv(csv_path)
                    count = len(df)

                folders_info.append({
                    "name": folder_path.name,
                    "count": count
                })

        return JSONResponse({"folders": sorted(folders_info, key=lambda x: x['name'])})

    except Exception as e:
        print(f"❌ Error in /api/folders/list: {str(e)}")
        import traceback
        traceback.print_exc()
        return JSONResponse({"folders": [], "error": str(e)}, status_code=500)


@app.post("/api/folders/add")
async def add_to_folder(request: AddToFolderRequest):
    """🌙 Moon Dev: Add backtests to a folder (duplicates rows, doesn't move)"""
    try:
        folder_name = request.folder_name
        backtests = request.backtests

        # Create folder if it doesn't exist
        folder_path = USER_FOLDERS_DIR / folder_name
        folder_path.mkdir(exist_ok=True)

        folder_csv = folder_path / "backtest_stats.csv"

        # Convert backtests to DataFrame
        new_df = pd.DataFrame(backtests)

        # If folder CSV exists, append to it; otherwise create new
        if folder_csv.exists():
            existing_df = pd.read_csv(folder_csv)
            combined_df = pd.concat([existing_df, new_df], ignore_index=True)
            combined_df.to_csv(folder_csv, index=False)
            print(f"📁 Added {len(new_df)} backtests to existing folder '{folder_name}'")
        else:
            new_df.to_csv(folder_csv, index=False)
            print(f"📁 Created new folder '{folder_name}' with {len(new_df)} backtests")

        return JSONResponse({
            "success": True,
            "message": f"Added {len(backtests)} backtest(s) to '{folder_name}'"
        })

    except Exception as e:
        print(f"❌ Error in /api/folders/add: {str(e)}")
        import traceback
        traceback.print_exc()
        return JSONResponse({
            "success": False,
            "error": str(e),
            "message": "Failed to add backtests to folder"
        }, status_code=500)


@app.get("/api/folders/{folder_name}/paths")
async def get_folder_paths(folder_name: str):
    """🌙 Moon Dev: Get all file paths from a folder"""
    try:
        folder_path = USER_FOLDERS_DIR / folder_name
        csv_path = folder_path / "backtest_stats.csv"

        if not csv_path.exists():
            return JSONResponse({
                "success": False,
                "message": f"Folder '{folder_name}' has no backtest data"
            }, status_code=404)

        # Read CSV and extract file paths
        df = pd.read_csv(csv_path)

        if 'File Path' not in df.columns:
            return JSONResponse({
                "success": False,
                "message": "No 'File Path' column found in folder data"
            }, status_code=400)

        # Get all file paths, filter out any null/empty values
        paths = df['File Path'].dropna().tolist()

        print(f"📁 Retrieved {len(paths)} paths from folder '{folder_name}'")

        return JSONResponse({
            "success": True,
            "paths": paths,
            "count": len(paths)
        })

    except Exception as e:
        print(f"❌ Error in /api/folders/{folder_name}/paths: {str(e)}")
        import traceback
        traceback.print_exc()
        return JSONResponse({
            "success": False,
            "error": str(e),
            "message": "Failed to get folder paths"
        }, status_code=500)


@app.get("/api/backtest/status/{run_name}")
async def get_backtest_status(run_name: str):
    """🌙 Moon Dev: Check status of a running backtest"""
    try:
        if run_name not in running_backtests:
            return JSONResponse({
                "status": "not_found",
                "message": f"No backtest found with name '{run_name}'"
            })

        status_info = running_backtests[run_name]
        return JSONResponse({
            "status": status_info["status"],
            "new_count": status_info["new_count"],
            "run_name": run_name
        })

    except Exception as e:
        print(f"❌ Error in /api/backtest/status: {str(e)}")
        return JSONResponse({
            "status": "error",
            "error": str(e)
        }, status_code=500)


@app.post("/api/folders/delete")
async def delete_folder(request: DeleteFolderRequest):
    """🌙 Moon Dev: Delete a folder and all its contents"""
    try:
        folder_name = request.folder_name
        folder_path = USER_FOLDERS_DIR / folder_name

        if not folder_path.exists():
            return JSONResponse({
                "success": False,
                "message": f"Folder '{folder_name}' does not exist"
            }, status_code=404)

        # Delete the entire folder
        shutil.rmtree(folder_path)
        print(f"🗑️ Deleted folder '{folder_name}'")

        return JSONResponse({
            "success": True,
            "message": f"Deleted folder '{folder_name}'"
        })

    except Exception as e:
        print(f"❌ Error in /api/folders/delete: {str(e)}")
        import traceback
        traceback.print_exc()
        return JSONResponse({
            "success": False,
            "error": str(e),
            "message": "Failed to delete folder"
        }, status_code=500)


@app.post("/api/backtest/run")
async def run_backtest(request: BacktestRunRequest):
    """🌙 Moon Dev: Run rbi_agent_pp_multi.py with custom ideas"""
    try:
        ideas = request.ideas
        run_name = request.run_name

        print(f"\n🚀 Starting backtest run: '{run_name}'")
        print(f"📝 Ideas:\n{ideas}\n")

        # Create temp ideas file in the template base dir
        temp_ideas_file = TEMPLATE_BASE_DIR / f"temp_ideas_{run_name}.txt"
        with open(temp_ideas_file, 'w') as f:
            f.write(ideas)

        print(f"📁 Created temp ideas file: {temp_ideas_file}")

        # Path to rbi_agent_pp_multi.py
        script_path = Path(__file__).parent.parent / "agents" / "rbi_agent_pp_multi.py"

        if not script_path.exists():
            return JSONResponse({
                "success": False,
                "message": f"Script not found at {script_path}"
            }, status_code=404)

        # Create snapshot of CSV before running (for auto-add to folder later)
        csv_before_path = TEMPLATE_BASE_DIR / f"temp_csv_before_{run_name}.csv"
        if STATS_CSV.exists():
            shutil.copy(STATS_CSV, csv_before_path)
            print(f"📸 Created CSV snapshot for comparison")

        # Function to run in background
        def run_backtest_background():
            try:
                print(f"\n{'='*60}")
                print(f"🏃 Running backtest script for '{run_name}'...")
                print(f"{'='*60}\n")

                running_backtests[run_name] = {"status": "running", "new_count": 0}

                # Run the script with temp ideas file
                result = subprocess.run(
                    ["python", str(script_path), "--ideas-file", str(temp_ideas_file), "--run-name", run_name],
                    capture_output=True,
                    text=True,
                    timeout=3600  # 1 hour timeout
                )

                print(f"\n{'='*60}")
                print(f"✅ BACKTEST COMPLETED: '{run_name}'")
                print(f"{'='*60}")
                print(f"Return code: {result.returncode}")
                if result.returncode != 0:
                    print(f"⚠️ Script exited with non-zero code")

                # Show last 50 lines of output
                stdout_lines = result.stdout.split('\n')
                print(f"\n📊 Last 50 lines of output:")
                print('\n'.join(stdout_lines[-50:]))

                if result.stderr:
                    print(f"\n⚠️ STDERR:\n{result.stderr}")

                # Clean up temp file
                if temp_ideas_file.exists():
                    temp_ideas_file.unlink()
                    print(f"\n🗑️ Cleaned up temp ideas file")

                # Auto-add results to folder
                print(f"\n{'='*60}")
                new_count = auto_add_to_folder(run_name, str(csv_before_path))
                print(f"{'='*60}\n")

                running_backtests[run_name] = {"status": "complete", "new_count": new_count}

            except subprocess.TimeoutExpired:
                print(f"\n{'='*60}")
                print(f"❌ Backtest '{run_name}' timed out after 1 hour")
                print(f"{'='*60}\n")
                running_backtests[run_name] = {"status": "timeout", "new_count": 0}
            except Exception as e:
                print(f"\n{'='*60}")
                print(f"❌ Error running backtest '{run_name}': {str(e)}")
                print(f"{'='*60}")
                import traceback
                traceback.print_exc()
                running_backtests[run_name] = {"status": "error", "new_count": 0}

        # Start background thread
        thread = threading.Thread(target=run_backtest_background, daemon=True)
        thread.start()

        return JSONResponse({
            "success": True,
            "message": f"Backtest '{run_name}' started in background",
            "run_name": run_name
        })

    except Exception as e:
        print(f"❌ Error in /api/backtest/run: {str(e)}")
        import traceback
        traceback.print_exc()
        return JSONResponse({
            "success": False,
            "error": str(e),
            "message": "Failed to start backtest"
        }, status_code=500)


# ============================================================================
# 🌙 DATA PORTAL ROUTES
# ============================================================================

@app.get("/data", response_class=HTMLResponse)
async def data_portal(request: Request):
    """Render the data portal page"""
    return templates.TemplateResponse("data.html", {"request": request})


@app.get("/download/liquidations")
async def download_liquidations():
    """Download liquidation data"""
    file_path = DATA_DIR / "liquidations.csv"
    if file_path.exists():
        return FileResponse(
            file_path,
            media_type="text/csv",
            filename="moon_dev_liquidations.csv"
        )
    return JSONResponse({"error": "Data not available yet"}, status_code=404)


@app.get("/download/oi")
async def download_oi():
    """Download open interest data"""
    file_path = DATA_DIR / "oi.csv"
    if file_path.exists():
        return FileResponse(
            file_path,
            media_type="text/csv",
            filename="moon_dev_oi.csv"
        )
    return JSONResponse({"error": "Data not available yet"}, status_code=404)


@app.get("/download/testdata/{dataset_name}")
async def download_test_data(dataset_name: str):
    """Download test dataset for backtesting"""
    # 🌙 Moon Dev: Serve historical test datasets for backtesting
    file_path = TEST_DATA_DIR / f"{dataset_name}.csv"

    if not file_path.exists():
        return JSONResponse({"error": f"Dataset {dataset_name} not found"}, status_code=404)

    return FileResponse(
        file_path,
        media_type="text/csv",
        filename=f"moondev_testdata_{dataset_name}.csv"
    )


@app.get("/api/data-status")
async def get_data_status():
    """Get current data status"""
    return JSONResponse(data_status)


@app.post("/api/refresh-data")
async def refresh_data(background_tasks: BackgroundTasks):
    """Manually trigger data refresh"""
    background_tasks.add_task(fetch_all_data)
    return JSONResponse({"message": "Data refresh initiated"})


# ============================================================================
# 🌙 BACKTEST FOLDER OPERATIONS
# ============================================================================

def auto_add_to_folder(run_name: str, csv_before_path: str) -> int:
    """🌙 Moon Dev: Automatically add new winning backtests to a folder"""
    try:
        print(f"📁 Auto-adding results to folder '{run_name}'...")

        # Read main CSV
        if not STATS_CSV.exists():
            print(f"❌ Main CSV not found")
            return 0

        df_after = pd.read_csv(STATS_CSV)

        # Read CSV snapshot from before run
        if not Path(csv_before_path).exists():
            print(f"❌ Before-run CSV snapshot not found")
            return 0

        df_before = pd.read_csv(csv_before_path)

        # Find new rows (rows in df_after that aren't in df_before)
        # Simple approach: compare row counts and take the difference
        before_count = len(df_before)
        after_count = len(df_after)
        new_count = after_count - before_count

        if new_count <= 0:
            print(f"ℹ️ Zero backtests passed the {SAVE_IF_OVER_RETURN}% return threshold")
            return 0

        print(f"✅ Found {new_count} new backtest(s)")

        # Get the new rows (last N rows)
        new_rows = df_after.tail(new_count)

        # Create folder
        folder_path = USER_FOLDERS_DIR / run_name
        folder_path.mkdir(exist_ok=True)

        folder_csv = folder_path / "backtest_stats.csv"

        # Save new rows to folder CSV
        new_rows.to_csv(folder_csv, index=False)

        print(f"✅ Successfully added {new_count} backtest(s) to folder '{run_name}'")
        print(f"📂 Folder location: {folder_path}")

        # Clean up snapshot
        Path(csv_before_path).unlink()
        print(f"🗑️ Cleaned up CSV snapshot")

        return new_count

    except Exception as e:
        print(f"❌ Error in auto_add_to_folder: {str(e)}")
        import traceback
        traceback.print_exc()
        return 0


# ============================================================================
# 🌙 STARTUP EVENT
# ============================================================================

@app.on_event("startup")
async def startup_event():
    """Initialize scheduler and fetch data on startup"""
    logger.info("🌙 Moon Dev's AI Agent Backtests Dashboard starting up...")
    if TEST_MODE:
        logger.info("🧪 TEST MODE ENABLED - Using sample data for Data Portal")
    logger.info("")
    logger.info("🚀 Server is now available at: http://localhost:8001")
    logger.info("📊 Analysis Dashboard: http://localhost:8001/")
    logger.info("📊 Data Portal: http://localhost:8001/data")
    logger.info("📊 Data will begin downloading in the background...")
    logger.info("")

    # Setup scheduler for periodic data updates (every 5 minutes)
    scheduler = AsyncIOScheduler()

    # Schedule data fetch every 5 minutes
    scheduler.add_job(
        fetch_all_data,
        IntervalTrigger(minutes=5),
        id='fetch_all_data',
        name='Fetch all data every 5 minutes',
        replace_existing=True
    )

    # Start scheduler
    scheduler.start()
    logger.info("⏰ Scheduler started - will update data every 5 minutes")

    # Start data fetch in background (truly non-blocking)
    asyncio.create_task(background_data_fetch())


if __name__ == "__main__":
    print("\n" + "="*80)
    print("🌙 Moon Dev's AI Agent Backtests Dashboard 🚀")
    print("="*80)
    print(f"\n📊 CSV Path: {STATS_CSV}")
    print(f"📁 Templates: {TEMPLATE_BASE_DIR}")
    print(f"📂 Data Downloads: {DATA_DIR}")
    print(f"🌐 Starting server at: http://localhost:8001")
    print(f"\n💡 Page 1 (Analysis): http://localhost:8001/")
    print(f"💡 Page 2 (Data Portal): http://localhost:8001/data")
    if TEST_MODE:
        print(f"\n🧪 TEST MODE: Data portal will use sample data")
        print(f"   Set TEST_MODE = False in backtestdashboard.py for real data")
    print(f"\n💡 NOTE: Make sure you've run rbi_agent_pp_multi.py first to generate backtest data!")
    print(f"💡 Port 8001 is used to avoid conflict with main API on port 8000")
    print("\nPress CTRL+C to stop\n")

    uvicorn.run(app, host="0.0.0.0", port=8001, log_level="info")
