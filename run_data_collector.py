"""
Wrapper script to run data collector with emoji handling for Windows
"""
import sys
import os

# Force UTF-8 encoding for stdout
if sys.platform == 'win32':
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, errors='replace')
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, errors='replace')

# Now import and run the data collector
from src.agents.polymarket_data_collector import main

if __name__ == "__main__":
    main()
