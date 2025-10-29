#!/usr/bin/env python
"""
Wrapper script to run Pinescript Converter with proper encoding
"""

import sys
import codecs
if sys.platform == 'win32':
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, errors='replace')
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, errors='replace')

from src.agents.pinescript_converter_agent import main

if __name__ == "__main__":
    main()
