#!/usr/bin/env python3
"""Asset Tracking System - Demo Runner.

Quick script to run the Streamlit demo application.
"""

import subprocess
import sys
from pathlib import Path


def main():
    """Run the Streamlit demo."""
    demo_path = Path(__file__).parent / "demo" / "streamlit_demo.py"
    
    if not demo_path.exists():
        print(f"Demo file not found: {demo_path}")
        sys.exit(1)
    
    print("Starting Asset Tracking System Demo...")
    print("Access the demo at: http://localhost:8501")
    print("Press Ctrl+C to stop the demo")
    
    try:
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", 
            str(demo_path),
            "--server.port", "8501",
            "--server.address", "localhost"
        ])
    except KeyboardInterrupt:
        print("\nDemo stopped by user")
    except Exception as e:
        print(f"Error running demo: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
