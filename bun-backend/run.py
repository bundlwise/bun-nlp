#!/usr/bin/env python
import os
import sys
import uvicorn

# Add the src directory to the path
src_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "src")
sys.path.append(src_path)

if __name__ == "__main__":
    print("Starting Subscription Email Analysis API...")
    print(f"API will be available at http://localhost:8000")
    print(f"Documentation will be available at http://localhost:8000/docs")
    uvicorn.run("src.app:app", host="0.0.0.0", port=8000, reload=True) 