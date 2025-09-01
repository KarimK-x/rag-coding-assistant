#!/usr/bin/env python3
"""
Setup script for the RAG Coding Assistant project.
"""

import subprocess
import sys
import os
from pathlib import Path


def run_command(command, description):
    """Run a command and handle errors."""
    print(f"\n{description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True) #check = True raises CalledProcessError
        print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed:")
        print(f"Error: {e.stderr}")
        return False


def main():
    """Setup the project environment."""
    print("🚀 Setting up RAG Coding Assistant...")
    
    # Check Python version
    print(f"Python version: {sys.version}")
    if sys.version_info < (3, 8):
        print("❌ Python 3.8 or higher is required")
        sys.exit(1)
    
    # Create necessary directories
    print("\n📁 Creating project directories...")
    directories = ["data"]
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        print(f"✅ Created directory: {directory}")
    
    # Install dependencies
    if not run_command("pip install -r requirements.txt", "Installing dependencies"):
        print("❌ Failed to install dependencies. Please check your internet connection and try again.")
        sys.exit(1)
    
    # Install the package in development mode
    if not run_command("pip install -e .", "Installing package in development mode"):
        print("❌ Failed to install package")
        sys.exit(1)
    
    print("\n🎉 Setup completed successfully!")
    print("\n📖 Next steps:")
    print("1. Run the CLI version: python main.py --mode cli")
    print("2. Run the Gradio interface: python main.py --mode gradio") 
    print("3. Run evaluation: python main.py --mode evaluation")
    print("\n📚 For more information, see README.md")


if __name__ == "__main__":
    main()
