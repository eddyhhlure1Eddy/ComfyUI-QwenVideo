"""
Installation script for ComfyUI-QwenVideo node
Author: eddy
"""

import subprocess
import sys
import os
from pathlib import Path


def check_ffmpeg():
    """Check if ffmpeg is installed"""
    try:
        result = subprocess.run(
            ['ffmpeg', '-version'],
            capture_output=True,
            timeout=5
        )
        if result.returncode == 0:
            print("✓ ffmpeg is installed")
            return True
        else:
            print("✗ ffmpeg not found")
            return False
    except Exception:
        print("✗ ffmpeg not found")
        return False


def install_requirements():
    """Install Python requirements"""
    print("\nInstalling Python dependencies...")
    requirements_file = Path(__file__).parent / "requirements.txt"
    
    if not requirements_file.exists():
        print("✗ requirements.txt not found")
        return False
    
    try:
        subprocess.check_call([
            sys.executable,
            "-m",
            "pip",
            "install",
            "-r",
            str(requirements_file)
        ])
        print("✓ Python dependencies installed")
        return True
    except subprocess.CalledProcessError:
        print("✗ Failed to install Python dependencies")
        return False


def main():
    print("=" * 70)
    print("ComfyUI Qwen Video Node Installation")
    print("=" * 70)
    
    # Check ffmpeg
    print("\n[1/2] Checking ffmpeg...")
    ffmpeg_ok = check_ffmpeg()
    
    if not ffmpeg_ok:
        print("\nWARNING: ffmpeg is required for video frame extraction")
        print("Download from: https://ffmpeg.org/download.html")
        print("Make sure to add ffmpeg to your system PATH")
    
    # Install requirements
    print("\n[2/2] Installing dependencies...")
    deps_ok = install_requirements()
    
    # Summary
    print("\n" + "=" * 70)
    print("Installation Summary")
    print("=" * 70)
    print(f"ffmpeg: {'✓ OK' if ffmpeg_ok else '✗ NOT FOUND (required)'}")
    print(f"Python deps: {'✓ OK' if deps_ok else '✗ FAILED'}")
    
    if ffmpeg_ok and deps_ok:
        print("\n✓ Installation completed successfully!")
        print("\nNext steps:")
        print("1. Restart ComfyUI")
        print("2. Look for 'Qwen Video Prompt Reversal' in the node menu")
        print("   under 'video/analysis' category")
        return 0
    else:
        print("\n✗ Installation incomplete")
        if not ffmpeg_ok:
            print("   Please install ffmpeg and try again")
        if not deps_ok:
            print("   Please check Python package installation errors")
        return 1


if __name__ == "__main__":
    sys.exit(main())
