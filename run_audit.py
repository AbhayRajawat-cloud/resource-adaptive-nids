#!/usr/bin/env python3
"""
Standalone runner for Stage 1 dataset audit.
Run from project root: python3 run_audit.py
"""
import sys
from pathlib import Path

# Ensure project root is on path
root = Path(__file__).resolve().parent
sys.path.insert(0, str(root))

from src.data.audit import run_full_audit, print_audit_summary

if __name__ == "__main__":
    report = run_full_audit()
    print_audit_summary(report)

