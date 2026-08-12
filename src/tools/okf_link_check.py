#!/usr/bin/env python3
"""
OKF KB Link Checker Tool
------------------------
Generalized link checker using src.okf.validator core engine.

Usage:
    python src/tools/okf_link_check.py [path_to_okf] [-v]
"""

import sys
import argparse
from pathlib import Path

# Add project root to sys.path
project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.okf.validator import OKFValidator


def main():
    parser = argparse.ArgumentParser(description="OKF KB Link & Integrity Checker")
    parser.add_argument("path", nargs="?", default=str(project_root / ".okf"), help="Path to .okf directory")
    parser.add_argument("-v", "--verbose", action="store_true", help="Print verbose warnings")
    args = parser.parse_args()

    okf_path = Path(args.path)
    if not okf_path.exists():
        print(f"❌ Error: Path '{okf_path}' does not exist.")
        sys.exit(1)

    validator = OKFValidator(okf_path)
    report = validator.run_validation(verbose=args.verbose)
    report.print_summary(verbose=args.verbose)

    sys.exit(0 if report.is_valid else 1)


if __name__ == "__main__":
    main()