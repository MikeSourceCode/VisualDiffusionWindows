#!/usr/bin/env python3
"""This script no longer downloads LoRAs.

Place LoRA files directly in models/lora/ instead.
"""

import sys


def main() -> None:
    print("This script no longer downloads LoRAs. Place LoRA files in models/lora/ manually.")
    sys.exit(0)


if __name__ == "__main__":
    main()
