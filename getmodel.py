#!/usr/bin/env python3
"""This script no longer downloads models.

Place checkpoint files directly in models/checkpoints/ or models/model_set/ manually.
"""

import sys


def main() -> None:
    print(
        "This script no longer downloads models. "
        "Place checkpoint files in models/checkpoints/ or models/model_set/ manually."
    )
    sys.exit(0)


if __name__ == "__main__":
    main()
