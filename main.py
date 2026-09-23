"""Main entry point for asset-observation."""

import sys
from src.cli import main as cli_main


def main():
    if len(sys.argv) == 1:
        # Show help if invoked with no arguments
        cli_main(["--help"])
    else:
        sys.exit(cli_main())


if __name__ == "__main__":
    main()
