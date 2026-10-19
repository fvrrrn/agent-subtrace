"""Offline evidence operations. No implicit inference or experiment resumption."""

import argparse
import json

from .legacy import reproduce_safety
from .inventory import inventory


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    reproduce = commands.add_parser("reproduce-safety", help="Verify an archived NOUL sweep without loading model weights")
    reproduce.add_argument("--source", required=True)
    reproduce.add_argument("--output", required=True)
    index = commands.add_parser("inventory", help="Hash and index historical report/configuration files")
    index.add_argument("--source", required=True)
    index.add_argument("--output", required=True)
    args = parser.parse_args()
    operation = reproduce_safety if args.command == "reproduce-safety" else inventory
    result = operation(args.source, args.output)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
