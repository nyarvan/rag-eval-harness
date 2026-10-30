"""Shared argparse types for the command line tools."""
import argparse


def positive_int(text: str) -> int:
    """Parse a strictly positive integer, for counts such as `--n`."""
    try:
        value = int(text)
    except ValueError:
        raise argparse.ArgumentTypeError(
            f'expected a positive integer, got {text!r}'
        ) from None
    if value <= 0:
        raise argparse.ArgumentTypeError(
            f'expected a positive integer, got {value}'
        )
    return value
