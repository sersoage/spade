#!/usr/bin/env python3
"""SPADE training wrapper for Slime.

This script wraps Slime's training loop with SPADE-specific argument handling.
It uses Slime's extension points to add SPADE arguments without modifying Slime's core.
"""

from slime.train import train


def main():
    """Parse arguments with SPADE extensions and run training."""
    from slime.utils.arguments import parse_args

    try:
        from spade.slime.arguments import add_spade_arguments

        args = parse_args(add_custom_arguments=add_spade_arguments)
    except ImportError as e:
        print(f"Warning: SPADE arguments not available ({e}), using default Slime args")
        args = parse_args()

    train(args)


if __name__ == "__main__":
    main()
