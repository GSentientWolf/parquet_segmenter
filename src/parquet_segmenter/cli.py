"""Tiny CLI for the parquet_segmenter testing generators.

Provides simple subcommands to list the binary store, generate a parquet
by target size, and generate an edge-case parquet. This is intentionally
small and meant for local testing only.
"""
from __future__ import annotations

import argparse
import sys
from typing import Optional

from parquet_segmenter.functional_testing.binary_store import PrecalculatedBinaryStore
from parquet_segmenter.functional_testing.parquet_gen import (
    generate_parquet_by_size,
    generate_parquet_with_edge_cases,
    generate_and_write_parquet,
)


def cmd_list(args: argparse.Namespace) -> int:
    """List binaries in the precalculated store and print an index table."""
    store = PrecalculatedBinaryStore(args.store_dir)
    idx = store.list_index()
    if not idx:
        print("(no binaries in store yet)")
        return 0
    print("index\tsize_bytes\tpath")
    for i, size, path in idx:
        print(f"{i}\t{size}\t{path}")
    return 0


def cmd_by_size(args: argparse.Namespace) -> int:
    """Generate a parquet file by target size using store indices."""
    store = PrecalculatedBinaryStore(args.store_dir)
    # parse comma separated indices
    indices = [int(x.strip()) for x in args.indices.split(",") if x.strip()]
    df, meta = generate_parquet_by_size(
        args.out,
        int(args.target),
        indices,
        binary_store=store,
        compression=None,
    )
    msg = f"wrote {args.out} target={meta['target']} produced={meta['produced']} rows={len(df)}"
    print(msg)
    return 0


def cmd_edge(args: argparse.Namespace) -> int:
    """Generate a parquet file with edge-case blobs."""
    store = PrecalculatedBinaryStore(args.store_dir)
    edge_sizes = None
    if args.edge_sizes:
        edge_sizes = [s.strip() for s in args.edge_sizes.split(",") if s.strip()]
    df, meta = generate_parquet_with_edge_cases(
        args.out,
        n_rows_mean=args.rows_mean,
        n_rows_std=args.rows_std,
        num_edge_cases=args.num_edge_cases,
        edge_sizes=edge_sizes,
        binary_store=store,
        compression=None,
    )
    msg = f"wrote {args.out} rows={len(df)} file_size={meta['file_size']}"
    print(msg)
    return 0


def cmd_simple(args: argparse.Namespace) -> int:
    """Convenience wrapper for generate_and_write_parquet."""
    store = PrecalculatedBinaryStore(args.store_dir) if args.store_dir else None
    df = generate_and_write_parquet(
        args.out,
        n=args.rows,
        categories=None,
        compression=None,
        binaries_dir=args.binaries_dir,
        binary_store=store,
        blob_strategy=args.blob_strategy,
        target_file_size_bytes=int(args.target) if args.target else None,
    )
    print(f"wrote {args.out} rows={len(df)}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    """Build and return the top-level argument parser for the CLI."""
    p = argparse.ArgumentParser(prog="parquet-seg-cli")
    sub = p.add_subparsers(dest="cmd")
    p_list = sub.add_parser(
        "list",
        help="List binaries in the precalculated store",
    )
    store_help = "Path to store directory (default: .cache/precalc_binaries)"
    p_list.add_argument("--store-dir", default=None, help=store_help)
    p_list.set_defaults(func=cmd_list)
    p_by = sub.add_parser(
        "by-size",
        help="Generate parquet by target size using store indices",
    )
    p_by.add_argument("--store-dir", default=None)
    p_by.add_argument("--out", required=True, help="Output parquet path")
    p_by.add_argument("--target", required=True, help="Target size in bytes")
    p_by.add_argument(
        "--indices",
        required=True,
        help=("Comma-separated store indices, e.g. 0,1,2"),
    )
    p_by.set_defaults(func=cmd_by_size)

    p_edge = sub.add_parser(
        "edge",
        help="Generate parquet with edge-case blobs",
    )
    p_edge.add_argument("--store-dir", default=None)
    p_edge.add_argument("--out", required=True)
    p_edge.add_argument("--rows-mean", dest="rows_mean", type=int, default=1000)
    p_edge.add_argument("--rows-std", dest="rows_std", type=int, default=200)
    p_edge.add_argument("--num-edge-cases", dest="num_edge_cases", type=int, default=3)
    p_edge.add_argument(
        "--edge-sizes",
        default=None,
        help=("Comma-separated sizes, e.g. '900 kB,1 MB'"),
    )
    p_edge.set_defaults(func=cmd_edge)

    p_simple = sub.add_parser(
        "simple",
        help="Random parquet writer with optional target size",
    )
    p_simple.add_argument("--out", required=True)
    p_simple.add_argument("--rows", type=int, default=100)
    p_simple.add_argument("--binaries-dir", dest="binaries_dir", default=None)
    p_simple.add_argument("--store-dir", default=None)
    p_simple.add_argument(
        "--blob-strategy",
        default="nearest",
        choices=("nearest", "index"),
    )
    p_simple.add_argument(
        "--target",
        default=None,
        help="Target total file size in bytes",
    )
    p_simple.set_defaults(func=cmd_simple)

    return p


def main(argv: Optional[list[str]] = None) -> int:
    """CLI entrypoint: parse arguments and dispatch to command handlers."""
    parser = build_parser()
    args = parser.parse_args(argv)
    if not hasattr(args, "func"):
        parser.print_help()
        return 2
    try:
        return args.func(args)
    except FileNotFoundError as e:
        # Friendly, concise message for missing precalculated binaries
        print(f"Error: {e}", file=sys.stderr)
        return 1
    except OSError as e:
        # Generic OS errors (disk issues, permissions) -> surface concisely
        print(f"OS error: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
