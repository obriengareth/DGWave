"""Diagnostics and CPU/MPI elastic solve entry point."""

import argparse
import json

from dg_wave import __version__
from dg_wave.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser(
        description="DGWave: CPU/MPI heterogeneous elastic solves and Stage 0 FP64 diagnostics.",
        epilog=(
            "Four ranks: mpiexec --nooversubscribe --bind-to core --map-by core "
            "-n 4 .venv/bin/python -m dg_wave --config stage0.toml --diagnostics. "
            "Solve: dg-wave run elastic_box.toml; MPI runs use the same binding flags. "
            "Stage 1 operator benchmark: python -m dg_wave.benchmark --help."
        ),
    )
    parser.add_argument("--version", action="version", version=__version__)
    parser.add_argument("--config", metavar="TOML", help="validate a solver TOML file without running")
    parser.add_argument(
        "--diagnostics", action="store_true",
        help="run CPU float64 computation, host-buffer MPI check and JSON diagnostics",
    )
    commands = parser.add_subparsers(dest="command")
    run_parser = commands.add_parser("run", help="run an elastic box on CPU/MPI")
    run_parser.add_argument("input", metavar="TOML")
    run_parser.add_argument("--checkpoint", metavar="DIR", help="atomically checkpoint full state/history at this invocation's last step")
    run_parser.add_argument("--restart", metavar="DIR", help="resume a compatible same-process-count checkpoint")
    run_parser.add_argument("--extend-run-end", action="store_true",
                            help="allow only a larger time.steps on restart; all source/physics/schedule metadata must match")
    run_parser.add_argument("--stop-after", type=int, metavar="STEP", help="stop at this absolute step; requires --checkpoint")
    args = parser.parse_args()
    if getattr(args, "extend_run_end", False) and not args.restart:
        parser.error("--extend-run-end requires --restart")
    if args.command is not None and (args.config or args.diagnostics):
        parser.error("run cannot be combined with --config or --diagnostics")
    if not args.config and not args.diagnostics and args.command is None:
        parser.print_help()
        return
    try:
        if args.command == "run":
            from dg_wave.runtime import initialize_mpi_cpu

            comm, _, _ = initialize_mpi_cpu()
            config, error = None, None
            if comm.rank == 0:
                try:
                    config = load_config(args.input)
                except (OSError, ValueError) as exc:
                    error = str(exc)
            error = comm.bcast(error)
            if error is not None:
                raise ValueError(error)
            config = comm.bcast(config)
            if comm.size == 1:
                from dg_wave.run import run_simulation

                summary = run_simulation(
                    config, checkpoint=args.checkpoint, restart=args.restart, stop_after=args.stop_after,
                    extend_run_end=args.extend_run_end,
                )
            else:
                from dg_wave.mpi_run import run_distributed

                summary = run_distributed(
                    config, comm, checkpoint=args.checkpoint, restart=args.restart, stop_after=args.stop_after,
                    extend_run_end=args.extend_run_end,
                )
            if comm.rank == 0:
                print(json.dumps(summary, indent=2))
            return
        if args.config:
            load_config(args.config)
        if args.diagnostics:
            from dg_wave.runtime import diagnostics

            records = diagnostics()
            if records is not None:
                print(json.dumps({"ranks": records}, indent=2))
        else:
            print("Configuration valid (Stage 0; mesh not loaded, CFL not yet checked).")
    except (OSError, ValueError, RuntimeError) as error:
        parser.exit(2, f"dg-wave: error: {error}\n")


if __name__ == "__main__":
    main()
