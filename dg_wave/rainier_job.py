"""Approval-only launcher's bounded MPI worker; no work occurs on import."""


def main():
    import json
    import os
    import resource
    import sys

    limit = 3_000_000_000
    _, hard = resource.getrlimit(resource.RLIMIT_AS)
    if hard != resource.RLIM_INFINITY:
        limit = min(limit, hard)
    resource.setrlimit(resource.RLIMIT_AS, (limit, limit))
    os.environ["MALLOC_ARENA_MAX"] = "2"
    from dg_wave.runtime import initialize_mpi_cpu, physical_binding

    comm, _, local_rank = initialize_mpi_cpu()
    binding = physical_binding(comm, local_rank)
    if comm.size != 4:
        raise RuntimeError("approved Rainier job requires exactly four pinned MPI ranks")
    bindings = comm.gather(binding)
    if comm.rank == 0:
        print("RAINIER_BINDINGS "+json.dumps(bindings), file=sys.stderr, flush=True)
    from dg_wave.__main__ import main as cli

    try:
        cli()
    except MemoryError:
        print(f"Rainier rank{comm.rank}: allocation rejected under {limit} byte virtual-address safety cap",
              file=sys.stderr, flush=True)
        comm.Abort(2)


if __name__ == "__main__":
    main()
