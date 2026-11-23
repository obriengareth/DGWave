"""Run ONLY after explicit approval; no pilot and no interactive resource-review gate."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import resource
import signal
import subprocess
import sys
import time


ROOT = Path(__file__).parents[1]
SETUP = ROOT / "output" / "rainier_run_setup"
LIMIT = 15_000_000_000
RESERVE = 2_000_000_000


def memory_available(keepawake_pid, wsl_client_pid=None):
    pids = ",".join(str(int(pid)) for pid in (keepawake_pid, wsl_client_pid) if pid is not None)
    command = (
        "$ErrorActionPreference='Stop'; "
        f"$rss=(Get-Process -Id @({pids},$PID) -ErrorAction Stop | Measure-Object WorkingSet64 -Sum).Sum; "
        "$free=[double](Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory*1024; "
        "@{available=$free;job_rss=$rss}|ConvertTo-Json -Compress"
    )
    result = subprocess.run(
        ["/mnt/c/Windows/System32/WindowsPowerShell/v1.0/powershell.exe", "-NoProfile", "-NonInteractive",
         "-Command", command],
        text=True, capture_output=True, check=True, timeout=15,
    )
    info = json.loads(result.stdout.strip().lstrip("\ufeff"))
    windows = float(info["available"])
    linux = next(int(line.split()[1])*1024 for line in Path("/proc/meminfo").read_text().splitlines()
                 if line.startswith("MemAvailable:"))
    return windows, linux, int(info["job_rss"])


def tree_sample(pid):
    pending, seen, rss, ticks, ranks = [pid], set(), 0, 0, []
    while pending:
        current = pending.pop()
        if current in seen:
            continue
        seen.add(current)
        path = Path(f"/proc/{current}")
        try:
            lines = (path/"status").read_text().splitlines()
            resident = next((int(line.split()[1])*1024 for line in lines if line.startswith("VmRSS:")), 0)
            rss += resident
            stat = (path/"stat").read_text().rsplit(")", 1)[1].split()
            ticks += int(stat[11])+int(stat[12])
            command = (path/"cmdline").read_bytes().split(b"\0")
            if command and b"python" in command[0] and b"dg_wave.rainier_job" in command:
                environment = (path/"environ").read_bytes().split(b"\0")
                rank = next(int(item.split(b"=", 1)[1]) for item in environment
                            if item.startswith(b"OMPI_COMM_WORLD_RANK="))
                ranks.append({"rank": rank, "pid": current, "rss_bytes": resident,
                              "affinity_cpus": next(line.split(":", 1)[1].strip()
                                                    for line in lines if line.startswith("Cpus_allowed_list:"))})
            pending.extend(map(int, (path/"task"/str(current)/"children").read_text().split()))
        except FileNotFoundError:
            continue
    return rss, ticks, sorted(ranks, key=lambda row: row["rank"])


def atomic_json(path, data):
    temporary = path.with_suffix(path.suffix+".tmp")
    with temporary.open("w") as stream:
        json.dump(data, stream, indent=2)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def mpi_limits():
    # Workers explicitly raise this soft limit to the fixed 3GB hard limit.
    resource.setrlimit(resource.RLIMIT_AS, (500_000_000, 3_000_000_000))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--approve-run", action="store_true", help="Use only AFTER user approves the saved source/run proposal")
    parser.add_argument("--status-path", type=Path, default=SETUP/"live-status.json")
    parser.add_argument("--keepawake-info", type=Path)
    parser.add_argument("--no-host-reserve", action="store_true", help="Explicit user-approved removal of free-memory reserve; job memory caps remain active")
    parser.add_argument("--attempt-id", default="rainier-attempt-001")
    parser.add_argument("--setup-dir", type=Path, help="separate approved configuration/log directory")
    parser.add_argument("--restart", type=Path, help="approved checkpoint for run-end continuation")
    args = parser.parse_args()
    setup = args.setup_dir or SETUP
    reserve = 0 if args.no_host_reserve else RESERVE
    if not args.approve_run:
        parser.error("no job started: explicit approval and --approve-run are required")
    approval = json.loads((setup/"approval.json").read_text())
    if approval.get("run_blocked", False):
        raise ValueError(approval["blocked_reason"])
    config = setup/"rainier.toml"
    if hashlib.sha256(config.read_bytes()).hexdigest() != approval["config_sha256"]:
        raise ValueError("configuration changed since the saved run proposal")
    model = ROOT/"output"/"rainier_layered_model"
    for name, digest in approval["protected_model_file_sha256"].items():
        if hashlib.sha256((model/name).read_bytes()).hexdigest() != digest:
            raise ValueError(f"model changed since run proposal: {name}")
    if args.keepawake_info is None:
        raise ValueError("use the Windows job wrapper so a scoped sleep-prevention request owns the job lifetime")
    keepawake = json.loads(args.keepawake_info.read_text())
    if not keepawake["active"]:
        raise RuntimeError("Windows sleep prevention is not active")
    windows, linux, windows_rss = memory_available(keepawake["pid"], keepawake.get("wsl_client_pid"))
    if reserve and min(windows, linux) < reserve:
        raise RuntimeError("automatic safety stop: insufficient host/WSL headroom")
    from dg_wave.config import load_config

    parsed = load_config(config)
    destination = parsed.output
    checkpoint_destination = ROOT/"output"/"rainier_checkpoint"
    continuation = approval.get("continuation")
    if args.restart is not None:
        if continuation is None or str(args.restart.resolve()) != continuation["restart_directory"]:
            raise ValueError("restart path is not the approved continuation checkpoint")
        if hashlib.sha256((args.restart/"complete.json").read_bytes()).hexdigest() != continuation["checkpoint_pointer_sha256"]:
            raise ValueError("approved restart checkpoint generation changed")
        checkpoint_destination = Path(continuation["checkpoint_directory"])
        if checkpoint_destination.resolve() == args.restart.resolve() or checkpoint_destination.exists():
            raise ValueError("continuation requires a new checkpoint destination")
        seeded = continuation["seeded_surface_sha256"]
        existing = {str(path.relative_to(destination)) for path in destination.rglob("*") if path.is_file()}
        if existing != set(seeded):
            raise ValueError("continuation output must contain only the approved copied surface history")
        for name, digest in seeded.items():
            if hashlib.sha256((destination/name).read_bytes()).hexdigest() != digest:
                raise ValueError(f"copied restart surface history changed: {name}")
    elif continuation is not None:
        raise ValueError("approved continuation requires --restart; refusing a run from zero")
    if destination.exists():
        if args.restart is None:
            raise ValueError("refusing to overwrite an existing Rainier run")
    command = [
        "mpiexec", "--nooversubscribe", "--bind-to", "core", "--map-by", "core", "-n", "4",
        sys.executable, "-m", "dg_wave.rainier_job", "run", str(config),
        "--checkpoint", str(checkpoint_destination),
    ]
    if args.restart is not None:
        command.extend(["--restart", str(args.restart), "--extend-run-end"])
    progress_path = setup/"progress.json"
    env = {**os.environ, "MALLOC_ARENA_MAX": "2", "DG_WAVE_LIVE_PROGRESS": str(progress_path)}
    started, peak, minimum = time.perf_counter(), 0, [windows, linux]
    failure = None
    previous_cpu, previous_step = None, None
    live = {
        "state": "starting", "stage": "MPI startup", "step": continuation["start_step"] if continuation else 0,
        "total_steps": parsed.steps,
        "attempt_id": args.attempt_id,
        "simulation_time_s": continuation["start_step"]*parsed.dt if continuation else 0,
        "duration_s": parsed.steps*parsed.dt, "launcher_pid": os.getpid(),
        "restart_origin": continuation,
        "keepawake": keepawake,
        "guards": {key: value for key, value in approval["safeguards"].items()
                   if key != "no_pilot_or_simulation_performed"},
        "errors": [],
        "result_paths": {"output": str(destination/"mpi4"), "checkpoint": str(checkpoint_destination),
                         "stdout": str(setup/"run.stdout"), "stderr": str(setup/"run.stderr")},
        "cpu_definition": "process-tree CPU time delta / wall delta;400% is four occupied logical CPUs",
        "rss_definition": "Linux process-tree VmRSS including launcher + most recent Windows helper/WSL client/memory-query working sets (<=5s skew); shared pages may repeat; sampled, not unique RAM",
    }
    live["guards"].update(host_reserve_bytes=reserve, free_memory_reserve_enabled=bool(reserve))
    live["guards"]["monitor"] = (
        "Linux process-tree RSS every0.2s plus Windows helper/client/query working sets every5s; "
        f"atomic status every10s; free-memory reserve {reserve} bytes;15GB job limit retained"
    )

    def publish(state, now, rss, ticks, ranks):
        nonlocal previous_cpu, previous_step
        progress = json.loads(progress_path.read_text()) if progress_path.exists() else {}
        live.update(progress)
        live.update(state=state, timestamp_utc=datetime.now(timezone.utc).isoformat(),
                    elapsed_seconds=now-started, concurrent_rss_bytes=rss+windows_rss, peak_sampled_rss_bytes=peak,
                    linux_process_tree_rss_bytes=rss, windows_helper_client_query_rss_bytes=windows_rss,
                    ranks=ranks, host_available_bytes={"windows": windows, "wsl": linux})
        if previous_cpu is not None:
            elapsed = now-previous_cpu[0]
            live["cpu_percent"] = max(0., 100*(ticks-previous_cpu[1])/os.sysconf("SC_CLK_TCK")/elapsed)
            live["occupied_logical_cpus"] = live["cpu_percent"]/100
        previous_cpu = now, ticks
        if state == "running" and progress.get("stage") == "solve":
            step = progress["step"]
            if previous_step is None:
                previous_step = now, step
            elif step > previous_step[1]:
                rate = (step-previous_step[1])/(now-previous_step[0])
                live.update(recent_steps_per_second=rate, eta_seconds=(live["total_steps"]-step)/rate)
                previous_step = now, step
        else:
            previous_step = None
            live.pop("eta_seconds", None)
            live.pop("recent_steps_per_second", None)
        atomic_json(args.status_path, live)

    with (setup/"run.stdout").open("x") as out, (setup/"run.stderr").open("x") as err:
        process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=out, stderr=err,
                                   start_new_session=True, preexec_fn=mpi_limits)
        live["mpiexec_pid"] = process.pid
        next_host, next_status = 0., 0.
        try:
            while process.poll() is None:
                # The launcher's own RSS is included; shared pages may be counted twice.
                rss, ticks, ranks = tree_sample(os.getpid())
                now = time.perf_counter()
                wsl = next(int(line.split()[1])*1024 for line in Path("/proc/meminfo").read_text().splitlines()
                           if line.startswith("MemAvailable:"))
                minimum[1] = min(minimum[1], wsl)
                if now >= next_host:
                    keepawake = json.loads(args.keepawake_info.read_text())
                    windows, linux, windows_rss = memory_available(keepawake["pid"], keepawake.get("wsl_client_pid"))
                    minimum = [min(minimum[0], windows), min(minimum[1], linux)]
                    next_host = now+5
                peak = max(peak, rss+windows_rss)
                if peak >= LIMIT-1_000_000_000:
                    raise RuntimeError(f"automatic safety stop: sampled job RSS {peak} >= {LIMIT-1_000_000_000} bytes")
                if reserve and min(minimum) < reserve:
                    system = "Windows" if minimum[0] < reserve else "WSL"
                    available = minimum[0] if system == "Windows" else minimum[1]
                    raise RuntimeError(f"automatic safety stop: {system} available RAM {available} < {reserve} byte reserve")
                if now >= next_status:
                    publish("running", now, rss, ticks, ranks)
                    next_status = now+10
                time.sleep(.2)
        except (OSError, RuntimeError, subprocess.SubprocessError) as error:
            failure = str(error)
        finally:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
    result = {
        "returncode": process.returncode, "wall_seconds": time.perf_counter()-started,
        "sampled_aggregate_rss_peak_bytes": peak, "minimum_available_windows_wsl_bytes": minimum,
        "total_job_budget_bytes": LIMIT, "rss_abort_threshold_bytes": LIMIT-1_000_000_000,
        "attempt_id": args.attempt_id, "host_free_memory_reserve_bytes": reserve,
        "per_rank_hard_virtual_address_limit_bytes": 3_000_000_000,
        "memory_caveat": live["rss_definition"]+"; not a kernel aggregate memory quota; hard per-worker virtual limits may reject JIT below the RSS budget",
        "command": command,
        "safety_error": failure,
    }
    (setup/"run-job.json").write_text(json.dumps(result, indent=2))
    live["errors"] = [failure] if failure else ([] if process.returncode == 0 else ["MPI job failed; inspect run.stderr"])
    live["returncode"] = process.returncode
    publish("completed" if process.returncode == 0 and failure is None else "failed",
            time.perf_counter(), *tree_sample(os.getpid()))
    if failure:
        raise RuntimeError(failure)
    if process.returncode:
        raise RuntimeError("Rainier job failed; inspect run.stderr (including conservative allocator/virtual-limit failures)")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
