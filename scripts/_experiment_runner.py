"""Shared runner for Phase-1 E1.x experiment scripts.

Each E1.x script builds a list of (run_name, command) pairs; this module
handles the execute / print / smoke modes uniformly so the three scripts
don't each re-implement the loop.

A `command` here is a list[str] ready for `subprocess.run`. The headless
EGL env vars are injected via `os.environ` on the child process — not
prepended to the argv — so Popen sees them without needing a shell.
"""

from __future__ import annotations

import argparse
import datetime as dt
import os
import shlex
import signal
import subprocess
import sys
from pathlib import Path
from typing import Callable, Iterable, List, Tuple

HEADLESS_ENV = {"MUJOCO_GL": "egl", "PYOPENGL_PLATFORM": "egl"}


def require_amass() -> str:
    amass = os.environ.get("AMASS_DATA_DIR")
    if not amass:
        sys.stderr.write(
            "AMASS_DATA_DIR is not set. Export it to the CMU AMASS root.\n"
        )
        sys.exit(1)
    return amass


def format_printed_line(cmd: List[str]) -> str:
    """Render a command the way a user would paste it into a shell."""
    env_prefix = " ".join(f"{k}={v}" for k, v in HEADLESS_ENV.items())
    return f"{env_prefix} {shlex.join(cmd)}"


def _child_env() -> dict:
    env = os.environ.copy()
    env.update(HEADLESS_ENV)
    return env


def run_experiment(
    name: str,
    log_subdir: str,
    build_commands: Callable[[argparse.Namespace, str], Iterable[Tuple[str, List[str]]]],
    extra_args: Callable[[argparse.ArgumentParser], None] | None = None,
) -> None:
    """Drive one E1.x experiment.

    Parameters
    ----------
    name:
        Human-readable experiment name, used in logs and stdout.
    log_subdir:
        Path under `experiments/` where `runs_<tag>.txt` is written.
    build_commands:
        Function receiving (argparse.Namespace, timestamp_tag) and yielding
        (run_name, argv_list) tuples. One tuple per training run.
    extra_args:
        Optional hook to register experiment-specific CLI flags.
    """
    parser = argparse.ArgumentParser(description=f"Phase-1 {name} runner.")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--run", action="store_true",
        help="Execute every command serially on the current machine.",
    )
    mode.add_argument(
        "--smoke", action="store_true",
        help="Execute the first command with num_train_frames=100, wandb disabled.",
    )
    mode.add_argument(
        "--print-only", action="store_true",
        help="Print commands and log them to experiments/... (default when no mode given).",
    )
    parser.add_argument(
        "--continue-on-error", action="store_true",
        help="Keep going after a failing run instead of stopping the sweep.",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="With --run: print what would execute, do not invoke.",
    )
    if extra_args is not None:
        extra_args(parser)
    args = parser.parse_args()

    require_amass()
    repo_root = Path(__file__).resolve().parent.parent
    tag = dt.datetime.now().strftime("%Y%m%d%H%M%S")
    log_dir = repo_root / "experiments" / log_subdir
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"runs_{tag}.txt"

    commands = list(build_commands(args, tag))
    if not commands:
        sys.stderr.write("No commands to run (check --tasks / --modes filters).\n")
        sys.exit(2)

    default_print = not (args.run or args.smoke)

    if args.smoke:
        run_name, cmd = commands[0]
        smoke_cmd = list(cmd) + ["num_train_frames=100", "wandb.use=false"]
        print(f"# [{name}] smoke run: {run_name}")
        print(format_printed_line(smoke_cmd))
        if args.dry_run:
            return
        rc = _invoke(smoke_cmd, cwd=repo_root)
        sys.exit(rc)

    # Always write the full command log, regardless of mode.
    with log_path.open("w") as fh:
        for run_name, cmd in commands:
            fh.write(f"# {run_name}\n{format_printed_line(cmd)}\n")
    print(f"# [{name}] Logged {len(commands)} commands to {log_path}")

    if default_print:
        for _, cmd in commands:
            print(format_printed_line(cmd))
        return

    # --run path.
    print(f"# [{name}] Running {len(commands)} commands serially.")
    print(f"# [{name}] Ctrl+C interrupts the current run and aborts the sweep.")
    results: list[tuple[str, int]] = []
    start = dt.datetime.now()
    try:
        for i, (run_name, cmd) in enumerate(commands, start=1):
            header = f"=== [{name}] {i}/{len(commands)} — {run_name} ==="
            print()
            print(header)
            print(format_printed_line(cmd))
            if args.dry_run:
                results.append((run_name, 0))
                continue
            rc = _invoke(cmd, cwd=repo_root)
            results.append((run_name, rc))
            if rc != 0 and not args.continue_on_error:
                print(
                    f"# [{name}] run '{run_name}' exited {rc}; aborting sweep. "
                    f"Pass --continue-on-error to skip past failures."
                )
                break
    finally:
        elapsed = dt.datetime.now() - start
        _write_summary(log_path, results, elapsed)
        _print_summary(results, elapsed, name)

    failed = [n for n, rc in results if rc != 0]
    sys.exit(1 if failed else 0)


def _invoke(cmd: List[str], *, cwd: Path) -> int:
    """Run `cmd` with the headless env and forward SIGINT/SIGTERM cleanly."""
    try:
        proc = subprocess.Popen(cmd, cwd=cwd, env=_child_env())
    except FileNotFoundError as e:
        print(f"# failed to start: {e}", file=sys.stderr)
        return 127
    try:
        return proc.wait()
    except KeyboardInterrupt:
        proc.send_signal(signal.SIGINT)
        try:
            return proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            proc.kill()
            return 130
        finally:
            # Re-raise so the outer loop breaks.
            raise


def _write_summary(
    log_path: Path,
    results: list[tuple[str, int]],
    elapsed: dt.timedelta,
) -> None:
    with log_path.open("a") as fh:
        fh.write("\n# ---- results ----\n")
        for name, rc in results:
            fh.write(f"# {name}: rc={rc}\n")
        fh.write(f"# total_elapsed={elapsed}\n")


def _print_summary(
    results: list[tuple[str, int]],
    elapsed: dt.timedelta,
    name: str,
) -> None:
    ok = sum(1 for _, rc in results if rc == 0)
    fail = len(results) - ok
    print()
    print(f"# [{name}] summary: {ok} ok, {fail} failed, elapsed {elapsed}")
    for run_name, rc in results:
        flag = "OK " if rc == 0 else f"FAIL({rc})"
        print(f"#   {flag}  {run_name}")
