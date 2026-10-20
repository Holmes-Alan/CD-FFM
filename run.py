#!/usr/bin/env python3
"""
run.py -- run a command list from configs/commands/ on this computer, line by line, in scripts/.

Every non-comment line is one command (`python -u <script>.py ...`); `python` is replaced by the interpreter that
runs this file, $CDFFM_ROOT is expanded, and output is shown and saved to logs/<list>_<line>.log.

  python run.py configs/commands/08_eval.txt                    # all lines, in order
  python run.py configs/commands/08_eval.txt --lines 0-2,5      # selected lines (0-based, as listed by --list)
  python run.py configs/commands/08_eval.txt --grep w5_flowmap  # lines containing a string
  python run.py configs/commands/05_train.txt --list            # show the numbered lines, run nothing
  python run.py configs/commands/05_train.txt --dry-run         # print the final commands, run nothing

Stops at the first failing command unless --keep-going. CDFFM_ROOT defaults to ./work (see scripts/paths.py).
"""
import argparse
import os
import re
import shlex
import subprocess
import sys
import time

REPO = os.path.dirname(os.path.abspath(__file__))


def read_lines(path):
    with open(path) as f:
        return [l.strip() for l in f if l.strip() and not l.lstrip().startswith("#")]


def parse_sel(sel, n):
    out = []
    for part in sel.split(","):
        a, _, b = part.partition("-")
        out += list(range(int(a), (int(b) if b else int(a)) + 1))
    bad = [i for i in out if not 0 <= i < n]
    if bad:
        raise SystemExit(f"line numbers out of range 0..{n - 1}: {bad}")
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("commands", help="a file in configs/commands/")
    ap.add_argument("--lines", default=None, help="e.g. 0-3,7 (0-based)")
    ap.add_argument("--grep", default=None, help="only lines containing this string")
    ap.add_argument("--list", action="store_true", help="list the numbered command lines and exit")
    ap.add_argument("--dry-run", action="store_true", help="print the commands without running them")
    ap.add_argument("--keep-going", action="store_true", help="continue after a failing command")
    a = ap.parse_args()

    os.environ.setdefault("CDFFM_ROOT", os.path.join(REPO, "work"))
    lines = read_lines(a.commands)
    todo = list(range(len(lines))) if a.lines is None else parse_sel(a.lines, len(lines))
    if a.grep:
        todo = [i for i in todo if a.grep in lines[i]]
    if a.list:
        for i, l in enumerate(lines):
            print(f"{i:3d}  {l}")
        return
    py = shlex.quote(sys.executable)
    name = os.path.splitext(os.path.basename(a.commands))[0]
    os.makedirs(os.path.join(REPO, "logs"), exist_ok=True)
    print(f"CDFFM_ROOT={os.environ['CDFFM_ROOT']}; {len(todo)} of {len(lines)} commands from {a.commands}", flush=True)
    failed = []
    for k, i in enumerate(todo):
        cmd = os.path.expandvars(re.sub(r"(^|&&\s*)python(?=\s)", lambda m: m.group(1) + py, lines[i]))
        print(f"\n[{k + 1}/{len(todo)}] line {i}: {cmd}", flush=True)
        if a.dry_run:
            continue
        log = os.path.join(REPO, "logs", f"{name}_{i:03d}.log")
        t0 = time.time()
        with open(log, "w") as f:
            p = subprocess.Popen(cmd, shell=True, cwd=os.path.join(REPO, "scripts"), stdout=subprocess.PIPE,
                                 stderr=subprocess.STDOUT, text=True, bufsize=1)
            for out in p.stdout:
                sys.stdout.write(out)
                f.write(out)
            rc = p.wait()
        print(f"[{k + 1}/{len(todo)}] line {i}: exit {rc} after {(time.time() - t0) / 60:.1f} min (log: {log})", flush=True)
        if rc != 0:
            failed.append(i)
            if not a.keep_going:
                raise SystemExit(f"stopped at line {i}; rerun from here with --lines {i}-{len(lines) - 1}")
    if failed:
        raise SystemExit(f"failed lines: {failed}")


if __name__ == "__main__":
    main()
