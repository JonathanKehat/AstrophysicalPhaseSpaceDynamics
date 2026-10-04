"""CLI:  python -m experiments {list|run <id>|index}"""
import sys
import os

import experiments as X


def main(argv):
    if not argv or argv[0] in ("-h", "--help", "help"):
        print(__doc__.strip()); return 0
    cmd = argv[0]
    if cmd == "list":
        X.info(); return 0
    if cmd == "index":
        root = os.path.join(os.path.dirname(__file__), os.pardir)
        out = os.path.abspath(os.path.join(root, "INDEX.md"))
        with open(out, "w") as f:
            f.write(X.index_md())
        print("wrote", out); return 0
    if cmd == "run":
        if len(argv) < 2:
            print("usage: python -m experiments run <id> [<id> ...]"); return 2
        for tid in argv[1:]:
            print(f"[run] {tid} ...", flush=True)
            res = X.run(tid, force=True)
            keys = ", ".join(k for k in res if k not in ("id",))
            print(f"[run] {tid} done -> cached ({keys[:120]})")
        return 0
    print(f"unknown command {cmd!r}"); return 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
