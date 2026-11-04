"""Prove what phase1/sandbox.py blocks. Run: python3 -I phase1/test_sandbox.py"""
import os
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
SANDBOX = os.path.join(HERE, "sandbox.py")
SECRET = os.path.join(os.path.dirname(HERE), "WSP_RESEARCH_PROMPT.md")
PY = sys.executable


def inside(rw, ro, code):
    """Exit code of code run in the sandbox. 0 means the action succeeded."""
    cmd = [PY, "-I", SANDBOX, "--rw", rw, "--ro", ro, "--", PY, "-I", "-c", code]
    return subprocess.run(cmd, capture_output=True, timeout=60).returncode


def main():
    failures = 0
    with tempfile.TemporaryDirectory(dir=os.path.dirname(HERE) + "/.tmp") as base:
        rw, ro, other = (os.path.join(base, d) for d in ("rw", "ro", "other"))
        for d in (rw, ro, other):
            os.mkdir(d)
        ro_file = os.path.join(ro, "f")
        open(ro_file, "w").write("x")
        other_file = os.path.join(other, "f")
        open(other_file, "w").write("x")

        cases = [
            ("runs a command", "pass", True),
            ("writes in an rw path", f"open({os.path.join(rw, 'new')!r}, 'w').write('x')", True),
            ("reads an ro path", f"open({ro_file!r}).read()", True),
            ("writes /dev/null", "open('/dev/null', 'w').write('x')", True),
            ("blocks writing an ro path", f"open({ro_file!r}, 'w').write('y')", False),
            ("blocks reading an unlisted path", f"open({other_file!r}).read()", False),
            ("blocks reading the research plan", f"open({SECRET!r}).read()", False),
            ("blocks reading the home directory", f"import os; os.listdir({os.path.expanduser('~')!r})", False),
            ("blocks an IPv4 socket", "import socket; socket.socket(socket.AF_INET, socket.SOCK_STREAM)", False),
            ("blocks a UDP socket", "import socket; socket.socket(socket.AF_INET, socket.SOCK_DGRAM)", False),
            ("blocks an IPv6 socket", "import socket; socket.socket(socket.AF_INET6, socket.SOCK_STREAM)", False),
            ("blocks a Unix socket", "import socket; socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)", False),
            ("clears secrets from the environment", "import os,sys; sys.exit(0 if 'SANDBOX_TEST_SECRET' in os.environ else 1)", False),
        ]
        os.environ["SANDBOX_TEST_SECRET"] = "x"
        for name, code, should_work in cases:
            worked = inside(rw, ro, code) == 0
            ok = worked == should_work
            failures += not ok
            print(f"{'PASS' if ok else 'FAIL'}: {name}")
    print(f"{failures} failures")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
