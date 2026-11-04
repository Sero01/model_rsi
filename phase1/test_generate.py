"""Test generate's resume, locking and write-failure handling, and the driver's token count check.

    python -I phase1/test_generate.py <model.gguf> <generate binary> <work dir>

Runs the real engine for a few tokens at a time, so it needs about 4 GB of RAM and a minute or two.
"""
import json
import os
import resource
import shutil
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run_local  # noqa: E402

PROMPT = [248045, 846, 198, 16, 10, 16, 28, 30, 248046, 198, 248045, 74455, 198]  # token ids of a short chat turn
THREADS = "4"
LOCK_EXIT_CODE = 3
WRITE_LIMIT_BYTES = 20
STARTUP_TIMEOUT_SECONDS = 120


def count_lines(path):
    with open(path) as f:
        return sum(1 for line in f if line.strip())


def command(binary, gguf, work, max_new):
    return [binary, gguf, os.path.join(work, "prompt.txt"), os.path.join(work, "tokens.txt"), "0", THREADS, str(max_new)]


def fresh(work):
    shutil.rmtree(work, ignore_errors=True)
    os.makedirs(work)
    with open(os.path.join(work, "prompt.txt"), "w") as f:
        f.write("\n".join(map(str, PROMPT)))


def result_json(stdout):
    return json.loads(stdout.strip().splitlines()[-1])


def test_resume(binary, gguf, work):
    fresh(work)
    subprocess.run(command(binary, gguf, work, 5), check=True, capture_output=True, text=True)
    first = count_lines(os.path.join(work, "tokens.txt"))
    done = subprocess.run(command(binary, gguf, work, 8), check=True, capture_output=True, text=True)
    return first == 5 and count_lines(os.path.join(work, "tokens.txt")) == 8 and result_json(done.stdout)["tokens"] == 8


def test_second_engine_refused(binary, gguf, work):
    fresh(work)
    tokens = os.path.join(work, "tokens.txt")
    first = subprocess.Popen(command(binary, gguf, work, 30), stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
    deadline = time.time() + STARTUP_TIMEOUT_SECONDS
    while not (os.path.exists(tokens) and count_lines(tokens) > 0) and time.time() < deadline:
        time.sleep(0.5)
    second = subprocess.run(command(binary, gguf, work, 30), capture_output=True, text=True)
    stdout, _ = first.communicate()
    return second.returncode == LOCK_EXIT_CODE and first.returncode == 0 and count_lines(tokens) == result_json(stdout)["tokens"] == 30


def test_write_failure_is_fatal(binary, gguf, work):
    fresh(work)

    def limit_file_size():
        resource.setrlimit(resource.RLIMIT_FSIZE, (WRITE_LIMIT_BYTES, WRITE_LIMIT_BYTES))

    # Python ignores SIGXFSZ, and keeping that across exec turns the oversized write into an error the engine must see.
    done = subprocess.run(command(binary, gguf, work, 30), capture_output=True, text=True,
                          preexec_fn=limit_file_size, restore_signals=False)
    return done.returncode != 0


class FakeTokenizer:
    def decode(self, ids, skip_special_tokens):
        return "answer \\boxed{2}"


def test_grade_rejects_wrong_count(work):
    fresh(work)
    with open(os.path.join(work, "tokens.txt"), "w") as f:
        f.write("1\n2\n3\n")
    run_local.grade(FakeTokenizer(), work, "2", 3)
    try:
        run_local.grade(FakeTokenizer(), work, "2", 4)
    except SystemExit:
        return True
    return False


def main(gguf, binary, work):
    failures = 0
    cases = [("resume continues to the new limit", lambda: test_resume(binary, gguf, work)),
             ("second engine on one file is refused", lambda: test_second_engine_refused(binary, gguf, work)),
             ("failed write is fatal", lambda: test_write_failure_is_fatal(binary, gguf, work)),
             ("grading rejects a wrong token count", lambda: test_grade_rejects_wrong_count(work))]
    for name, case in cases:
        ok = case()
        failures += not ok
        print(f"{'PASS' if ok else 'FAIL'}: {name}", flush=True)
    print(f"{failures} failures")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main(os.path.abspath(sys.argv[1]), os.path.abspath(sys.argv[2]), os.path.abspath(sys.argv[3]))
