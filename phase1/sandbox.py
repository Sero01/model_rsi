"""Run a command with no sockets and only the listed paths visible, without root or namespaces.

    python3 -I phase1/sandbox.py [--ro PATH]... [--rw PATH]... -- command [args...]

Used for anything that executes llama.cpp code, which is too large to have been read in full. bubblewrap cannot run
on this laptop: AppArmor restricts unprivileged user namespaces and has no profile for it. Landlock limits the
filesystem and seccomp refuses to create sockets, and both only ever remove rights. The environment is replaced, and
HOME and TMPDIR point at the first --rw path.
"""
import argparse
import ctypes
import os
import struct
import sys

SYS_SOCKET, SYS_IO_URING_SETUP = 41, 425
SYS_LANDLOCK_CREATE_RULESET, SYS_LANDLOCK_ADD_RULE, SYS_LANDLOCK_RESTRICT_SELF = 444, 445, 446
PR_SET_SECCOMP, PR_SET_NO_NEW_PRIVS, SECCOMP_MODE_FILTER = 22, 38, 2
LANDLOCK_CREATE_RULESET_VERSION, LANDLOCK_RULE_PATH_BENEATH = 1, 1
MIN_LANDLOCK_ABI = 6

FS_EXECUTE, FS_WRITE_FILE, FS_READ_FILE, FS_READ_DIR = 1 << 0, 1 << 1, 1 << 2, 1 << 3
FS_TRUNCATE, FS_IOCTL_DEV = 1 << 14, 1 << 15
FS_ALL = (1 << 16) - 1
SCOPE_ABSTRACT_UNIX, SCOPE_SIGNAL = 1, 2

READ = FS_EXECUTE | FS_READ_FILE | FS_READ_DIR
FILE_RIGHTS = FS_EXECUTE | FS_WRITE_FILE | FS_READ_FILE | FS_TRUNCATE | FS_IOCTL_DEV
SYSTEM_READ = ("/usr", "/lib", "/lib64", "/bin", "/sbin", "/etc", "/proc", "/sys")
DEVICES = ("/dev/null", "/dev/zero", "/dev/urandom", "/dev/random")
KEEP_ENV = ("PATH", "LANG", "LC_ALL", "OMP_NUM_THREADS")

BPF_LD_W_ABS, BPF_JEQ_K, BPF_JGE_K, BPF_RET_K = 0x20, 0x15, 0x35, 0x06
SECCOMP_DATA_NR, SECCOMP_DATA_ARCH = 0, 4
AUDIT_ARCH_X86_64 = 0xC000003E
X32_SYSCALL_BIT = 0x40000000
RET_ALLOW, RET_KILL, RET_EPERM = 0x7FFF0000, 0x80000000, 0x00050001

libc = ctypes.CDLL(None, use_errno=True)


def checked(result, what):
    if result < 0:
        raise OSError(ctypes.get_errno(), what)
    return result


def add_rule(ruleset, path, access):
    fd = os.open(path, os.O_PATH | os.O_CLOEXEC)
    try:
        attr = ctypes.create_string_buffer(struct.pack("<Qi", access, fd))
        checked(libc.syscall(SYS_LANDLOCK_ADD_RULE, ruleset, LANDLOCK_RULE_PATH_BENEATH, attr, 0), f"add rule {path}")
    finally:
        os.close(fd)


def restrict_filesystem(ro, rw):
    abi = checked(libc.syscall(SYS_LANDLOCK_CREATE_RULESET, None, 0, LANDLOCK_CREATE_RULESET_VERSION), "landlock abi")
    if abi < MIN_LANDLOCK_ABI:
        raise SystemExit(f"Landlock ABI {abi} is below the required {MIN_LANDLOCK_ABI}")
    attr = ctypes.create_string_buffer(struct.pack("<QQQ", FS_ALL, 0, SCOPE_ABSTRACT_UNIX | SCOPE_SIGNAL))
    ruleset = checked(libc.syscall(SYS_LANDLOCK_CREATE_RULESET, attr, len(attr.raw), 0), "create ruleset")
    for path in SYSTEM_READ:
        if os.path.exists(path):
            add_rule(ruleset, path, READ)
    for path in DEVICES:
        add_rule(ruleset, path, FS_READ_FILE | FS_WRITE_FILE)
    for path in ro:
        add_rule(ruleset, path, READ if os.path.isdir(path) else READ & FILE_RIGHTS)
    for path in rw:
        add_rule(ruleset, path, FS_ALL if os.path.isdir(path) else FILE_RIGHTS)
    checked(libc.syscall(SYS_LANDLOCK_RESTRICT_SELF, ruleset, 0), "restrict self")
    os.close(ruleset)


def refuse_sockets():
    """Seccomp filter: socket and io_uring_setup, which can open sockets without socket(), fail with EPERM."""
    def stmt(code, k):
        return struct.pack("<HBBI", code, 0, 0, k)

    def jump(code, k, jt, jf):
        return struct.pack("<HBBI", code, jt, jf, k)

    program = [
        stmt(BPF_LD_W_ABS, SECCOMP_DATA_ARCH),
        jump(BPF_JEQ_K, AUDIT_ARCH_X86_64, 1, 0),
        stmt(BPF_RET_K, RET_KILL),
        stmt(BPF_LD_W_ABS, SECCOMP_DATA_NR),
        jump(BPF_JGE_K, X32_SYSCALL_BIT, 2, 0),
        jump(BPF_JEQ_K, SYS_SOCKET, 1, 0),
        jump(BPF_JEQ_K, SYS_IO_URING_SETUP, 0, 1),
        stmt(BPF_RET_K, RET_EPERM),
        stmt(BPF_RET_K, RET_ALLOW),
    ]
    filt = ctypes.create_string_buffer(b"".join(program))
    prog = struct.pack("<HxxxxxxQ", len(program), ctypes.addressof(filt))
    checked(libc.prctl(PR_SET_SECCOMP, SECCOMP_MODE_FILTER, ctypes.create_string_buffer(prog), 0, 0), "seccomp")


def main():
    argv = sys.argv[1:]
    if "--" not in argv:
        raise SystemExit("usage: sandbox.py [--ro PATH]... [--rw PATH]... -- command [args...]")
    split = argv.index("--")
    parser = argparse.ArgumentParser()
    parser.add_argument("--ro", action="append", default=[])
    parser.add_argument("--rw", action="append", default=[])
    args = parser.parse_args(argv[:split])
    command = argv[split + 1:]
    if not command or not args.rw:
        raise SystemExit("a command and at least one --rw path are required")
    ro, rw = [os.path.realpath(p) for p in args.ro], [os.path.realpath(p) for p in args.rw]

    env = {k: os.environ[k] for k in KEEP_ENV if k in os.environ}
    env.update(HOME=rw[0], TMPDIR=rw[0])

    checked(libc.prctl(PR_SET_NO_NEW_PRIVS, 1, 0, 0, 0), "no_new_privs")
    restrict_filesystem(ro, rw)
    refuse_sockets()
    os.chdir(rw[0])
    os.execvpe(command[0], command, env)


if __name__ == "__main__":
    main()
