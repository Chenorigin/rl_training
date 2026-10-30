#!/usr/bin/env python3
"""One-command CTS model deployment and existing operator-controlled SDK start."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import time
import uuid

HERE = Path(__file__).resolve().parent
TRAINING = HERE.parents[1]
BUNDLE_FILES = {"policy.onnx", "policy.pt", "observation_contract.json", "deployment.json",
                "policy.onnx.json", "input.f32", "expected.f32"}


def run(argv, **kwargs):
    return subprocess.run([str(x) for x in argv], check=True, **kwargs)


def target(value):
    if not re.fullmatch(r"[A-Za-z0-9_][A-Za-z0-9_.@:%\[\]-]*", value):
        raise argparse.ArgumentTypeError("Use an SSH alias or user@host, not shell text/options")
    return value


class Board:
    def __init__(self, host, role, root, temporary, jump=None):
        self.host, self.role, self.root = host, role, root
        self.socket = temporary / (role.lower() + ".sock")
        self.helper = None
        self.jump = jump
        self.temporary = temporary

    def connect(self):
        argv = ["ssh", "-M", "-N", "-f", "-o", "ControlPersist=600", "-o", "ConnectTimeout=10",
                "-o", "ServerAliveInterval=5", "-o", "ServerAliveCountMax=2",
                "-o", "StrictHostKeyChecking=yes", "-S", self.socket]
        if self.jump:
            proxy = shlex.join(["ssh", "-S", str(self.jump.socket), "-W", "%h:%p", self.jump.host])
            argv += ["-o", "ProxyCommand=" + proxy]
        run(argv + [self.host])

    def ssh(self, command, interactive=False, **kwargs):
        argv = ["ssh", "-o", "ConnectTimeout=10", "-S", self.socket]
        if interactive:
            argv.append("-tt")
        return run(argv + [self.host, command], **kwargs)

    def upload(self, files, remote_dir):
        ssh = self.temporary / (self.role + "-ssh")
        scp = self.temporary / (self.role + "-scp")
        for path, argv in ((ssh, ["ssh", "-S", str(self.socket)]),
                           (scp, ["scp", "-o", "ControlPath=" + str(self.socket)])):
            path.write_text("#!/bin/sh\nexec " + shlex.join(argv) + ' "$@"\n')
            path.chmod(0o700)
        env = dict(os.environ, SAFEPUSH_SSH=str(ssh), SAFEPUSH_SCP=str(scp))
        run(["safepush"] + list(files) + [self.host + ":" + remote_dir + "/"], env=env)

    def install_helper(self):
        # Fresh path; never overwrite a running release or its scripts.
        directory = "/tmp/m20_operator_" + uuid.uuid4().hex
        self.ssh("umask 077; mkdir " + shlex.quote(directory))
        self.upload([str(HERE / "m20_robot_board.py")], directory)
        self.helper = directory + "/m20_robot_board.py"

    def action(self, name, *extra):
        if self.helper is None:
            self.install_helper()
        argv = ["sudo", "-E", "python3", self.helper, name, "--root", self.root, "--role", self.role] + list(extra)
        # Keep an actual interactive tty for sudo and the original START_STUDENT.
        self.ssh(shlex.join(argv), interactive=True)

    def describe(self):
        if self.helper is None:
            self.install_helper()
        command = shlex.join(["python3", self.helper, "describe", "--root", self.root, "--role", self.role])
        result = self.ssh(command, capture_output=True, text=True)
        prefix = "M20_IDENTITY="
        if prefix not in result.stdout:
            raise RuntimeError("Missing board identity response")
        return json.JSONDecoder().raw_decode(result.stdout.split(prefix, 1)[1])[0]

    def close(self):
        # Remove only our throwaway helper; board-local supervisors/logs persist.
        if self.helper:
            try:
                self.ssh("rm -f " + shlex.quote(self.helper) + "; rmdir " + shlex.quote(str(Path(self.helper).parent)))
            except subprocess.CalledProcessError:
                pass
        subprocess.run(["ssh", "-S", str(self.socket), "-O", "exit", self.host],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)


def bundle_files(bundle):
    bundle = Path(bundle).resolve(strict=True)
    if {p.name for p in bundle.iterdir()} != BUNDLE_FILES or any(p.is_symlink() or not p.is_file() for p in bundle.iterdir()):
        raise ValueError("Bundle must contain exactly the seven CTS export files")
    deployed = json.loads((bundle / "deployment.json").read_text())
    onnx = json.loads((bundle / "policy.onnx.json").read_text())
    digest = lambda name: hashlib.sha256((bundle / name).read_bytes()).hexdigest()
    if not re.fullmatch(r"[0-9a-f]{64}", deployed.get("checkpoint_sha256", "")):
        raise ValueError("Invalid checkpoint SHA256")
    if (digest("policy.pt") != deployed["policy_sha256"] or digest("policy.pt") != onnx["torchscript_sha256"] or
            digest("policy.onnx") != onnx["policy_sha256"] or digest("observation_contract.json") != onnx["contract_sha256"] or
            digest("observation_contract.json") != deployed["contract_sha256"] or
            any(digest(n) != onnx["fixture_sha256"][n] for n in ("input.f32", "expected.f32"))):
        raise ValueError("Export bundle checksum mismatch")
    return bundle, "cts_" + deployed["checkpoint_sha256"]


def export_checkpoint(args):
    checkpoint = Path(args.checkpoint).expanduser().resolve(strict=True)
    source = Path(args.deploy_source).expanduser().resolve(strict=True)
    memory = dict(line.split(":", 1) for line in Path("/proc/meminfo").read_text().splitlines())
    available = int(memory["MemAvailable"].split()[0]) * 1024
    # Existing exporter loads full checkpoint plus Torch/ORT; no GPU allocation.
    needed = checkpoint.stat().st_size * 8 + 1024 ** 3
    if available < needed or shutil.disk_usage(TRAINING).free < checkpoint.stat().st_size * 8:
        raise RuntimeError("Insufficient host memory/disk for CPU export")
    output = TRAINING / "logs/evaluation" / ("m20_operator_" + time.strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:8])
    output.mkdir(parents=True)
    # Canonical CPU loader; version admission is never patched or bypassed here.
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
    run([args.python, source / "scripts/export_cts_checkpoint.py", "--checkpoint", checkpoint,
         "--training", TRAINING, "--output", output / "bundle"], env=env)
    run([args.python, source / "scripts/offline_cts.py", "--model-dir", output / "bundle"], env=env)
    print("导出记录：" + str(output), flush=True)
    return output / "bundle"


def matching_models(a, g, allow_pending=False):
    for key in ("model_sha256", "contract_sha256", "checkpoint_sha256"):
        if a[key] != g[key]:
            raise RuntimeError("AOS/GOS models differ (" + key + "); deployment/recovery required before start")
    if not allow_pending and (a["deployment_pending"] or g["deployment_pending"]):
        raise RuntimeError("Incomplete deployment transaction; startup blocked")
    return a["model_sha256"]


def deploy(args, aos, gos, bundle):
    bundle, name = bundle_files(bundle)
    print("将切换模型。请确认狗已趴稳、SDK开启、急停可用，且无人操作手柄。", flush=True)
    if input("输入 DEPLOY_PRONE：").strip() != "DEPLOY_PRONE":
        raise RuntimeError("Deployment cancelled; models unchanged")
    transaction = uuid.uuid4().hex
    aos.action("begin-deploy", "--transaction", transaction)
    gos.action("begin-deploy", "--transaction", transaction)
    # Validate both independent immutable candidates before switching either link.
    for board in (aos, gos):
        incoming = "/tmp/m20_bundle_" + uuid.uuid4().hex
        board.ssh("umask 077; mkdir " + shlex.quote(incoming))
        try:
            board.upload([str(bundle / n) for n in sorted(BUNDLE_FILES)], incoming)
            board.action("stage", "--incoming", incoming, "--transaction", transaction)
        finally:
            # Only the exact seven names in our fresh directory, never rm -rf.
            board.ssh("rm -f " + " ".join(shlex.quote(incoming + "/" + n) for n in sorted(BUNDLE_FILES)) +
                      "; rmdir " + shlex.quote(incoming))
    gos.action("stop-perception")
    # Each board has local rollback on failed acceptance. If the second board
    # fails, leave all motors stopped, report the split, never auto-start.
    try:
        gos.action("activate", "--bundle", name, "--transaction", transaction)
        aos.action("activate", "--bundle", name, "--transaction", transaction)
        model_sha = matching_models(aos.describe(), gos.describe(), allow_pending=True)
        gos.action("finish-deploy", "--transaction", transaction, "--model-sha", model_sha)
        aos.action("finish-deploy", "--transaction", transaction, "--model-sha", model_sha)
    except BaseException:
        print("激活未全部完成。控制器保持停止；检查两板 status 和 runtime/activate_*/before.json 后恢复。", file=sys.stderr)
        raise
    print("双板模型已切换；控制器尚未启动。", flush=True)


def parser():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("action", choices=["start", "deploy", "status", "check", "stop", "logs"])
    ap.add_argument("--aos", required=True, type=target, help="Verified SSH alias or user@host")
    ap.add_argument("--gos", type=target, help="GOS target, reached through AOS")
    ap.add_argument("--release", default="/home/user/m20_heightmap/current", help="Existing installed release")
    ap.add_argument("--gos-release", help="Defaults to --release")
    model = ap.add_mutually_exclusive_group()
    model.add_argument("--checkpoint", type=Path)
    model.add_argument("--bundle", type=Path)
    ap.add_argument("--python", default=sys.executable, help="Existing export environment Python")
    ap.add_argument("--deploy-source", default=str(TRAINING.parent / "m20_student_deploy"))
    return ap


def main(argv=None):
    ap = parser()
    args = ap.parse_args(argv)
    if args.action in ("start", "deploy", "check") and not args.gos:
        ap.error("--gos is required for this operation")
    if args.action == "deploy" and not (args.bundle or args.checkpoint):
        ap.error("deploy requires --checkpoint or --bundle; no automatic latest selection")
    if args.action in ("status", "check", "stop", "logs") and (args.bundle or args.checkpoint):
        ap.error("This action does not accept a model change")
    if not sys.stdin.isatty() and args.action in ("start", "deploy"):
        ap.error("Run from a real interactive terminal")
    if not shutil.which("safepush"):
        ap.error("safepush is required for checked transfers")
    bundle = export_checkpoint(args) if args.checkpoint else args.bundle
    with tempfile.TemporaryDirectory(prefix="m20_op_") as directory:
        temporary = Path(directory)
        aos = Board(args.aos, "AOS", args.release, temporary)
        gos = Board(args.gos, "GOS", args.gos_release or args.release, temporary, aos) if args.gos and args.action not in ("stop", "logs") else None
        boards = [aos] + ([gos] if gos else [])
        try:
            for board in boards:
                board.connect()
                if args.action in ("start", "deploy"):
                    board.action("status")
            if bundle:
                deploy(args, aos, gos, bundle)
            if args.action == "start":
                model_sha = matching_models(aos.describe(), gos.describe())
                gos.action("start")
                aos.action("start", "--model-sha", model_sha)
            elif args.action in ("status", "check", "stop", "logs"):
                if args.action == "check":
                    matching_models(aos.describe(), gos.describe())
                # Stop is intentionally AOS-only: it needs neither live GOS nor
                # a working model, and does not tear down diagnostic perception.
                for board in ([aos] if args.action in ("stop", "logs") else boards):
                    board.action(args.action)
        finally:
            for board in reversed(boards):
                board.close()
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (RuntimeError, ValueError, OSError, subprocess.SubprocessError) as error:
        print("M20操作未完成：" + str(error), file=sys.stderr)
        sys.exit(1)
    except (KeyboardInterrupt, EOFError):
        print("操作已取消。若已断开 tmux，使用 status/stop 核实板端状态。", file=sys.stderr)
        sys.exit(130)
