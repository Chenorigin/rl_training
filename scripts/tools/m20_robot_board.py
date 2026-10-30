#!/usr/bin/env python3
"""Board half of m20_robot.py. Wraps the installed operator scripts, never the FSM."""
import argparse
import fcntl
import hashlib
import json
import os
import re
from pathlib import Path
import shlex
import shutil
import signal
import subprocess
import sys
import time
import tempfile
import uuid
from contextlib import contextmanager

# Resource ledger for this wrapper, based on the existing v4 SDK/trace runtime.
GIB = 1024 ** 3
TRACE_MAX_BYTES = 2 * GIB
TRACE_RESERVE_BYTES = GIB
MIN_AVAILABLE_BYTES = GIB
BUNDLE_FILES = {"policy.onnx", "policy.pt", "observation_contract.json",
                "deployment.json", "policy.onnx.json", "input.f32", "expected.f32"}
READY_MARKERS = ("[CTS_WARMUP] runs=8", "Robot state stable for 2 seconds;",
                 "Fresh joint state received after reset command.",
                 "GAMEPAD TELEOP", "timer start ...")
STATE_HOME = Path("/run/m20_one_command")


@contextmanager
def state_lock(name):
    # Public non-secret identity/transaction markers, root-writable only.
    STATE_HOME.mkdir(mode=0o755, exist_ok=True)
    with (STATE_HOME / name).open("a+") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | (fcntl.LOCK_NB if name == "sdk.lock" else 0))
        yield handle


def epoch():
    path = STATE_HOME / "epoch"
    return path.read_text() if path.exists() else "initial"


def process_identity(pid):
    p = Path("/proc", str(pid))
    try:
        return {"pid": pid, "start_ticks": (p / "stat").read_text().rsplit(")", 1)[1].split()[19],
                "exe": str((p / "exe").resolve()), "cwd": str((p / "cwd").resolve())}
    except FileNotFoundError:
        return None


def no_transaction(root=None):
    if (STATE_HOME / "deploy.json").exists() or (root is not None and (root / "runtime/one_command_deploy.json").exists()):
        raise RuntimeError("Incomplete deployment transaction: start blocked; restore/complete both boards first")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".new")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    os.replace(str(tmp), str(path))


def run(argv, **kwargs):
    return subprocess.run([str(x) for x in argv], check=True, **kwargs)


def processes(name):
    result = []
    for p in Path("/proc").iterdir():
        if not p.name.isdigit():
            continue
        try:
            if (p / "comm").read_text().strip() == name:
                result.append({"pid": int(p.name), "exe": str((p / "exe").resolve()),
                               "cwd": str((p / "cwd").resolve()),
                               "start_ticks": (p / "stat").read_text().rsplit(")", 1)[1].split()[19]})
        except FileNotFoundError:
            pass
    return result


def signal_checked(process, sig):
    """Do not use process-name substring matching or stale PIDs."""
    current = [p for p in processes("student_control") if p["pid"] == process["pid"]]
    if not current:
        return
    if current[0] != process:
        raise RuntimeError("Controller PID identity changed; refusing signal")
    os.kill(process["pid"], sig)


def stop(root):
    # Cancel even a detached START_STUDENT prompt, and invalidate launches that
    # have registered but have not yet spawned their supervisor.
    with state_lock("metadata.lock"):
        (STATE_HOME / "epoch").write_text(uuid.uuid4().hex)
        owner_path = STATE_HOME / "owner.json"
        owner = json.loads(owner_path.read_text()) if owner_path.exists() else None
        if owner and owner["root"] == str(root) and process_identity(owner["supervisor"]["pid"]) == owner["supervisor"]:
            os.kill(owner["supervisor"]["pid"], signal.SIGINT)
    expected = str(root / "install/m20_student_sdk/lib/m20_student_sdk/student_control")
    owned = [p for p in processes("student_control") if p["exe"] == expected and p["cwd"] == str(root)]
    # Stop requires no model, preflight, resource or commissioning admission.
    for p in owned:
        signal_checked(p, signal.SIGINT)
    deadline = time.monotonic() + 15
    while any(p in processes("student_control") for p in owned):
        if time.monotonic() > deadline:
            raise RuntimeError("Graceful stop timed out; no SIGKILL or restart issued")
        time.sleep(.2)
    while owner and owner["root"] == str(root) and process_identity(owner["supervisor"]["pid"]) == owner["supervisor"]:
        if time.monotonic() > deadline:
            raise RuntimeError("Pending SDK supervisor did not exit; no restart issued")
        time.sleep(.2)
    return {"stopped_pids": [p["pid"] for p in owned]}


def identity(root, role):
    actual = Path("/opt/robot/os_type").read_text().strip()
    if actual != "CA9B_" + role:
        raise RuntimeError("Wrong board role: " + actual)
    if not (root / "scripts/operator_start_aos.sh").is_file():
        raise RuntimeError("Not an installed M20 release")
    return {"role": actual, "serial": Path("/proc/device-tree/serial-number").read_bytes().rstrip(b"\0").decode(),
            "boot_id": Path("/proc/sys/kernel/random/boot_id").read_text().strip(), "release": str(root)}


def status(root, role):
    data = identity(root, role)
    data.update(model=str((root / "models/v1").resolve()),
                model_sha256=sha(root / "models/v1/policy.onnx"),
                contract=json.loads((root / "models/v1/observation_contract.json").read_text()).get("cts_version"),
                controllers=processes("student_control"), mappers=processes("gos_mapper"))
    pending = root / "runtime/one_command_deploy.json"
    data["deployment_pending"] = json.loads(pending.read_text()) if pending.exists() else None
    latest = root / "runtime/one_command_latest.json"
    if latest.exists():
        directory = Path(json.loads(latest.read_text())["run_dir"])
        state = directory / "state.json"
        data["latest_run"] = json.loads(state.read_text()) if state.exists() else {"phase": "pending", "run_dir": str(directory)}
    return data


def describe(root, role):
    data = identity(root, role)
    data.update(model_sha256=sha(root / "models/v1/policy.onnx"),
                contract_sha256=sha(root / "models/v1/observation_contract.json"),
                checkpoint_sha256=json.loads((root / "models/v1/deployment.json").read_text())["checkpoint_sha256"],
                deployment_pending=(root / "runtime/one_command_deploy.json").exists())
    print("M20_IDENTITY=" + json.dumps(data))


def resources(trace_base):
    memory = dict(line.split(":", 1) for line in Path("/proc/meminfo").read_text().splitlines())
    available = int(memory["MemAvailable"].split()[0]) * 1024
    free = shutil.disk_usage(trace_base).free
    if available < MIN_AVAILABLE_BYTES or free < TRACE_MAX_BYTES + TRACE_RESERVE_BYTES:
        raise RuntimeError("Insufficient available memory or trace disk reserve")
    return {"available_memory_bytes": available, "trace_free_bytes": free}


def shell_env(root, command):
    return ["bash", "-c", "set -euo pipefail; cd " + shlex.quote(str(root)) +
            "; source scripts/board_env.sh; " + command]


def check(root, role):
    print(json.dumps(identity(root, role)), flush=True)
    if role == "AOS" and processes("student_control"):
        raise RuntimeError("A controller already exists; use status/stop first")
    resources(Path("/var/opt/robot/data"))
    if role == "AOS":
        command = ("nice -n 15 taskset -c 4 python3 scripts/offline_cts.py --model-dir models/v1 "
                   "--probe install/m20_student_sdk/lib/m20_student_sdk/student_onnx_probe; "
                   "export M20_VERIFIED_VENDOR_PID=$(python3 scripts/check_vendor_owner.py --pid-only); "
                   "python3 scripts/live_preflight.py --profile v1 --control --supported-init --seconds 3")
    else:
        command = "python3 scripts/validate_cts_bundle.py --metadata-only"
    run(shell_env(root, command))


def start_gos(root):
    mappers = processes("gos_mapper")
    if mappers:
        if len(mappers) != 1 or mappers[0]["cwd"] != str(root) or not mappers[0]["exe"].startswith("/usr/bin/python"):
            raise RuntimeError("Existing GOS mapper ownership is ambiguous")
        print(json.dumps({"perception": "already_running", "mapper_pid": mappers[0]["pid"]}))
        return
    check(root, "GOS")
    directory = root / "runtime" / ("operator_gos_" + uuid.uuid4().hex[:12])
    directory.mkdir(mode=0o700)
    with (directory / "supervisor.log").open("wb") as log:
        p = subprocess.Popen(["bash", str(root / "scripts/operator_start_gos.sh"), "v1"], cwd=str(root),
                             stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    save(directory / "receipt.json", {"supervisor_pid": p.pid, "release": str(root)})
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        if p.poll() is not None:
            raise RuntimeError("GOS supervisor exited; read " + str(directory / "supervisor.log"))
        mappers = processes("gos_mapper")
        if len(mappers) == 1 and mappers[0]["cwd"] == str(root):
            print(json.dumps({"perception": "started", "mapper_pid": mappers[0]["pid"], "log": str(directory)}))
            return
        time.sleep(.5)
    # Only our newly created supervisor; its original trap owns child cleanup.
    p.terminate()
    p.wait(timeout=15)
    raise RuntimeError("GOS startup timed out; our supervisor stopped")


def start_aos(root, trace_base, expected_sha=None):
    identity(root, "AOS")
    resources(trace_base)
    no_transaction(root)
    registered_epoch = epoch()
    if expected_sha and sha(root / "models/v1/policy.onnx") != expected_sha:
        raise RuntimeError("Model changed after two-board verification")
    if processes("student_control"):
        raise RuntimeError("Controller exists; refusing duplicate start")
    directory = trace_base / ("operator_" + time.strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:8])
    directory.mkdir(mode=0o700)
    # Own the lifetime lock before creating a tmux session. It is handed to the
    # supervisor; no interval permits another instance to race the preflight.
    with state_lock("sdk.lock") as lock:
        code_directory = root / "runtime/one_command_code"
        code_directory.mkdir(mode=0o755, exist_ok=True)
        code_sha = sha(__file__)
        copied = code_directory / (code_sha + ".py")
        if copied.exists():
            if sha(copied) != code_sha:
                raise RuntimeError("Installed operator helper changed; refusing overwrite")
        else:
            with copied.open("xb") as handle:
                handle.write(Path(__file__).read_bytes())
        session = "m20_" + uuid.uuid4().hex[:12]
        save(directory / "session.json", {"session": session, "release": str(root), "epoch": registered_epoch,
                                          "model_sha256": sha(root / "models/v1/policy.onnx"),
                                          "helper_sha256": sha(copied),
                                          "helper_path": str(copied),
                                          "operator_sha256": sha(root / "scripts/operator_start_aos.sh"),
                                          "robot_sha256": sha(root / "deploy/robot.json"),
                                          "config_sha256": sha(root / "runtime/heightmap_v1.local.json")})
        save(root / "runtime/one_command_latest.json", {"run_dir": str(directory), "session": session})
        launch = ["python3", str(copied), "supervise", "--root", str(root), "--role", "AOS", "--run-dir", str(directory)]
        run(["tmux", "new-session", "-d", "-s", session, "-c", root, "bash --noprofile --norc"])
        try:
            run(["tmux", "pipe-pane", "-o", "-t", session + ":0", "cat >> " + shlex.quote(str(directory / "sdk.log"))])
            # Unlock immediately before starting supervisor; its own lock blocks
            # all rival supervisors even if multiple interactive launches race.
            fcntl.flock(lock, fcntl.LOCK_UN)
            run(["tmux", "send-keys", "-l", "-t", session + ":0", "exec " + shlex.join(launch)])
            run(["tmux", "send-keys", "-t", session + ":0", "Enter"])
        except BaseException:
            run(["tmux", "kill-session", "-t", session])
            raise
    print("\n原始 START_STUDENT 确认将出现在 tmux 中。Ctrl-b d 断开显示，策略不会随 SSH 退出。", flush=True)
    print("日志：" + str(directory), flush=True)
    run(["tmux", "attach-session", "-t", session])
    # Detaching is not a readiness assertion: explicitly show persisted phase.
    state = directory / "state.json"
    print(state.read_text() if state.exists() else "SDK状态尚未生成；不能认定启动成功")
    if state.exists() and json.loads(state.read_text())["phase"] in ("fault_stopped", "failed_or_cancelled"):
        raise RuntimeError("SDK stopped or failed; read this run's log")


def supervise(root, directory):
    with state_lock("sdk.lock"):
        if processes("student_control"):
            raise RuntimeError("Controller appeared during start; refusing duplicate")
        env = dict(os.environ, M20_CONTROL_TRACE_DIR=str(directory),
                   M20_TRACE_MAX_BYTES=str(TRACE_MAX_BYTES), M20_TRACE_MIN_FREE_BYTES=str(TRACE_RESERVE_BYTES))
        p = None
        state = {"phase": "preflight_or_operator_confirmation",
                 "model_sha256": sha(root / "models/v1/policy.onnx"), "run_dir": str(directory)}
        seen, offset, native_started = "", 0, None
        recovery_lines = ""
        cancelled = [False]
        def request_stop(_signum, _frame):
            cancelled[0] = True
        previous_handlers = {s: signal.signal(s, request_stop) for s in
                             (signal.SIGINT, signal.SIGTERM, signal.SIGHUP)}
        try:
            # All post-spawn operations, including the first receipt write,
            # share this cleanup boundary. Cancellation cannot orphan a prompt.
            with state_lock("metadata.lock"):
                no_transaction(root)
                session = json.loads((directory / "session.json").read_text())
                if session["epoch"] != epoch() or session["model_sha256"] != sha(root / "models/v1/policy.onnx"):
                    raise RuntimeError("Launch cancelled or model changed before operator confirmation")
                save(STATE_HOME / "owner.json", {"root": str(root), "supervisor": process_identity(os.getpid())})
                p = subprocess.Popen(["bash", str(root / "scripts/operator_start_aos.sh"), "v1", "supported-init"],
                                     cwd=str(root), env=env)
            state["pid"] = p.pid
            save(directory / "state.json", state)
            while p.poll() is None:
                if cancelled[0]:
                    state["phase"] = "cancel_stopping"
                    p.send_signal(signal.SIGINT)
                    break
                log = directory / "sdk.log"
                if log.exists():
                    with log.open(errors="replace") as f:
                        f.seek(offset)
                        fragment = f.read()
                        seen = (seen + fragment)[-131072:]
                        offset = f.tell()
                    recovery_lines += fragment
                    lines = recovery_lines.split("\n")
                    recovery_lines = lines.pop()[-8192:]
                    for line in lines:
                        if "[STUDENT_RECOVERABLE_FAULT]" in line:
                            state["phase"] = "fault_latched"
                            state["recovery"] = "R1 acknowledge then separate R2; healthy idle required"
                            save(directory / "state.json", state)
                        elif "[STUDENT_RECOVERY_READY]" in line and state["phase"] == "fault_latched":
                            state["phase"] = "handset_ready"
                            state["recovery"] = "reset to idle; fresh L1 then L2 required"
                            save(directory / "state.json", state)
                if "[STUDENT_FAULT]" in seen:
                    state["phase"] = "fault_stopping"
                    save(directory / "state.json", state)
                    p.send_signal(signal.SIGINT)
                    break
                if "Loading M20 policy:" in seen and native_started is None:
                    native_started = time.monotonic()
                if all(marker in seen for marker in READY_MARKERS) and state["phase"] == "preflight_or_operator_confirmation":
                    state["phase"] = "handset_ready"
                    state["physical_stability"] = "NOT_VALIDATED"
                    save(directory / "state.json", state)
                    print("\n[M20_READY] SDK就绪；物理模式由现场手柄控制。", flush=True)
                if native_started and state["phase"] == "preflight_or_operator_confirmation" and time.monotonic() - native_started > 60:
                    state["phase"] = "initialization_timeout_stopping"
                    save(directory / "state.json", state)
                    p.send_signal(signal.SIGINT)
                    break
                time.sleep(.2)
            code = p.wait(timeout=15)
            prior = state["phase"]
            state.update(phase="fault_stopped" if prior == "fault_stopping" else
                         "stopped" if prior == "handset_ready" and code == 0 else "failed_or_cancelled",
                         exit_code=code)
            save(directory / "state.json", state)
            print("\n[M20_EXIT] " + json.dumps(state, ensure_ascii=False), flush=True)
            return 0 if state["phase"] == "stopped" else 1
        finally:
            try:
                if p is not None and p.poll() is None:
                    p.send_signal(signal.SIGINT)
                    p.wait(timeout=15)
                if state.get("phase") not in ("stopped", "fault_stopped", "failed_or_cancelled"):
                    state.update(phase="failed_or_cancelled", exit_code=p.returncode if p else None)
                    save(directory / "state.json", state)
            finally:
                for signum, handler in previous_handlers.items():
                    signal.signal(signum, handler)


def stage(root, role, incoming):
    if processes("student_control"):
        raise RuntimeError("Deploy requires stopped controller; first lie down then stop")
    source = Path(incoming).resolve(strict=True)
    if {p.name for p in source.iterdir()} != BUNDLE_FILES or any(p.is_symlink() or not p.is_file() for p in source.iterdir()):
        raise RuntimeError("Expected exactly the seven regular CTS bundle files")
    record = json.loads((source / "deployment.json").read_text())
    if not re.fullmatch(r"[0-9a-f]{64}", record.get("checkpoint_sha256", "")):
        raise RuntimeError("Invalid checkpoint SHA256")
    name = "cts_" + record["checkpoint_sha256"]
    destination = root / "models" / name
    if shutil.disk_usage(root / "models").free < sum((source / n).stat().st_size for n in BUNDLE_FILES) * 2:
        raise RuntimeError("Insufficient release disk space for immutable candidate")
    scratch = None
    try:
        if destination.exists():
            if {p.name for p in destination.iterdir()} != BUNDLE_FILES or any(sha(destination / n) != sha(source / n) for n in BUNDLE_FILES):
                raise RuntimeError("Existing immutable bundle differs; not overwriting")
            checking = destination
        else:
            scratch = Path(tempfile.mkdtemp(prefix=".m20_stage_", dir=str(root / "models")))
            for file_name in BUNDLE_FILES:
                shutil.copyfile(str(source / file_name), str(scratch / file_name))
            checking = scratch
        command = "python3 scripts/validate_cts_bundle.py --metadata-only --model-dir " + shlex.quote(str(checking))
        if role == "AOS":
            command += ("; nice -n 15 taskset -c 4 python3 scripts/offline_cts.py --model-dir " + shlex.quote(str(checking)) +
                        " --probe install/m20_student_sdk/lib/m20_student_sdk/student_onnx_probe")
        run(shell_env(root, command))
        if scratch:
            scratch.chmod(0o755)
            os.rename(str(scratch), str(destination))
            scratch = None
    finally:
        if scratch:
            shutil.rmtree(str(scratch))  # Only this invocation's private scratch.
    print("M20_STAGED=" + json.dumps({"bundle": name, "sha256": {n: sha(destination / n) for n in BUNDLE_FILES}}))


def stop_perception(root):
    mappers = processes("gos_mapper")
    if any(p["cwd"] != str(root) for p in mappers):
        raise RuntimeError("Mapper belongs to another release; not stopping it")
    owners = []
    for p in processes("bash"):
        if p["cwd"] != str(root):
            continue
        try:
            args = Path("/proc", str(p["pid"]), "cmdline").read_bytes().split(b"\0")
            if len(args) > 2 and args[2] == b"v1" and Path(args[1].decode()).name == "operator_start_gos.sh":
                owners.append(p)
        except FileNotFoundError:
            pass
    if not owners and not mappers:
        return
    if len(owners) != 1:
        raise RuntimeError("Cannot identify original perception supervisor")
    os.kill(owners[0]["pid"], signal.SIGTERM)
    end = time.monotonic() + 20
    while processes("gos_mapper") or Path("/proc", str(owners[0]["pid"])).exists():
        if time.monotonic() > end:
            raise RuntimeError("Perception stop timed out; activation blocked")
        time.sleep(.2)


def activate(root, role, bundle):
    if processes("student_control") or processes("gos_mapper"):
        raise RuntimeError("Activation requires stopped controller and mapper")
    candidate = root / "models" / bundle
    if candidate.parent != root / "models" or not candidate.is_dir() or candidate.is_symlink():
        raise RuntimeError("Invalid immutable bundle name")
    backup = root / "runtime" / ("activate_" + uuid.uuid4().hex)
    backup.mkdir(mode=0o700)
    link = root / "models/v1"
    if not link.is_symlink():
        raise RuntimeError("Expected models/v1 symlink; refusing directory replacement")
    old = os.readlink(str(link))
    save(backup / "before.json", {"previous_model": old, "candidate": bundle})
    receipt = root / "runtime/offline_acceptance.json"
    if receipt.exists():
        shutil.copyfile(receipt, backup / "offline_acceptance.json")
    temporary = link.with_name("v1." + uuid.uuid4().hex)
    os.symlink(bundle, str(temporary))
    os.replace(str(temporary), str(link))
    try:
        if role == "AOS":
            run(shell_env(root, "nice -n 15 taskset -c 4 python3 scripts/accept_offline_release.py"))
        else:
            run(shell_env(root, "python3 scripts/validate_cts_bundle.py --metadata-only"))
    except BaseException:
        os.symlink(old, str(temporary))
        os.replace(str(temporary), str(link))
        if (backup / "offline_acceptance.json").exists():
            shutil.copyfile(backup / "offline_acceptance.json", receipt)
        raise
    print("M20_ACTIVATED=" + json.dumps({"model": str(link.resolve()), "sha256": sha(link / "policy.onnx"),
                                          "backup": str(backup), "role": role}))


def begin_deploy(root, role, transaction):
    with state_lock("metadata.lock"):
        marker = STATE_HOME / "deploy.json"
        persistent = root / "runtime/one_command_deploy.json"
        if marker.exists() or persistent.exists():
            raise RuntimeError("Previous deployment is incomplete; inspect both boards before recovery")
        record = {"transaction": transaction, "root": str(root)}
        save(persistent, record)
        save(marker, record)
    if role == "AOS":
        stop(root)


def require_transaction(root, transaction):
    marker = json.loads((root / "runtime/one_command_deploy.json").read_text())
    if marker != {"transaction": transaction, "root": str(root)}:
        raise RuntimeError("Deployment transaction ownership mismatch")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("action", choices=["status", "describe", "check", "start", "stop", "logs", "begin-deploy", "finish-deploy", "stop-perception", "stage", "activate", "supervise"])
    ap.add_argument("--root", required=True, type=Path)
    ap.add_argument("--role", choices=["AOS", "GOS"], required=True)
    ap.add_argument("--trace-base", type=Path, default=Path("/var/opt/robot/data/m20_control_traces"))
    ap.add_argument("--run-dir", type=Path)
    ap.add_argument("--incoming")
    ap.add_argument("--bundle")
    ap.add_argument("--model-sha")
    ap.add_argument("--transaction")
    args = ap.parse_args()
    root = args.root.expanduser().resolve(strict=True)
    if args.action != "stop":
        identity(root, args.role)
    if args.action == "status":
        print(json.dumps(status(root, args.role), ensure_ascii=False, indent=2))
    elif args.action == "describe":
        describe(root, args.role)
    elif args.action == "stop":
        print(json.dumps(stop(root)))
    elif args.action == "stop-perception":
        if args.role != "GOS":
            raise RuntimeError("Perception stop requires GOS")
        stop_perception(root)
    elif args.action == "logs":
        latest = json.loads((root / "runtime/one_command_latest.json").read_text())
        run(["tail", "-n", "80", Path(latest["run_dir"]) / "sdk.log"])
    elif args.action == "begin-deploy":
        begin_deploy(root, args.role, args.transaction)
    elif args.action == "finish-deploy":
        with state_lock("metadata.lock"):
            require_transaction(root, args.transaction)
            if sha(root / "models/v1/policy.onnx") != args.model_sha:
                raise RuntimeError("Deployment readback SHA mismatch")
            global_marker = STATE_HOME / "deploy.json"
            if global_marker.exists():
                if json.loads(global_marker.read_text())["transaction"] != args.transaction:
                    raise RuntimeError("Another global deployment owner exists; leave inhibition in place")
            (root / "runtime/one_command_deploy.json").unlink()
            if global_marker.exists():
                global_marker.unlink()
    elif args.action == "check":
        check(root, args.role)
    elif args.action == "stage":
        require_transaction(root, args.transaction)
        stage(root, args.role, args.incoming)
    elif args.action == "activate":
        require_transaction(root, args.transaction)
        activate(root, args.role, args.bundle)
    elif args.action == "supervise":
        return supervise(root, args.run_dir)
    elif args.role == "GOS":
        no_transaction(root)
        start_gos(root)
    else:
        start_aos(root, args.trace_base, args.model_sha)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (RuntimeError, OSError, ValueError, subprocess.SubprocessError) as error:
        print("M20操作失败：" + str(error), file=sys.stderr)
        sys.exit(1)
