"""No robot/SSH/model inference. Exercise the actual wrapper failure paths."""
import importlib.util
import contextlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]


def module(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts/tools" / (name + ".py"))
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


board = module("m20_robot_board")
client = module("m20_robot")


class BoardTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="m20_operator_test_")
        self.root = Path(self.tmp.name)
        (self.root / "runtime").mkdir()
        (self.root / "scripts").mkdir()
        (self.root / "models/v1").mkdir(parents=True)
        (self.root / "models/v1/policy.onnx").write_bytes(b"test-only, not a network")
        self.directory = self.root / "run"
        self.directory.mkdir()
        self.state_patch = patch.object(board, "STATE_HOME", self.root / "operator_state")
        self.state_patch.start()
        board.save(self.directory / "session.json", {"epoch": board.epoch(),
                   "model_sha256": board.sha(self.root / "models/v1/policy.onnx")})

    def tearDown(self):
        self.state_patch.stop()
        self.tmp.cleanup()

    def operator(self, body):
        # Child is the original-operator boundary. Execute its real process,
        # signal, exit and log behavior; never replace the supervisor itself.
        source = "import os,time,signal,sys\nfrom pathlib import Path\n"
        source += "p=Path(os.environ['M20_CONTROL_TRACE_DIR'])/'sdk.log'\n"
        source += "signal.signal(signal.SIGINT, lambda *a: sys.exit(0))\n" + body
        program = self.root / "operator_fixture.py"
        program.write_text(source)
        import shlex
        (self.root / "scripts/operator_start_aos.sh").write_text(
            "#!/bin/bash\nexec " + shlex.join([sys.executable, str(program)]) + "\n")

    def run_supervisor(self):
        with patch.object(board, "processes", return_value=[]):
            code = board.supervise(self.root, self.directory)
        return code, json.loads((self.directory / "state.json").read_text())

    def test_preflight_failure_is_not_ready(self):
        self.operator("p.write_text('preflight rejected\\n')\nsys.exit(2)\n")
        code, state = self.run_supervisor()
        self.assertEqual(code, 1)
        self.assertEqual(state["phase"], "failed_or_cancelled")
        self.assertEqual(state["exit_code"], 2)

    def test_fault_after_ready_stops_real_child_and_does_not_restart(self):
        markers = "\n".join(board.READY_MARKERS)
        self.operator("p.write_text(" + repr("Loading M20 policy:\n" + markers + "\n") + ")\n"
                      "time.sleep(.6)\nwith p.open('a') as f:f.write('[STUDENT_FAULT] target outside limits\\n')\n"
                      "time.sleep(20)\n")
        code, state = self.run_supervisor()
        self.assertEqual(code, 1)
        self.assertEqual(state["phase"], "fault_stopped")
        self.assertFalse(Path("/proc", str(state["pid"])).exists())

    def test_recoverable_fault_stays_alive_until_explicit_reset_and_fatal_still_stops(self):
        markers = "\n".join(board.READY_MARKERS)
        initial = "Loading M20 policy:\n" + markers + "\n"
        state_file = self.directory / "state.json"
        self.operator("p.write_text(" + repr(initial) + ")\ntime.sleep(.6)\n"
                      "with p.open('a') as f:f.write('[STUDENT_RECOVERABLE_FAULT] position limit\\n')\n"
                      "time.sleep(.6)\n"
                      "assert __import__('json').loads(Path(" + repr(str(state_file)) + ").read_text())['phase']=='fault_latched'\n"
                      "with p.open('a') as f:f.write('[STUDENT_RECOVERY_READY] explicit reset\\n')\n"
                      "time.sleep(.6)\n"
                      "assert __import__('json').loads(Path(" + repr(str(state_file)) + ").read_text())['phase']=='handset_ready'\n"
                      "with p.open('a') as f:f.write('[STUDENT_FAULT] stale body\\n')\ntime.sleep(20)\n")
        code, state = self.run_supervisor()
        self.assertEqual(code, 1)
        self.assertEqual(state["phase"], "fault_stopped")
        self.assertEqual(state["exit_code"], 0)

    def test_ready_positive_control_then_clean_exit(self):
        markers = "\n".join(board.READY_MARKERS)
        self.operator("p.write_text(" + repr("Loading M20 policy:\n" + markers + "\n") + ")\ntime.sleep(.6)\n")
        code, state = self.run_supervisor()
        self.assertEqual(code, 0)
        self.assertEqual(state["phase"], "stopped")
        self.assertEqual(state["physical_stability"], "NOT_VALIDATED")

    def test_real_pending_supervisor_and_child_are_cancelled_by_stop(self):
        self.operator("p.write_text('Type START_STUDENT to start\\n')\ntime.sleep(20)\n")
        source = ("import importlib.util,sys\nfrom pathlib import Path\n"
                  "s=importlib.util.spec_from_file_location('board'," + repr(str(ROOT / 'scripts/tools/m20_robot_board.py')) + ")\n"
                  "b=importlib.util.module_from_spec(s);s.loader.exec_module(b)\n"
                  "b.STATE_HOME=Path(" + repr(str(board.STATE_HOME)) + ")\n"
                  "sys.exit(b.supervise(Path(" + repr(str(self.root)) + "),Path(" + repr(str(self.directory)) + ")))\n")
        p = subprocess.Popen([sys.executable, "-c", source], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                             cwd=str(self.root), start_new_session=True)
        try:
            deadline = time.monotonic() + 5
            while not (self.directory / "state.json").exists():
                if p.poll() is not None or time.monotonic() > deadline:
                    self.fail("Pending supervisor failed to register")
                time.sleep(.02)
            child = json.loads((self.directory / "state.json").read_text())["pid"]
            board.stop(self.root)
            p.wait(timeout=5)
            self.assertFalse(Path("/proc", str(child)).exists())
            self.assertEqual(json.loads((self.directory / "state.json").read_text())["phase"], "failed_or_cancelled")
        finally:
            if p.poll() is None:
                p.send_signal(2)
                p.wait(timeout=5)

    def test_interrupt_on_first_post_spawn_receipt_still_cleans_child(self):
        from unittest.mock import Mock
        child = Mock(pid=999, returncode=0)
        child.poll.side_effect = [None, 0]
        original_save = board.save
        interrupted = [False]
        def interrupt(path, value):
            if Path(path).name == "state.json" and not interrupted[0]:
                interrupted[0] = True
                raise KeyboardInterrupt()
            original_save(path, value)
        with patch.object(board, "processes", return_value=[]), patch.object(board.subprocess, "Popen", return_value=child), patch.object(board, "save", side_effect=interrupt):
            with self.assertRaises(KeyboardInterrupt):
                board.supervise(self.root, self.directory)
        child.send_signal.assert_called_once_with(2)
        child.wait.assert_called_once()

    def test_duplicate_controller_never_invokes_operator(self):
        with patch.object(board, "processes", return_value=[{"pid": 123}]), patch.object(board.subprocess, "Popen") as popen:
            with self.assertRaisesRegex(RuntimeError, "appeared"):
                board.supervise(self.root, self.directory)
            popen.assert_not_called()

    def test_same_pid_with_changed_identity_is_never_signalled(self):
        old = {"pid": 123, "start_ticks": "1"}
        with patch.object(board, "processes", return_value=[{"pid": 123, "start_ticks": "2"}]), patch.object(board.os, "kill") as kill:
            with self.assertRaisesRegex(RuntimeError, "identity changed"):
                board.signal_checked(old, 2)
            kill.assert_not_called()

    def test_stop_does_not_load_model_or_preflight(self):
        (self.root / "models/v1/policy.onnx").unlink()
        with patch.object(board, "processes", return_value=[]), patch.object(board, "check", side_effect=AssertionError("positive gate called")):
            self.assertEqual(board.stop(self.root), {"stopped_pids": []})

    def test_activation_failure_restores_previous_link_and_receipt(self):
        import shutil
        shutil.rmtree(self.root / "models/v1")
        (self.root / "models/old").mkdir()
        (self.root / "models/new").mkdir()
        (self.root / "models/v1").symlink_to("old")
        receipt = self.root / "runtime/offline_acceptance.json"
        receipt.write_text('{"old":true}')
        def reject(*args, **kwargs):
            receipt.write_text('{"partially_written":true}')
            raise subprocess.CalledProcessError(2, "offline_acceptance")
        with patch.object(board, "processes", return_value=[]), patch.object(board, "run", side_effect=reject):
            with self.assertRaises(subprocess.CalledProcessError):
                board.activate(self.root, "AOS", "new")
        self.assertEqual(os.readlink(self.root / "models/v1"), "old")
        self.assertEqual(receipt.read_text(), '{"old":true}')

    def test_symlink_bundle_is_rejected_before_copy(self):
        source = self.root / "incoming"
        source.mkdir()
        for name in board.BUNDLE_FILES:
            (source / name).write_bytes(b"x")
        (source / "policy.onnx").unlink()
        (source / "policy.onnx").symlink_to(self.root / "models/v1/policy.onnx")
        with patch.object(board, "processes", return_value=[]), patch.object(board.shutil, "copytree") as copy:
            with self.assertRaisesRegex(RuntimeError, "regular"):
                board.stage(self.root, "AOS", source)
            copy.assert_not_called()

    def test_stop_invalidates_not_yet_spawned_start(self):
        with patch.object(board, "processes", return_value=[]):
            board.stop(self.root)
            with patch.object(board.subprocess, "Popen") as popen:
                with self.assertRaisesRegex(RuntimeError, "cancelled"):
                    board.supervise(self.root, self.directory)
                popen.assert_not_called()

    def test_incomplete_transaction_blocks_native_before_spawn(self):
        with board.state_lock("metadata.lock"):
            board.save(board.STATE_HOME / "deploy.json", {"transaction": "test", "root": str(self.root)})
        with patch.object(board, "processes", return_value=[]), patch.object(board.subprocess, "Popen") as popen:
            with self.assertRaisesRegex(RuntimeError, "Incomplete"):
                board.supervise(self.root, self.directory)
            popen.assert_not_called()

    def test_global_lifetime_lock_rejects_another_release(self):
        with board.state_lock("sdk.lock"):
            with self.assertRaises(BlockingIOError):
                with board.state_lock("sdk.lock"):
                    self.fail("Second lifetime lock acquired")

    def test_persistent_transaction_still_blocks_after_run_state_loss(self):
        board.save(self.root / "runtime/one_command_deploy.json", {"transaction": "partial"})
        self.assertFalse(board.STATE_HOME.exists())
        with self.assertRaisesRegex(RuntimeError, "Incomplete"):
            board.no_transaction(self.root)

    def test_failed_staging_never_publishes_partial_immutable_directory(self):
        source = self.root / "incoming"
        source.mkdir()
        for name in board.BUNDLE_FILES:
            (source / name).write_bytes(b"x")
        (source / "deployment.json").write_text(json.dumps({"checkpoint_sha256": "a" * 64}))
        with patch.object(board, "processes", return_value=[]), patch.object(board, "run", side_effect=RuntimeError("validation rejected")):
            with self.assertRaisesRegex(RuntimeError, "validation rejected"):
                board.stage(self.root, "AOS", source)
        self.assertFalse((self.root / "models" / ("cts_" + "a" * 64)).exists())
        self.assertEqual(list((self.root / "models").glob(".m20_stage_*")), [])

    def test_successful_stage_receipt_names_model_not_last_file(self):
        source = self.root / "incoming"
        source.mkdir()
        for name in board.BUNDLE_FILES:
            (source / name).write_bytes(b"x")
        (source / "deployment.json").write_text(json.dumps({"checkpoint_sha256": "a" * 64}))
        output = io.StringIO()
        with patch.object(board, "processes", return_value=[]), patch.object(board, "run"), contextlib.redirect_stdout(output):
            board.stage(self.root, "AOS", source)
        value = json.loads(output.getvalue().split("M20_STAGED=", 1)[1])
        self.assertEqual(value["bundle"], "cts_" + "a" * 64)
        self.assertTrue((self.root / "models" / value["bundle"]).is_dir())


class ClientTests(unittest.TestCase):
    def test_target_cannot_inject_ssh_options_or_shell(self):
        for value in ("-oProxyCommand=evil", "a;touch x", "host x", "$(id)"):
            with self.assertRaises(Exception):
                client.target(value)
        self.assertEqual(client.target("user@10.21.41.1"), "user@10.21.41.1")

    def test_stop_never_contacts_gos_or_runs_status(self):
        actions = []
        class FakeBoard:
            def __init__(self, host, role, *args):
                self.role = role
                actions.append((role, "created"))
            def connect(self):
                pass
            def action(self, name):
                actions.append((self.role, name))
            def close(self):
                pass
        with patch.object(client, "Board", FakeBoard), patch.object(client.sys.stdin, "isatty", return_value=False), patch.object(client.shutil, "which", return_value="safepush"):
            self.assertEqual(client.main(["stop", "--aos", "aos", "--gos", "unreachable"]), 0)
        self.assertEqual(actions, [("AOS", "created"), ("AOS", "stop")])

    def test_split_board_models_prevent_all_starts(self):
        base = {"model_sha256": "a" * 64, "contract_sha256": "c" * 64,
                "checkpoint_sha256": "d" * 64, "deployment_pending": False}
        other = dict(base, model_sha256="b" * 64)
        with self.assertRaisesRegex(RuntimeError, "differ"):
            client.matching_models(base, other)
        self.assertEqual(client.matching_models(base, dict(base)), base["model_sha256"])
        with self.assertRaisesRegex(RuntimeError, "Incomplete"):
            client.matching_models(base, dict(base, deployment_pending=True))

    def test_full_start_entry_rejects_mismatch_before_any_start(self):
        actions = []
        class FakeBoard:
            def __init__(self, host, role, *args):
                self.role = role
            def connect(self):
                pass
            def action(self, name, *args):
                actions.append((self.role, name))
            def describe(self):
                return {"model_sha256": self.role, "contract_sha256": "same", "checkpoint_sha256": "same", "deployment_pending": False}
            def close(self):
                pass
        with patch.object(client, "Board", FakeBoard), patch.object(client.sys.stdin, "isatty", return_value=True), patch.object(client.shutil, "which", return_value="safepush"):
            with self.assertRaisesRegex(RuntimeError, "differ"):
                client.main(["start", "--aos", "aos", "--gos", "gos"])
        self.assertNotIn(("AOS", "start"), actions)
        self.assertNotIn(("GOS", "start"), actions)

    def test_deploy_requires_explicit_model(self):
        with self.assertRaises(SystemExit):
            client.main(["deploy", "--aos", "aos", "--gos", "gos"])


if __name__ == "__main__":
    unittest.main()
