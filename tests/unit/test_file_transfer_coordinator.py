"""Unit tests for file transfer orchestration."""

import threading
from types import SimpleNamespace
from unittest.mock import MagicMock

from carveracontroller.ui.file_browser.transfer import (
    OP_DOWNLOAD,
    OP_MKDIR_LOCAL,
    OP_MKDIR_REMOTE,
    OP_REMOVE_REMOTE,
    OP_UPLOAD,
    TransferOp,
)
from carveracontroller.ui.file_browser.transfer_coordinator import (
    DIRECTION_DOWNLOAD,
    DIRECTION_UPLOAD,
    ERROR_DOWNLOAD,
    ERROR_LISTING,
    ERROR_MKDIR_REMOTE,
    ERROR_UPLOAD,
    FileTransferCallbacks,
    FileTransferCoordinator,
)


class TransferHarness:
    def __init__(self, tmp_path):
        self.tmp_path = tmp_path
        self.coordinator = None
        self.listings = {}
        self.uploads = []
        self.downloads = []
        self.removals = []
        self.commands = []
        self.progress = []
        self.starts = []
        self.finishes = []
        self.failures = []
        self.conflicts = []
        self.io_active = False
        self.io_cancels = 0
        self.decompressing = False
        self.decompression_checked = threading.Event()
        self.finished = threading.Event()
        self.failed = threading.Event()

    def callbacks(self):
        return FileTransferCallbacks(
            start_remote_listing=self.start_remote_listing,
            start_remote_mkdir=self.start_remote_mkdir,
            start_remote_remove=self.start_remote_remove,
            upload=self.upload,
            download=self.download,
            cancel_io=self.cancel_io,
            is_io_active=lambda: self.io_active,
            is_decompressing=self.is_decompressing,
            on_start=lambda direction, preparing: self.starts.append((direction, preparing)),
            on_progress=lambda stage, op, index, total: self.progress.append((stage, op, index, total)),
            on_finish=self.on_finish,
            on_failure=self.on_failure,
            on_conflicts=lambda ops, names, direction: self.conflicts.append((ops, names, direction)),
        )

    def start_remote_listing(self, path):
        self.coordinator.complete_listing(self.listings[path])

    def start_remote_mkdir(self, _path):
        self.commands.append(("mkdir", _path))
        self.coordinator.complete_mkdir()

    def start_remote_remove(self, path):
        self.commands.append(("remove", path))
        self.removals.append(path)
        self.coordinator.complete_remove()

    def upload(self, op):
        self.commands.append(("upload", op.dest))
        self.uploads.append(op)
        return True

    def download(self, op):
        self.downloads.append(op)
        return 1

    def cancel_io(self):
        self.io_cancels += 1

    def is_decompressing(self):
        self.decompression_checked.set()
        return self.decompressing

    def on_finish(self, direction, refresh):
        self.finishes.append((direction, refresh))
        self.finished.set()

    def on_failure(self, failure):
        self.failures.append(failure)
        self.failed.set()


def test_coordinator_runs_batch_and_replaces_confirmed_type_conflict(tmp_path):
    harness = TransferHarness(tmp_path)
    coordinator = FileTransferCoordinator(harness.callbacks(), command_timeout=0.1)
    harness.coordinator = coordinator

    blocked = tmp_path / "folder"
    blocked.write_text("file blocks downloaded folder")
    ops = [
        TransferOp(OP_MKDIR_LOCAL, dest=str(blocked)),
        TransferOp(OP_MKDIR_REMOTE, dest="/sd/gcodes/new"),
        TransferOp(OP_UPLOAD, source="/tmp/a.nc", dest="/sd/gcodes/new/a.nc"),
        TransferOp(OP_DOWNLOAD, source="/sd/gcodes/b.nc", dest=str(blocked / "b.nc"), size=12),
    ]

    assert coordinator.start_batch(ops, DIRECTION_DOWNLOAD, overwrite=True)
    assert harness.finished.wait(1)
    assert blocked.is_dir()
    assert harness.uploads == [ops[2]]
    assert harness.downloads == [ops[3]]
    assert harness.finishes == [(DIRECTION_DOWNLOAD, True)]
    assert harness.failures == []
    assert coordinator.busy is False


def test_coordinator_prepares_recursive_download_from_listing_snapshots(tmp_path):
    harness = TransferHarness(tmp_path)
    coordinator = FileTransferCoordinator(harness.callbacks(), command_timeout=0.1)
    harness.coordinator = coordinator
    harness.listings = {
        "/sd/gcodes/tools": [
            {
                "name": "inner",
                "path": "/sd/gcodes/tools/inner",
                "is_dir": True,
                "size": 0,
            },
            {
                "name": "a.nc",
                "path": "/sd/gcodes/tools/a.nc",
                "is_dir": False,
                "size": 10,
            },
        ],
        "/sd/gcodes/tools/inner": [
            {
                "name": "b.nc",
                "path": "/sd/gcodes/tools/inner/b.nc",
                "is_dir": False,
                "size": 20,
            }
        ],
    }

    assert coordinator.prepare_download(
        [("/sd/gcodes/tools", True), ("/sd/gcodes/job.nc", False)],
        str(tmp_path),
        [{"path": "/sd/gcodes/job.nc", "is_dir": False, "size": 30}],
    )
    assert harness.finished.wait(1)
    assert [(op.source, op.size) for op in harness.downloads] == [
        ("/sd/gcodes/tools/a.nc", 10),
        ("/sd/gcodes/tools/inner/b.nc", 20),
        ("/sd/gcodes/job.nc", 30),
    ]
    assert harness.starts == [(DIRECTION_DOWNLOAD, True)]
    assert harness.finishes == [(DIRECTION_DOWNLOAD, True)]
    assert harness.failures == []


def test_coordinator_reports_recursive_listing_timeout(tmp_path):
    harness = TransferHarness(tmp_path)
    harness.start_remote_listing = lambda _path: None
    coordinator = FileTransferCoordinator(harness.callbacks(), command_timeout=0.01)
    harness.coordinator = coordinator

    assert coordinator.prepare_download(
        [("/sd/gcodes/tools", True)],
        str(tmp_path),
        [],
    )
    assert harness.failed.wait(1)
    assert harness.failures[0].kind == ERROR_LISTING
    assert harness.failures[0].path == "/sd/gcodes/tools"
    assert harness.finishes == [(DIRECTION_DOWNLOAD, False)]
    assert coordinator.busy is False


def test_cancel_during_listing_does_not_cancel_next_stream_operation(tmp_path):
    harness = TransferHarness(tmp_path)
    listing_started = threading.Event()
    harness.start_remote_listing = lambda _path: listing_started.set()
    coordinator = FileTransferCoordinator(harness.callbacks(), command_timeout=0.01)
    harness.coordinator = coordinator

    assert coordinator.prepare_download(
        [("/sd/gcodes/tools", True)],
        str(tmp_path),
        [],
    )
    assert listing_started.wait(1)
    coordinator.cancel()
    assert harness.finished.wait(1)
    assert harness.io_cancels == 0


def test_cancel_during_active_io_reaches_stream(tmp_path):
    harness = TransferHarness(tmp_path)
    upload_started = threading.Event()
    upload_released = threading.Event()

    def upload(op):
        harness.io_active = True
        upload_started.set()
        upload_released.wait(1)
        harness.io_active = False
        return None

    def cancel_io():
        harness.io_cancels += 1
        upload_released.set()

    harness.upload = upload
    harness.cancel_io = cancel_io
    coordinator = FileTransferCoordinator(harness.callbacks(), command_timeout=0.1)
    harness.coordinator = coordinator

    assert coordinator.start_batch(
        [TransferOp(OP_UPLOAD, source="/tmp/a.nc", dest="/sd/gcodes/a.nc")],
        DIRECTION_UPLOAD,
    )
    assert upload_started.wait(1)
    coordinator.cancel()
    assert harness.finished.wait(1)
    assert harness.io_cancels == 1
    assert harness.failures == []


def test_cancel_during_decompression_waits_before_finishing(tmp_path):
    harness = TransferHarness(tmp_path)

    def upload(op):
        harness.uploads.append(op)
        harness.decompressing = True
        return True

    harness.upload = upload
    coordinator = FileTransferCoordinator(harness.callbacks(), command_timeout=0.1)
    harness.coordinator = coordinator
    first = TransferOp(OP_UPLOAD, source="/tmp/a.nc", dest="/sd/gcodes/a.nc")
    second = TransferOp(OP_UPLOAD, source="/tmp/b.nc", dest="/sd/gcodes/b.nc")

    assert coordinator.start_batch([first, second], DIRECTION_UPLOAD)
    assert harness.decompression_checked.wait(1)
    coordinator.cancel()
    assert harness.finished.wait(0.05) is False
    assert coordinator.busy is True

    harness.decompressing = False
    assert harness.finished.wait(1)
    assert harness.uploads == [first]
    assert coordinator.busy is False


def test_coordinator_removes_remote_type_conflict_before_copy(tmp_path):
    harness = TransferHarness(tmp_path)
    coordinator = FileTransferCoordinator(harness.callbacks(), command_timeout=0.1)
    harness.coordinator = coordinator
    ops = [
        TransferOp(OP_REMOVE_REMOTE, dest="/sd/gcodes/tools"),
        TransferOp(OP_MKDIR_REMOTE, dest="/sd/gcodes/tools"),
        TransferOp(OP_UPLOAD, source="/tmp/a.nc", dest="/sd/gcodes/tools/a.nc"),
    ]

    assert coordinator.start_batch(ops, DIRECTION_UPLOAD, overwrite=True)
    assert harness.finished.wait(1)
    assert harness.commands == [
        ("remove", "/sd/gcodes/tools"),
        ("mkdir", "/sd/gcodes/tools"),
        ("upload", "/sd/gcodes/tools/a.nc"),
    ]
    assert harness.failures == []


def test_coordinator_replaces_nested_file_that_blocks_remote_folder(tmp_path):
    harness = TransferHarness(tmp_path)
    mkdir_attempts = 0

    def start_remote_mkdir(path):
        nonlocal mkdir_attempts
        mkdir_attempts += 1
        harness.commands.append(("mkdir", path))
        harness.coordinator.complete_mkdir(failed=mkdir_attempts == 1)

    harness.start_remote_mkdir = start_remote_mkdir
    harness.listings["/sd/gcodes/tools"] = [
        {"name": "inner", "path": "/sd/gcodes/tools/inner", "is_dir": False},
    ]
    coordinator = FileTransferCoordinator(harness.callbacks(), command_timeout=0.1)
    harness.coordinator = coordinator
    mkdir = TransferOp(OP_MKDIR_REMOTE, dest="/sd/gcodes/tools/inner")

    assert coordinator.start_batch([mkdir], DIRECTION_UPLOAD, overwrite=True)
    assert harness.finished.wait(1)
    assert harness.commands == [
        ("mkdir", mkdir.dest),
        ("remove", mkdir.dest),
        ("mkdir", mkdir.dest),
    ]
    assert harness.failures == []


def test_coordinator_replaces_nested_folder_that_blocks_upload(tmp_path):
    harness = TransferHarness(tmp_path)

    def upload(op):
        harness.commands.append(("upload", op.dest))
        harness.uploads.append(op)
        return len(harness.uploads) > 1

    harness.upload = upload
    harness.listings["/sd/gcodes/tools"] = [
        {"name": "job.nc", "path": "/sd/gcodes/tools/job.nc", "is_dir": True},
    ]
    coordinator = FileTransferCoordinator(harness.callbacks(), command_timeout=0.1)
    harness.coordinator = coordinator
    op = TransferOp(OP_UPLOAD, source="/tmp/job.nc", dest="/sd/gcodes/tools/job.nc")

    assert coordinator.start_batch([op], DIRECTION_UPLOAD, overwrite=True)
    assert harness.finished.wait(1)
    assert harness.commands == [
        ("upload", op.dest),
        ("remove", op.dest),
        ("upload", op.dest),
    ]
    assert harness.failures == []


def test_coordinator_preflights_nested_compressed_upload_type_conflict(tmp_path):
    harness = TransferHarness(tmp_path)
    harness.listings["/sd/gcodes/tools"] = [
        {"name": "job.nc", "path": "/sd/gcodes/tools/job.nc", "is_dir": True},
    ]
    coordinator = FileTransferCoordinator(harness.callbacks(), command_timeout=0.1)
    harness.coordinator = coordinator
    op = TransferOp(
        OP_UPLOAD,
        source="/tmp/job.nc",
        dest="/sd/gcodes/tools/job.nc",
        check_remote_type=True,
    )

    assert coordinator.start_batch([op], DIRECTION_UPLOAD, overwrite=True)
    assert harness.finished.wait(1)
    assert harness.commands == [
        ("remove", op.dest),
        ("upload", op.dest),
    ]
    assert harness.uploads == [op]
    assert harness.failures == []


def test_coordinator_reports_upload_timeout(tmp_path):
    harness = TransferHarness(tmp_path)
    harness.upload = lambda _op: None
    coordinator = FileTransferCoordinator(harness.callbacks(), command_timeout=0.1)
    harness.coordinator = coordinator
    upload = TransferOp(OP_UPLOAD, source="/tmp/a.nc", dest="/sd/gcodes/a.nc")

    assert coordinator.start_batch([upload], DIRECTION_UPLOAD)
    assert harness.failed.wait(1)
    assert harness.failures[0].kind == ERROR_UPLOAD
    assert harness.failures[0].path == upload.source
    assert harness.finishes == [(DIRECTION_UPLOAD, True)]


def test_coordinator_reports_unexpected_negative_download_result(tmp_path):
    harness = TransferHarness(tmp_path)
    harness.download = lambda _op: -1
    coordinator = FileTransferCoordinator(harness.callbacks(), command_timeout=0.1)
    harness.coordinator = coordinator
    download = TransferOp(OP_DOWNLOAD, source="/sd/gcodes/a.nc", dest=str(tmp_path / "a.nc"))

    assert coordinator.start_batch([download], DIRECTION_DOWNLOAD)
    assert harness.failed.wait(1)
    assert harness.failures[0].kind == ERROR_DOWNLOAD
    assert harness.failures[0].path == download.source


def test_coordinator_reports_real_remote_mkdir_failure(tmp_path):
    harness = TransferHarness(tmp_path)
    harness.start_remote_mkdir = lambda _path: harness.coordinator.complete_mkdir(failed=True)
    harness.listings["/sd/gcodes"] = []
    coordinator = FileTransferCoordinator(harness.callbacks(), command_timeout=0.1)
    harness.coordinator = coordinator
    upload = TransferOp(OP_UPLOAD, source="/tmp/a.nc", dest="/sd/gcodes/tools/a.nc")

    assert coordinator.start_batch(
        [TransferOp(OP_MKDIR_REMOTE, dest="/sd/gcodes/tools"), upload],
        DIRECTION_UPLOAD,
    )
    assert harness.failed.wait(1)
    assert harness.failures[0].kind == ERROR_MKDIR_REMOTE
    assert harness.failures[0].path == "/sd/gcodes/tools"
    assert harness.uploads == []


def test_coordinator_accepts_mkdir_error_when_folder_exists(tmp_path):
    harness = TransferHarness(tmp_path)
    harness.start_remote_mkdir = lambda _path: harness.coordinator.complete_mkdir(failed=True)
    harness.listings["/sd/gcodes"] = [
        {"name": "tools", "path": "/sd/gcodes/tools", "is_dir": True},
    ]
    coordinator = FileTransferCoordinator(harness.callbacks(), command_timeout=0.1)
    harness.coordinator = coordinator
    upload = TransferOp(OP_UPLOAD, source="/tmp/a.nc", dest="/sd/gcodes/tools/a.nc")

    assert coordinator.start_batch(
        [TransferOp(OP_MKDIR_REMOTE, dest="/sd/gcodes/tools"), upload],
        DIRECTION_UPLOAD,
    )
    assert harness.finished.wait(1)
    assert harness.uploads == [upload]
    assert harness.failures == []


def test_batch_download_propagates_post_processing_failure(tmp_path, monkeypatch):
    from carveracontroller.main import Makera

    destination = tmp_path / "job.nc"

    def download(tmp_filename, _md5, _progress):
        with open(tmp_filename, "wb") as output:
            output.write(b"\x00\x00compressed")
        return 12

    root = Makera.__new__(Makera)
    root.downloading_config = False
    root.file_transfer = SimpleNamespace(active=True)
    root.controller = SimpleNamespace(
        comms=SimpleNamespace(uses_framed_transfer=True),
        stream=SimpleNamespace(download=download),
        pauseStream=MagicMock(),
        resumeStream=MagicMock(),
        downloadCommand=MagicMock(),
    )
    root._decompress_downloaded_file_in_place = MagicMock(return_value=False)
    root._ingest_machine_gcode_thumbnail = MagicMock()
    root.update_recent_remote_dir_list = MagicMock()
    monkeypatch.setattr("carveracontroller.main.App.get_running_app", lambda: None)

    result = root.doDownload(
        "/sd/gcodes/job.nc",
        str(destination),
        show_progress=False,
        open_after=False,
    )

    assert result is None
    root._decompress_downloaded_file_in_place.assert_called_once_with(
        str(destination),
        show_error=False,
    )
    root._ingest_machine_gcode_thumbnail.assert_not_called()


def test_main_considers_controller_load_command_transfer_busy():
    from carveracontroller.main import Makera

    root = Makera.__new__(Makera)
    root.uploading = False
    root.downloading = False
    root.decompstatus = False
    root.file_transfer = SimpleNamespace(busy=False)
    root.controller = SimpleNamespace(loadNUM=1)

    assert root._file_xfer_busy() is True


def test_browser_listing_does_not_interrupt_active_transfer():
    from carveracontroller.main import Makera

    root = Makera.__new__(Makera)
    root.file_transfer = SimpleNamespace(busy=True)
    root.controller = SimpleNamespace(
        loadNUM=0,
        loadEOF=True,
        loadERR=True,
        lsCommand=MagicMock(),
    )

    assert root.loadRemoteDir("/sd/gcodes") is False
    root.controller.lsCommand.assert_not_called()

    assert root.loadRemoteDir("/sd/gcodes/tools", transfer=True) is True
    root.controller.lsCommand.assert_called_once()
    assert root.controller.loadEOF is False
    assert root.controller.loadERR is False


def test_transfer_remote_mutations_reset_load_completion_flags():
    from carveracontroller.main import Makera

    root = Makera.__new__(Makera)
    root.controller = SimpleNamespace(
        sendNUM=1,
        loadNUM=0,
        loadEOF=True,
        loadERR=True,
        mkdirCommand=MagicMock(),
        rmCommand=MagicMock(),
    )

    root._start_transfer_remote_mkdir("/sd/gcodes/tools")
    assert root.controller.loadEOF is False
    assert root.controller.loadERR is False
    root.controller.mkdirCommand.assert_called_once()

    root.controller.loadNUM = 0
    root.controller.loadEOF = True
    root.controller.loadERR = True
    root._start_transfer_remote_remove("/sd/gcodes/job.nc")
    assert root.controller.loadEOF is False
    assert root.controller.loadERR is False
    root.controller.rmCommand.assert_called_once()
