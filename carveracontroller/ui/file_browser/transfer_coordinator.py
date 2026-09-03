"""Coordinate asynchronous file-browser transfers without depending on Kivy."""

from __future__ import annotations

import logging
import os
import shutil
import threading
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Any

from .transfer import (
    OP_DOWNLOAD,
    OP_MKDIR_LOCAL,
    OP_MKDIR_REMOTE,
    OP_REMOVE_REMOTE,
    OP_UPLOAD,
    TransferOp,
    local_destination_requires_replacement,
    local_download_conflicts,
    machine_basename,
    machine_path_display,
    plan_download_dir_level,
    plan_download_folder_root,
    plan_machine_file_download,
)

logger = logging.getLogger(__name__)

DIRECTION_UPLOAD = "upload"
DIRECTION_DOWNLOAD = "download"

STAGE_MKDIR_REMOTE = "mkdir_remote"
STAGE_REMOVE_REMOTE = "remove_remote"
STAGE_UPLOAD = "upload"
STAGE_DOWNLOAD = "download"

ERROR_LISTING = "listing"
ERROR_REPLACE_LOCAL = "replace_local"
ERROR_REMOVE_REMOTE = "remove_remote"
ERROR_MKDIR_REMOTE = "mkdir_remote"
ERROR_MKDIR_LOCAL = "mkdir_local"
ERROR_UPLOAD = "upload"
ERROR_DOWNLOAD = "download"
ERROR_INTERNAL = "internal"


@dataclass(frozen=True)
class TransferFailure:
    kind: str
    path: str = ""
    detail: str = ""


@dataclass(frozen=True)
class FileTransferCallbacks:
    start_remote_listing: Callable[[str], None]
    start_remote_mkdir: Callable[[str], None]
    start_remote_remove: Callable[[str], None]
    upload: Callable[[TransferOp], Any]
    download: Callable[[TransferOp], Any]
    cancel_io: Callable[[], None]
    is_io_active: Callable[[], bool]
    is_decompressing: Callable[[], bool]
    on_start: Callable[[str, bool], None]
    on_progress: Callable[[str, TransferOp, int, int], None]
    on_finish: Callable[[str, bool], None]
    on_failure: Callable[[TransferFailure], None]
    on_conflicts: Callable[[list[TransferOp], list[str], str], None]


class _ListingError(Exception):
    def __init__(self, path: str):
        super().__init__(path)
        self.path = path


class FileTransferCoordinator:
    """Own one batch transfer, including recursive planning and command waits."""

    def __init__(self, callbacks: FileTransferCallbacks, command_timeout: float = 8.0):
        self._callbacks = callbacks
        self._command_timeout = command_timeout
        self._lock = threading.Lock()
        self._active = False
        self._canceled = False
        self._listing_in_flight = False
        self._listing_event = None
        self._listing_result = None
        self._listing_failed = False
        self._mkdir_in_flight = False
        self._mkdir_event = None
        self._mkdir_failed = False
        self._remove_in_flight = False
        self._remove_event = None
        self._remove_failed = False

    @property
    def active(self) -> bool:
        with self._lock:
            return self._active

    @property
    def canceled(self) -> bool:
        with self._lock:
            return self._canceled

    @property
    def listing_pending(self) -> bool:
        with self._lock:
            return self._listing_in_flight

    @property
    def mkdir_pending(self) -> bool:
        with self._lock:
            return self._mkdir_in_flight

    @property
    def remove_pending(self) -> bool:
        with self._lock:
            return self._remove_in_flight

    @property
    def busy(self) -> bool:
        with self._lock:
            return self._active or self._listing_in_flight or self._mkdir_in_flight or self._remove_in_flight

    def start_batch(self, ops: Iterable[TransferOp], direction: str, overwrite: bool = False) -> bool:
        if not self._activate():
            return False
        threading.Thread(
            target=self._run_batch,
            args=(list(ops), direction, False, overwrite),
            daemon=True,
        ).start()
        return True

    def prepare_download(
        self,
        selections: Iterable[tuple[str, bool]],
        dest_dir: str,
        current_entries: Iterable[dict],
    ) -> bool:
        if not self._activate():
            return False
        selected = list(selections)
        entries = [dict(entry) for entry in current_entries or []]
        self._callbacks.on_start(DIRECTION_DOWNLOAD, True)
        threading.Thread(
            target=self._prepare_download,
            args=(selected, dest_dir, entries),
            daemon=True,
        ).start()
        return True

    def cancel(self) -> None:
        with self._lock:
            if not self._active:
                return
            self._canceled = True
        # Listing, mkdir/remove, local compression, and remote decompression
        # do not have an XMODEM operation to cancel. Setting its canceled flag
        # in those stages would abort the next transfer instead.
        if self._callbacks.is_io_active():
            self._callbacks.cancel_io()

    def complete_listing(self, entries: Iterable[dict] | None, failed: bool = False) -> bool:
        with self._lock:
            if not self._listing_in_flight:
                return False
            event = self._listing_event
            self._listing_result = None if failed else list(entries or [])
            self._listing_failed = failed
            self._listing_in_flight = False
            self._listing_event = None
        if event is not None:
            event.set()
        return True

    def complete_mkdir(self, failed: bool = False) -> bool:
        with self._lock:
            if not self._mkdir_in_flight:
                return False
            event = self._mkdir_event
            self._mkdir_failed = failed
            self._mkdir_in_flight = False
            self._mkdir_event = None
        if event is not None:
            event.set()
        return True

    def complete_remove(self, failed: bool = False) -> bool:
        with self._lock:
            if not self._remove_in_flight:
                return False
            event = self._remove_event
            self._remove_failed = failed
            self._remove_in_flight = False
            self._remove_event = None
        if event is not None:
            event.set()
        return True

    def _activate(self) -> bool:
        with self._lock:
            if self._active or self._listing_in_flight or self._mkdir_in_flight or self._remove_in_flight:
                return False
            self._active = True
            self._canceled = False
        return True

    def _deactivate(self) -> None:
        with self._lock:
            self._active = False

    def _list_remote_wait(self, path: str) -> list[dict] | None:
        event = threading.Event()
        with self._lock:
            if self._listing_in_flight:
                return None
            self._listing_in_flight = True
            self._listing_event = event
            self._listing_result = None
            self._listing_failed = False
        try:
            self._callbacks.start_remote_listing(path)
        except Exception:
            logger.exception("Could not start remote listing for %s", path)
            self.complete_listing(None, failed=True)
        if not event.wait(timeout=self._command_timeout):
            with self._lock:
                if self._listing_event is event:
                    self._listing_in_flight = False
                    self._listing_event = None
                    self._listing_failed = True
            return None
        with self._lock:
            if self._listing_failed:
                return None
            return list(self._listing_result or [])

    def _remote_entry_wait(
        self,
        path: str,
        *,
        listing_cache: dict[str, list[dict]] | None = None,
        require_listing: bool = False,
    ) -> dict | None:
        normalized = machine_path_display(path).rstrip("/")
        parent, _, name = normalized.rpartition("/")
        parent = parent or "/"
        if listing_cache is not None and parent in listing_cache:
            listing = listing_cache[parent]
        else:
            listing = self._list_remote_wait(parent)
            if listing is not None and listing_cache is not None:
                listing_cache[parent] = listing
        if listing is None:
            if require_listing:
                raise _ListingError(parent)
            return None
        return next(
            (
                entry
                for entry in listing
                if machine_basename(entry.get("path") or entry.get("name") or "") == name
            ),
            None,
        )

    def _mkdir_remote_wait(self, path: str, replace_incompatible: bool = False) -> bool:
        event = threading.Event()
        with self._lock:
            if self._mkdir_in_flight:
                return False
            self._mkdir_in_flight = True
            self._mkdir_event = event
            self._mkdir_failed = False
        try:
            self._callbacks.start_remote_mkdir(path)
        except Exception:
            logger.exception("Could not start remote mkdir for %s", path)
            self.complete_mkdir(failed=True)
        if not event.wait(timeout=self._command_timeout):
            with self._lock:
                if self._mkdir_event is event:
                    self._mkdir_in_flight = False
                    self._mkdir_event = None
            return False
        with self._lock:
            failed = self._mkdir_failed
        if not failed:
            return True
        # The controller reports mkdir-existing as an error. Only suppress that
        # error when a fresh parent listing confirms the destination is a folder.
        existing = self._remote_entry_wait(path)
        if existing is None:
            return False
        if existing.get("is_dir"):
            return True
        if not replace_incompatible or not self._remove_remote_wait(path):
            return False
        return self._mkdir_remote_wait(path)

    def _remove_remote_wait(self, path: str) -> bool:
        event = threading.Event()
        with self._lock:
            if self._remove_in_flight:
                return False
            self._remove_in_flight = True
            self._remove_event = event
            self._remove_failed = False
        try:
            self._callbacks.start_remote_remove(path)
        except Exception:
            logger.exception("Could not start remote removal for %s", path)
            self.complete_remove(failed=True)
        if not event.wait(timeout=self._command_timeout):
            with self._lock:
                if self._remove_event is event:
                    self._remove_in_flight = False
                    self._remove_event = None
            return False
        with self._lock:
            return not self._remove_failed

    def _plan_download_ops(
        self,
        selections: list[tuple[str, bool]],
        dest_dir: str,
        current_entries: list[dict],
    ) -> list[TransferOp] | None:
        ops: list[TransferOp] = []
        for path, is_dir in selections:
            if self.canceled:
                return None
            if is_dir:
                local_root, mkdir_op = plan_download_folder_root(path, dest_dir)
                ops.append(mkdir_op)
                ops.extend(self._plan_download_tree(path, local_root))
            else:
                ops.extend(plan_machine_file_download([path], dest_dir, current_entries))
        return ops

    def _plan_download_tree(self, remote_dir: str, local_dir: str) -> list[TransferOp]:
        listing = self._list_remote_wait(remote_dir)
        if listing is None:
            raise _ListingError(remote_dir)
        ops, recurse = plan_download_dir_level(listing, local_dir)
        for remote, local in recurse:
            if self.canceled:
                return ops
            ops.extend(self._plan_download_tree(remote, local))
        return ops

    def _prepare_download(
        self,
        selections: list[tuple[str, bool]],
        dest_dir: str,
        current_entries: list[dict],
    ) -> None:
        try:
            ops = self._plan_download_ops(selections, dest_dir, current_entries)
            if self.canceled or ops is None:
                self._deactivate()
                self._callbacks.on_finish(DIRECTION_DOWNLOAD, False)
                return
            conflicts = local_download_conflicts(ops)
            if conflicts:
                self._deactivate()
                self._callbacks.on_finish(DIRECTION_DOWNLOAD, False)
                self._callbacks.on_conflicts(ops, conflicts, DIRECTION_DOWNLOAD)
                return
            self._run_batch(ops, DIRECTION_DOWNLOAD, True, False)
        except _ListingError as exc:
            self._deactivate()
            self._callbacks.on_finish(DIRECTION_DOWNLOAD, False)
            if not self.canceled:
                self._callbacks.on_failure(TransferFailure(ERROR_LISTING, exc.path))
        except Exception as exc:
            logger.exception("Unexpected error while preparing download")
            self._deactivate()
            self._callbacks.on_finish(DIRECTION_DOWNLOAD, False)
            if not self.canceled:
                self._callbacks.on_failure(TransferFailure(ERROR_INTERNAL, detail=str(exc)))

    def _run_batch(
        self,
        ops: list[TransferOp],
        direction: str,
        progress_already_open: bool,
        overwrite: bool,
    ) -> None:
        failure = None
        file_ops = [op for op in ops if op.kind in (OP_UPLOAD, OP_DOWNLOAD)]
        total_files = len(file_ops)
        file_index = 0
        remote_listing_cache: dict[str, list[dict]] = {}
        try:
            if not progress_already_open:
                self._callbacks.on_start(direction, False)
            for op in ops:
                if self.canceled:
                    break
                if overwrite and local_destination_requires_replacement(op):
                    try:
                        if os.path.isdir(op.dest) and not os.path.islink(op.dest):
                            shutil.rmtree(op.dest)
                        else:
                            os.remove(op.dest)
                    except OSError as exc:
                        failure = TransferFailure(ERROR_REPLACE_LOCAL, op.dest, str(exc))
                        break
                if op.kind == OP_REMOVE_REMOTE:
                    self._callbacks.on_progress(STAGE_REMOVE_REMOTE, op, file_index, total_files)
                    if not self._remove_remote_wait(op.dest):
                        failure = TransferFailure(ERROR_REMOVE_REMOTE, op.dest)
                        break
                    continue
                if op.kind == OP_MKDIR_REMOTE:
                    self._callbacks.on_progress(STAGE_MKDIR_REMOTE, op, file_index, total_files)
                    if not self._mkdir_remote_wait(op.dest, replace_incompatible=overwrite):
                        failure = TransferFailure(ERROR_MKDIR_REMOTE, op.dest)
                        break
                    continue
                if op.kind == OP_MKDIR_LOCAL:
                    try:
                        os.makedirs(op.dest, exist_ok=True)
                    except OSError as exc:
                        failure = TransferFailure(ERROR_MKDIR_LOCAL, op.dest, str(exc))
                        break
                    continue
                if op.kind == OP_UPLOAD:
                    if overwrite and op.check_remote_type:
                        existing = self._remote_entry_wait(
                            op.dest,
                            listing_cache=remote_listing_cache,
                            require_listing=True,
                        )
                        if existing is not None and existing.get("is_dir"):
                            self._callbacks.on_progress(STAGE_REMOVE_REMOTE, op, file_index, total_files)
                            if not self._remove_remote_wait(op.dest):
                                failure = TransferFailure(ERROR_REMOVE_REMOTE, op.dest)
                                break
                    file_index += 1
                    self._callbacks.on_progress(STAGE_UPLOAD, op, file_index, total_files)
                    result = self._callbacks.upload(op)
                    if overwrite and not self.canceled and result is not None and not result:
                        existing = self._remote_entry_wait(op.dest)
                        if existing is not None and existing.get("is_dir") and self._remove_remote_wait(op.dest):
                            result = self._callbacks.upload(op)
                    if self.canceled:
                        break
                    if result is None or not result:
                        failure = TransferFailure(ERROR_UPLOAD, op.source)
                        break
                    while self._callbacks.is_decompressing():
                        time.sleep(0.1)
                    continue
                if op.kind == OP_DOWNLOAD:
                    file_index += 1
                    self._callbacks.on_progress(STAGE_DOWNLOAD, op, file_index, total_files)
                    result = self._callbacks.download(op)
                    if self.canceled or result is None or result < 0:
                        if not self.canceled:
                            failure = TransferFailure(ERROR_DOWNLOAD, op.source)
                        break
        except _ListingError as exc:
            failure = TransferFailure(ERROR_LISTING, exc.path)
        except Exception as exc:
            logger.exception("Unexpected error during %s batch", direction)
            failure = TransferFailure(ERROR_INTERNAL, detail=str(exc))
        finally:
            self._deactivate()
            self._callbacks.on_finish(direction, True)
            if failure is not None and not self.canceled:
                self._callbacks.on_failure(failure)
