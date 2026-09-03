"""Plan file/folder copy ops for the file browser. No machine I/O."""

from __future__ import annotations

import os
from collections.abc import Iterable
from dataclasses import dataclass, replace

from .sources import machine_path_display

OP_MKDIR_REMOTE = "mkdir_remote"
OP_REMOVE_REMOTE = "remove_remote"
OP_MKDIR_LOCAL = "mkdir_local"
OP_UPLOAD = "upload"
OP_DOWNLOAD = "download"


@dataclass(frozen=True)
class TransferOp:
    kind: str
    source: str = ""
    dest: str = ""
    size: int = 0
    check_remote_type: bool = False


def is_hidden_name(name: str) -> bool:
    return bool(name) and str(name).startswith(".")


def join_machine_path(*parts: str) -> str:
    chunks: list[str] = []
    for part in parts:
        if not part:
            continue
        for piece in str(part).replace("\\", "/").split("/"):
            if piece and piece != ".":
                chunks.append(piece)
    return "/" + "/".join(chunks)


def machine_basename(path: str) -> str:
    return str(path).replace("\\", "/").rstrip("/").rsplit("/", 1)[-1]


def _relative_segments(rel: str) -> list[str]:
    if not rel or rel in (os.curdir, "."):
        return []
    return [piece for piece in rel.replace("\\", "/").split("/") if piece and piece != "."]


def _filter_hidden(names: list[str]) -> None:
    names[:] = [name for name in names if not is_hidden_name(name)]


def plan_local_upload(paths: Iterable[str], dest_machine_dir: str) -> list[TransferOp]:
    """Walk selected local files/folders into mkdir + upload ops under dest_machine_dir."""
    ops: list[TransferOp] = []
    dest_root = machine_path_display(dest_machine_dir)
    for raw in paths:
        if not raw:
            continue
        path = os.path.normpath(raw)
        name = os.path.basename(path)
        if not name or is_hidden_name(name):
            continue
        if os.path.isdir(path):
            remote_root = join_machine_path(dest_root, name)
            ops.append(TransferOp(OP_MKDIR_REMOTE, dest=remote_root))
            try:
                walker = os.walk(path)
            except OSError:
                continue
            for dirpath, dirnames, filenames in walker:
                _filter_hidden(dirnames)
                visible_files = [filename for filename in filenames if not is_hidden_name(filename)]
                rel = os.path.relpath(dirpath, path)
                segments = _relative_segments(rel)
                remote_dir = join_machine_path(remote_root, *segments) if segments else remote_root
                if segments:
                    ops.append(TransferOp(OP_MKDIR_REMOTE, dest=remote_dir))
                for filename in visible_files:
                    local = os.path.join(dirpath, filename)
                    ops.append(
                        TransferOp(
                            OP_UPLOAD,
                            source=os.path.normpath(local),
                            dest=join_machine_path(remote_dir, filename),
                        )
                    )
        elif os.path.isfile(path):
            ops.append(TransferOp(OP_UPLOAD, source=path, dest=join_machine_path(dest_root, name)))
    return ops


def _listing_by_name(entries: Iterable[dict]) -> dict[str, dict]:
    by_name: dict[str, dict] = {}
    for entry in entries or []:
        name = entry.get("name") or ""
        if name:
            by_name[name] = entry
    return by_name


def top_level_upload_conflicts(
    ops: Iterable[TransferOp],
    dest_machine_dir: str,
    entries: Iterable[dict],
) -> list[str]:
    """Names in the current machine folder that would be overwritten or collide."""
    dest_root = machine_path_display(dest_machine_dir)
    by_name = _listing_by_name(entries)
    conflicts: list[str] = []
    seen: set[str] = set()
    for op in ops:
        dest = machine_path_display(op.dest)
        parent, _, name = dest.rstrip("/").rpartition("/")
        if not name or join_machine_path(parent) != dest_root:
            continue
        existing = by_name.get(name)
        if existing is None or name in seen:
            continue
        exists_dir = bool(existing.get("is_dir"))
        if op.kind == OP_UPLOAD and not exists_dir:
            seen.add(name)
            conflicts.append(name)
        elif op.kind == OP_UPLOAD and exists_dir:
            seen.add(name)
            conflicts.append(name)
        elif op.kind == OP_MKDIR_REMOTE:
            seen.add(name)
            conflicts.append(name)
    return conflicts


def skip_existing_remote_mkdirs(
    ops: Iterable[TransferOp],
    dest_machine_dir: str,
    entries: Iterable[dict],
) -> list[TransferOp]:
    """Drop top-level mkdir ops when that folder already exists on the machine."""
    dest_root = machine_path_display(dest_machine_dir)
    existing_dirs = {entry.get("name") for entry in (entries or []) if entry.get("is_dir") and entry.get("name")}
    kept: list[TransferOp] = []
    for op in ops:
        if op.kind == OP_MKDIR_REMOTE:
            dest = machine_path_display(op.dest)
            parent, _, name = dest.rstrip("/").rpartition("/")
            if name in existing_dirs and join_machine_path(parent) == dest_root:
                continue
        kept.append(op)
    return kept


def add_remote_type_replacements(
    ops: Iterable[TransferOp],
    dest_machine_dir: str,
    entries: Iterable[dict],
) -> list[TransferOp]:
    """Insert removals before confirmed top-level file/directory type conflicts."""
    planned = list(ops)
    dest_root = machine_path_display(dest_machine_dir)
    by_name = _listing_by_name(entries)
    replacements: set[str] = set()
    for op in planned:
        if op.kind not in (OP_MKDIR_REMOTE, OP_UPLOAD):
            continue
        dest = machine_path_display(op.dest)
        parent, _, name = dest.rstrip("/").rpartition("/")
        if not name or join_machine_path(parent) != dest_root:
            continue
        existing = by_name.get(name)
        if existing is None:
            continue
        incoming_is_dir = op.kind == OP_MKDIR_REMOTE
        if incoming_is_dir != bool(existing.get("is_dir")):
            replacements.add(dest)

    result: list[TransferOp] = []
    inserted: set[str] = set()
    for op in planned:
        dest = machine_path_display(op.dest)
        if dest in replacements and dest not in inserted:
            result.append(TransferOp(OP_REMOVE_REMOTE, dest=dest))
            inserted.add(dest)
        result.append(op)
    return result


def require_nested_upload_type_checks(
    ops: Iterable[TransferOp],
    dest_machine_dir: str,
) -> list[TransferOp]:
    """Mark nested uploads whose compressed target cannot reveal a directory collision."""
    dest_root = machine_path_display(dest_machine_dir)
    result: list[TransferOp] = []
    for op in ops:
        dest = machine_path_display(op.dest)
        parent, _, _name = dest.rstrip("/").rpartition("/")
        if op.kind == OP_UPLOAD and join_machine_path(parent) != dest_root:
            op = replace(op, check_remote_type=True)
        result.append(op)
    return result


def plan_machine_file_download(
    remote_paths: Iterable[str],
    dest_dir: str,
    entries: Iterable[dict] = (),
) -> list[TransferOp]:
    """Non-recursive: each remote file into dest_dir / basename."""
    sizes = {
        machine_path_display(entry.get("path") or ""): int(entry.get("size") or 0)
        for entry in entries or []
        if entry.get("path") and not entry.get("is_dir")
    }
    ops: list[TransferOp] = []
    for path in remote_paths:
        name = machine_basename(path)
        if not name or is_hidden_name(name) or not dest_dir:
            continue
        size = sizes.get(machine_path_display(path), 0)
        ops.append(TransferOp(OP_DOWNLOAD, source=path, dest=os.path.join(dest_dir, name), size=size))
    return ops


def plan_download_folder_root(remote_folder: str, dest_dir: str) -> tuple[str, TransferOp]:
    name = machine_basename(remote_folder)
    local = os.path.join(dest_dir, name)
    return local, TransferOp(OP_MKDIR_LOCAL, dest=local)


def plan_download_dir_level(entries: Iterable[dict], local_dir: str) -> tuple[list[TransferOp], list[tuple[str, str]]]:
    """Ops for one machine listing, plus (remote_dir, local_dir) pairs to recurse."""
    ops: list[TransferOp] = []
    recurse: list[tuple[str, str]] = []
    for entry in entries or []:
        name = entry.get("name") or ""
        if is_hidden_name(name):
            continue
        remote = entry.get("path") or ""
        local = os.path.join(local_dir, name)
        if entry.get("is_dir"):
            ops.append(TransferOp(OP_MKDIR_LOCAL, dest=local))
            recurse.append((remote, local))
        else:
            ops.append(TransferOp(OP_DOWNLOAD, source=remote, dest=local, size=int(entry.get("size") or 0)))
    return ops, recurse


def local_download_conflicts(ops: Iterable[TransferOp]) -> list[str]:
    conflicts: list[str] = []
    seen: set[str] = set()
    for op in ops:
        dest = op.dest
        if not dest:
            continue
        key = os.path.normcase(os.path.normpath(dest))
        if key in seen:
            continue
        if op.kind == OP_DOWNLOAD and os.path.lexists(dest):
            seen.add(key)
            conflicts.append(dest)
        elif op.kind == OP_MKDIR_LOCAL and local_destination_requires_replacement(op):
            seen.add(key)
            conflicts.append(dest)
    return conflicts


def local_destination_requires_replacement(op: TransferOp) -> bool:
    """Whether overwriting *op.dest* requires removing an incompatible local node."""
    if not op.dest or not os.path.lexists(op.dest):
        return False
    if os.path.islink(op.dest):
        return True
    if op.kind == OP_MKDIR_LOCAL:
        return not os.path.isdir(op.dest)
    if op.kind == OP_DOWNLOAD:
        return os.path.isdir(op.dest)
    return False
