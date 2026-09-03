"""Unit tests for file browser listing, grouping, and action state."""

import os

from carveracontroller.ui.file_browser.sources import (
    ICON_FILE,
    ICON_FIRMWARE,
    ICON_FOLDER,
    ICON_GCODE,
    KIND_FILE,
    KIND_FOLDER,
    KIND_HEADER,
    LOCATION_DEVICE,
    LOCATION_MACHINE,
    compute_action_state,
    current_file_banner,
    current_row_badge,
    device_tab_path_display,
    download_dest_tooltip,
    file_type_key,
    file_type_label,
    group_and_sort_entries,
    is_compact_width,
    is_machine_root,
    is_under_machine_root,
    list_device_directory,
    local_child_path,
    local_dir_has_file,
    local_sibling_path,
    machine_listing_has,
    machine_parent_dir,
    machine_path_display,
    machine_tab_path_display,
    mkdir_local,
    remove_local_path,
    rename_local_path,
    row_icon,
    trim_breadcrumb_pairs,
    upload_dest_tooltip,
)

IDENTITY = lambda text: text  # noqa: E731


def _entry(name, *, is_dir=False, size=10, date=100, path=None):
    return {
        "name": name,
        "path": path or f"/sd/gcodes/{name}",
        "is_dir": is_dir,
        "size": 0 if is_dir else size,
        "date": date,
    }


def test_file_type_key_and_label():
    assert file_type_key("part.nc") == "gcode"
    assert file_type_key("job.gcode") == "gcode"
    assert file_type_key("fw.bin") == "firmware"
    assert file_type_key("job.lz") == "compressed"
    assert file_type_key("notes.txt") == "other"
    assert file_type_label("part.nc", translate=IDENTITY) == "G-code"
    assert file_type_label("fw.bin", translate=IDENTITY) == "Firmware"


def test_row_icon_by_kind_and_type():
    assert row_icon(KIND_FOLDER, "tools") == ICON_FOLDER
    assert row_icon(KIND_FILE, "part.nc") == ICON_GCODE
    assert row_icon(KIND_FILE, "fw.bin") == ICON_FIRMWARE
    assert row_icon(KIND_FILE, "notes.txt") == ICON_FILE
    assert row_icon(KIND_FILE, "job.lz") == ICON_FILE


def test_group_and_sort_puts_folders_above_files():
    entries = [
        _entry("b.nc", date=20, size=200),
        _entry("alpha", is_dir=True, date=10),
        _entry("a.nc", date=30, size=50),
        _entry("zeta", is_dir=True, date=40),
    ]
    rows = group_and_sort_entries(entries, sort_key="name", reverse=False, translate=IDENTITY)
    kinds = [row["kind"] for row in rows]
    names = [row["filename"] for row in rows]
    assert KIND_HEADER not in kinds
    assert kinds == [KIND_FOLDER, KIND_FOLDER, KIND_FILE, KIND_FILE]
    assert names == ["alpha", "zeta", "a.nc", "b.nc"]


def test_group_sort_by_date_and_size():
    entries = [
        _entry("old.nc", date=1, size=999),
        _entry("new.nc", date=50, size=1),
    ]
    by_date = group_and_sort_entries(entries, sort_key="date", reverse=True, translate=IDENTITY)
    file_names = [row["filename"] for row in by_date if row["kind"] == KIND_FILE]
    assert file_names == ["new.nc", "old.nc"]
    by_size = group_and_sort_entries(entries, sort_key="size", reverse=True, translate=IDENTITY)
    file_names = [row["filename"] for row in by_size if row["kind"] == KIND_FILE]
    assert file_names == ["old.nc", "new.nc"]


def test_firmware_mode_keeps_dirs_and_bin_only():
    entries = [
        _entry("keep", is_dir=True),
        _entry("fw.bin"),
        _entry("job.nc"),
    ]
    rows = group_and_sort_entries(entries, firmware_mode=True, translate=IDENTITY)
    names = [row["filename"] for row in rows if row["kind"] != KIND_HEADER]
    assert names == ["keep", "fw.bin"]


def test_search_filters_by_name():
    entries = [_entry("bracket.nc"), _entry("lid.nc"), _entry("tools", is_dir=True)]
    rows = group_and_sort_entries(entries, keyword="lid", translate=IDENTITY)
    names = [row["filename"] for row in rows if row["kind"] != KIND_HEADER]
    assert names == ["lid.nc"]


def test_current_job_badge_and_selection():
    entries = [_entry("job.nc", path="/sd/gcodes/job.nc")]
    rows = group_and_sort_entries(
        entries,
        current_job_path="/sd/gcodes/job.nc",
        highlight_path="/sd/gcodes/job.nc",
        translate=IDENTITY,
    )
    file_row = next(row for row in rows if row["kind"] == KIND_FILE)
    assert file_row["is_current_job"] is True
    assert file_row["current_badge"] == "Current"
    assert file_row["selected"] is True
    assert "G-code" in file_row["subtitle"]
    assert file_row["thumbnail"] == ""


def test_current_preview_badge():
    entries = [_entry("part.cnc", path="/home/me/part.cnc")]
    rows = group_and_sort_entries(
        entries,
        current_job_path="/home/me/part.cnc",
        current_is_preview=True,
        translate=IDENTITY,
    )
    file_row = next(row for row in rows if row["kind"] == KIND_FILE)
    assert file_row["is_current_job"] is True
    assert file_row["current_badge"] == "Current (Preview)"
    other = group_and_sort_entries(entries, translate=IDENTITY)
    assert next(row for row in other if row["kind"] == KIND_FILE)["current_badge"] == ""
    assert current_row_badge(is_current=True, is_preview=False, translate=IDENTITY) == "Current"
    assert current_row_badge(is_current=True, is_preview=True, translate=IDENTITY) == "Current (Preview)"
    assert current_row_badge(is_current=False, is_preview=True, translate=IDENTITY) == ""


def test_row_passes_through_thumbnail_for_files_only():
    entries = [
        _entry("tools", is_dir=True),
        _entry("job.nc", path="/sd/gcodes/job.nc"),
    ]
    entries[1]["thumbnail"] = "/tmp/job.png"
    entries[0]["thumbnail"] = "/tmp/should-ignore.png"
    rows = group_and_sort_entries(entries, translate=IDENTITY)
    folder = next(row for row in rows if row["kind"] == KIND_FOLDER)
    file_row = next(row for row in rows if row["kind"] == KIND_FILE)
    assert folder["thumbnail"] == ""
    assert file_row["thumbnail"] == "/tmp/job.png"


def test_current_file_banner_source_and_clear():
    assert current_file_banner("/sd/gcodes/job.nc", "", translate=IDENTITY) == (
        "job.nc",
        "Machine",
        True,
    )
    assert current_file_banner("", "/home/me/part.cnc", translate=IDENTITY) == (
        "part.cnc",
        "Local (Preview)",
        True,
    )
    assert current_file_banner("", "", translate=IDENTITY) == ("None", "", False)
    name, badge, can_clear = current_file_banner("/sd/gcodes/on-machine.nc", "/tmp/local.nc", translate=IDENTITY)
    assert (name, badge, can_clear) == ("on-machine.nc", "Machine", True)


def test_file_rows_include_type_size_date():
    rows = group_and_sort_entries([_entry("part.nc", size=2048, date=1_700_000_000)], translate=IDENTITY)
    file_row = next(row for row in rows if row["kind"] == KIND_FILE)
    assert file_row["file_type"] == "G-code"
    assert file_row["filesize"]
    assert file_row["filedate"]
    assert file_row["subtitle"].startswith("G-code")
    assert file_row["icon"] == ICON_GCODE


def test_machine_listing_has_ignores_directories():
    entries = [_entry("job.nc"), _entry("job.nc", is_dir=True, path="/sd/gcodes/job.nc/")]
    entries[1]["name"] = "tools"
    assert machine_listing_has(entries, "job.nc") is True
    assert machine_listing_has(entries, "missing.nc") is False
    assert machine_listing_has([_entry("tools", is_dir=True)], "tools") is False


def test_machine_root_and_parent():
    assert is_machine_root("/sd/gcodes")
    assert is_machine_root("/sd/gcodes/")
    assert machine_parent_dir("/sd/gcodes") is None
    assert machine_parent_dir("/sd/gcodes/jobs") == "/sd/gcodes"
    assert is_under_machine_root("/sd/gcodes")
    assert is_under_machine_root("/sd/gcodes/jobs/batch")
    assert is_under_machine_root("\\sd\\gcodes\\jobs")
    assert is_under_machine_root("/sd") is False
    assert is_under_machine_root("/tmp/gcodes") is False


def test_machine_path_display_keeps_sd_root():
    assert machine_path_display("/sd/gcodes") == "/sd/gcodes"
    assert machine_path_display("/sd/gcodes/") == "/sd/gcodes"
    assert machine_path_display("/sd/gcodes/jobs") == "/sd/gcodes/jobs"
    assert machine_path_display("/sd/gcodes/jobs/batch") == "/sd/gcodes/jobs/batch"
    assert machine_path_display("\\sd\\gcodes\\jobs") == "/sd/gcodes/jobs"
    assert machine_path_display("") == "/sd/gcodes"
    assert upload_dest_tooltip("/sd/gcodes/jobs", translate=IDENTITY) == "Upload to: /sd/gcodes/jobs"
    assert download_dest_tooltip("/home/user/gcodes", translate=IDENTITY) == "Download to: /home/user/gcodes"


def test_device_tab_path_display_native_separators():
    assert device_tab_path_display("") == ""
    assert device_tab_path_display("/home/user/gcodes") == "/home/user/gcodes"
    assert device_tab_path_display("/home/user/gcodes/") == "/home/user/gcodes"


def test_device_tab_path_display_windows_format(monkeypatch):
    import ntpath

    monkeypatch.setattr(
        "carveracontroller.ui.file_browser.sources.os.path.normpath",
        ntpath.normpath,
    )
    assert device_tab_path_display("C:\\Users\\me\\gcodes\\jobs") == "C:\\Users\\me\\gcodes\\jobs"
    assert device_tab_path_display("C:\\Users\\me\\gcodes\\jobs\\") == "C:\\Users\\me\\gcodes\\jobs"


def test_machine_tab_path_display_connected_and_not():
    assert machine_tab_path_display("/sd/gcodes/jobs", connected=True, translate=IDENTITY) == "/sd/gcodes/jobs"
    assert machine_tab_path_display("/sd/gcodes/jobs", connected=False, translate=IDENTITY) == "Not connected"


def test_trim_machine_breadcrumbs_drops_sd_and_empty_root():
    paths, labels = trim_breadcrumb_pairs(
        ["/", "/sd", "/sd/gcodes", "/sd/gcodes/jobs"],
        ["", "sd", "gcodes", "jobs"],
        machine=True,
    )
    assert paths == ["/sd/gcodes", "/sd/gcodes/jobs"]
    assert labels == ["gcodes", "jobs"]


def test_list_device_directory_skips_dotfiles(tmp_path):
    (tmp_path / "visible.nc").write_text("g")
    (tmp_path / ".hidden.nc").write_text("g")
    (tmp_path / "sub").mkdir()
    (tmp_path / ".skipdir").mkdir()
    entries = list_device_directory(str(tmp_path))
    names = {item["name"] for item in entries}
    assert names == {"visible.nc", "sub"}
    folders = [item for item in entries if item["is_dir"]]
    files = [item for item in entries if not item["is_dir"]]
    assert folders[0]["name"] == "sub"
    assert files[0]["size"] > 0


def test_local_dir_has_file(tmp_path):
    (tmp_path / "job.nc").write_text("g")
    (tmp_path / "tools").mkdir()
    assert local_dir_has_file(str(tmp_path), "job.nc") is True
    assert local_dir_has_file(str(tmp_path), "/sd/gcodes/job.nc") is True
    assert local_dir_has_file(str(tmp_path), "missing.nc") is False
    assert local_dir_has_file(str(tmp_path), "tools") is False
    assert local_dir_has_file("", "job.nc") is False
    assert local_dir_has_file(str(tmp_path), "") is False


def test_local_child_and_sibling_paths(tmp_path):
    src = str(tmp_path / "job.nc")
    assert local_child_path(str(tmp_path), "folder") == str(tmp_path / "folder")
    assert local_sibling_path(src, "renamed.nc") == str(tmp_path / "renamed.nc")
    assert local_child_path(str(tmp_path), "") == ""
    assert local_child_path(str(tmp_path), "../escape") == ""
    assert local_child_path(str(tmp_path), "a/b") == ""
    assert local_sibling_path(src, "..") == ""


def test_local_mkdir_rename_and_remove(tmp_path):
    folder = mkdir_local(str(tmp_path), "tools")
    assert os.path.isdir(folder)
    src = tmp_path / "job.nc"
    src.write_text("g")
    dest = str(tmp_path / "part.nc")
    rename_local_path(str(src), dest)
    assert os.path.isfile(dest)
    assert not src.exists()
    remove_local_path(dest)
    assert not os.path.exists(dest)
    nested = tmp_path / "tools" / "inner.nc"
    nested.write_text("g")
    remove_local_path(folder)
    assert not os.path.exists(folder)


def test_compact_width_helper():
    assert is_compact_width(400, threshold=720) is True
    assert is_compact_width(800, threshold=720) is False


def test_action_state_device_file_selected():
    state = compute_action_state(
        location=LOCATION_DEVICE,
        firmware_mode=False,
        ios=False,
        machine_connected=True,
        machine_idle=True,
        selected_is_file=True,
        selected_count=1,
        multi_select_mode=False,
    )
    assert state.show_preview is True
    assert state.show_upload is True
    assert state.show_upload_and_use is True
    assert state.show_download is False
    assert state.show_rename is True
    assert state.show_delete is True
    assert state.show_new_folder is True
    assert state.show_multi_toggle is True
    assert state.primary == "upload_and_use"


def test_action_state_device_requires_idle_for_upload():
    state = compute_action_state(
        location=LOCATION_DEVICE,
        firmware_mode=False,
        ios=False,
        machine_connected=True,
        machine_idle=False,
        selected_is_file=True,
        selected_count=1,
        multi_select_mode=False,
    )
    assert state.show_preview is True
    assert state.show_upload is False
    assert state.show_upload_and_use is False
    assert state.show_rename is True
    assert state.show_delete is True
    assert state.show_new_folder is True
    assert state.primary == ""


def test_action_state_device_folder_and_multi():
    folder = compute_action_state(
        location=LOCATION_DEVICE,
        firmware_mode=False,
        ios=False,
        machine_connected=True,
        machine_idle=True,
        selected_is_file=False,
        selected_count=1,
        multi_select_mode=False,
    )
    assert folder.show_preview is False
    assert folder.show_upload is True
    assert folder.show_upload_and_use is False
    assert folder.show_rename is True
    assert folder.show_delete is True
    assert folder.show_new_folder is True
    assert folder.primary == ""
    multi = compute_action_state(
        location=LOCATION_DEVICE,
        firmware_mode=False,
        ios=False,
        machine_connected=True,
        machine_idle=True,
        selected_is_file=False,
        selected_count=2,
        multi_select_mode=True,
    )
    assert multi.show_delete is True
    assert multi.show_cancel_multi is True
    assert multi.show_preview is False
    assert multi.show_upload is True
    assert multi.show_upload_and_use is False
    assert multi.show_rename is False
    assert multi.show_new_folder is False
    assert multi.primary == "delete"
    busy_multi = compute_action_state(
        location=LOCATION_DEVICE,
        firmware_mode=False,
        ios=False,
        machine_connected=True,
        machine_idle=False,
        selected_is_file=False,
        selected_count=2,
        multi_select_mode=True,
    )
    assert busy_multi.show_upload is False
    assert busy_multi.show_delete is True


def test_action_state_firmware_upload_only():
    state = compute_action_state(
        location=LOCATION_DEVICE,
        firmware_mode=True,
        ios=False,
        machine_connected=True,
        machine_idle=True,
        selected_is_file=True,
        selected_count=1,
        multi_select_mode=False,
    )
    assert state.show_preview is False
    assert state.show_upload is True
    assert state.show_upload_and_use is False
    assert state.search_enabled is False
    assert state.show_download is False
    assert state.show_rename is False
    assert state.show_multi_toggle is False
    assert state.primary == "upload"


def test_action_state_machine_file_and_folder():
    file_state = compute_action_state(
        location=LOCATION_MACHINE,
        firmware_mode=False,
        ios=False,
        machine_connected=True,
        machine_idle=True,
        selected_is_file=True,
        selected_count=1,
        multi_select_mode=False,
    )
    assert file_state.show_use_as_job is True
    assert file_state.show_download is True
    assert file_state.show_rename is True
    assert file_state.show_delete is True
    assert file_state.show_new_folder is True
    assert file_state.primary == "use_as_job"
    folder_state = compute_action_state(
        location=LOCATION_MACHINE,
        firmware_mode=False,
        ios=False,
        machine_connected=True,
        machine_idle=True,
        selected_is_file=False,
        selected_count=1,
        multi_select_mode=False,
    )
    assert folder_state.show_use_as_job is False
    assert folder_state.show_download is True
    assert folder_state.show_rename is True
    assert folder_state.show_delete is True
    busy = compute_action_state(
        location=LOCATION_MACHINE,
        firmware_mode=False,
        ios=False,
        machine_connected=True,
        machine_idle=False,
        selected_is_file=True,
        selected_count=1,
        multi_select_mode=False,
    )
    assert busy.show_use_as_job is True
    assert busy.show_download is False


def test_action_state_machine_disconnected_and_multi():
    disconnected = compute_action_state(
        location=LOCATION_MACHINE,
        firmware_mode=False,
        ios=False,
        machine_connected=False,
        machine_idle=False,
        selected_is_file=False,
        selected_count=0,
        multi_select_mode=False,
    )
    assert disconnected.show_use_as_job is False
    assert disconnected.show_download is False
    assert disconnected.search_enabled is False
    multi = compute_action_state(
        location=LOCATION_MACHINE,
        firmware_mode=False,
        ios=False,
        machine_connected=True,
        machine_idle=True,
        selected_is_file=False,
        selected_count=3,
        multi_select_mode=True,
    )
    assert multi.show_delete is True
    assert multi.show_cancel_multi is True
    assert multi.show_use_as_job is False
    assert multi.show_download is True
    assert multi.primary == "delete"


def test_action_state_ios_device_uses_browse():
    state = compute_action_state(
        location=LOCATION_DEVICE,
        firmware_mode=False,
        ios=True,
        machine_connected=True,
        machine_idle=True,
        selected_is_file=False,
        selected_count=0,
        multi_select_mode=False,
    )
    assert state.show_ios_browse is True
    assert state.show_places is False
    assert state.show_multi_toggle is False
    assert state.show_rename is False


def test_join_machine_path_and_basename():
    from carveracontroller.ui.file_browser.transfer import join_machine_path, machine_basename

    assert join_machine_path("/sd/gcodes", "tools", "a.nc") == "/sd/gcodes/tools/a.nc"
    assert join_machine_path("\\sd\\gcodes", "inner") == "/sd/gcodes/inner"
    assert machine_basename("/sd/gcodes/tools/") == "tools"
    assert machine_basename("\\sd\\gcodes\\job.nc") == "job.nc"


def test_plan_local_upload_nested_empty_and_hidden(tmp_path):
    from carveracontroller.ui.file_browser.transfer import (
        OP_MKDIR_REMOTE,
        OP_UPLOAD,
        plan_local_upload,
    )

    folder = tmp_path / "tools"
    (folder / "inner").mkdir(parents=True)
    (folder / "inner" / "a.nc").write_text("g")
    (folder / ".secret").write_text("x")
    (folder / "inner" / ".skip").write_text("y")
    (folder / "empty").mkdir()
    job = tmp_path / "job.nc"
    job.write_text("h")

    ops = plan_local_upload([str(folder), str(job)], "/sd/gcodes")
    mkdir_dests = {op.dest for op in ops if op.kind == OP_MKDIR_REMOTE}
    uploads = [(op.source, op.dest) for op in ops if op.kind == OP_UPLOAD]
    assert "/sd/gcodes/tools" in mkdir_dests
    assert "/sd/gcodes/tools/inner" in mkdir_dests
    assert "/sd/gcodes/tools/empty" in mkdir_dests
    assert (os.path.normpath(str(folder / "inner" / "a.nc")), "/sd/gcodes/tools/inner/a.nc") in uploads
    assert (os.path.normpath(str(job)), "/sd/gcodes/job.nc") in uploads
    assert all(".secret" not in dest and ".skip" not in dest for _src, dest in uploads)
    assert all(op.kind in (OP_MKDIR_REMOTE, OP_UPLOAD) for op in ops)


def test_top_level_upload_conflicts_and_skip_existing_mkdir():
    from carveracontroller.ui.file_browser.transfer import (
        OP_MKDIR_REMOTE,
        OP_REMOVE_REMOTE,
        OP_UPLOAD,
        TransferOp,
        add_remote_type_replacements,
        require_nested_upload_type_checks,
        skip_existing_remote_mkdirs,
        top_level_upload_conflicts,
    )

    entries = [
        {"name": "job.nc", "is_dir": False},
        {"name": "tools", "is_dir": True},
        {"name": "notes.txt", "is_dir": False},
        {"name": "blocked.nc", "is_dir": True},
    ]
    ops = [
        TransferOp(OP_UPLOAD, source="/tmp/job.nc", dest="/sd/gcodes/job.nc"),
        TransferOp(OP_MKDIR_REMOTE, dest="/sd/gcodes/tools"),
        TransferOp(OP_UPLOAD, source="/tmp/a.nc", dest="/sd/gcodes/tools/a.nc"),
        TransferOp(OP_MKDIR_REMOTE, dest="/sd/gcodes/notes.txt"),
        TransferOp(OP_UPLOAD, source="/tmp/blocked.nc", dest="/sd/gcodes/blocked.nc"),
        TransferOp(OP_UPLOAD, source="/tmp/new.nc", dest="/sd/gcodes/new.nc"),
    ]
    conflicts = top_level_upload_conflicts(ops, "/sd/gcodes", entries)
    assert conflicts == ["job.nc", "tools", "notes.txt", "blocked.nc"]
    filtered = skip_existing_remote_mkdirs(ops, "/sd/gcodes", entries)
    assert [op.dest for op in filtered if op.kind == OP_MKDIR_REMOTE] == ["/sd/gcodes/notes.txt"]
    replacement_ops = add_remote_type_replacements(filtered, "/sd/gcodes", entries)
    assert [(op.kind, op.dest) for op in replacement_ops] == [
        (OP_UPLOAD, "/sd/gcodes/job.nc"),
        (OP_UPLOAD, "/sd/gcodes/tools/a.nc"),
        (OP_REMOVE_REMOTE, "/sd/gcodes/notes.txt"),
        (OP_MKDIR_REMOTE, "/sd/gcodes/notes.txt"),
        (OP_REMOVE_REMOTE, "/sd/gcodes/blocked.nc"),
        (OP_UPLOAD, "/sd/gcodes/blocked.nc"),
        (OP_UPLOAD, "/sd/gcodes/new.nc"),
    ]
    checked_ops = require_nested_upload_type_checks(replacement_ops, "/sd/gcodes")
    assert [op.dest for op in checked_ops if op.check_remote_type] == ["/sd/gcodes/tools/a.nc"]


def test_plan_download_and_local_conflicts(tmp_path):
    from carveracontroller.ui.file_browser.transfer import (
        OP_DOWNLOAD,
        OP_MKDIR_LOCAL,
        TransferOp,
        local_download_conflicts,
        local_destination_requires_replacement,
        plan_download_dir_level,
        plan_download_folder_root,
        plan_machine_file_download,
    )

    dest = tmp_path / "inbox"
    dest.mkdir()
    existing = dest / "job.nc"
    existing.write_text("old")
    ops = plan_machine_file_download(
        ["/sd/gcodes/job.nc", "/sd/gcodes/new.nc"],
        str(dest),
        [
            {"path": "/sd/gcodes/job.nc", "is_dir": False, "size": 42},
            {"path": "\\sd\\gcodes\\new.nc", "is_dir": False, "size": 17},
        ],
    )
    assert [op.dest for op in ops] == [str(existing), str(dest / "new.nc")]
    assert [op.size for op in ops] == [42, 17]
    assert local_download_conflicts(ops) == [str(existing)]

    first_duplicate = dest / "first" / "config.nc"
    second_duplicate = dest / "second" / "config.nc"
    first_duplicate.parent.mkdir()
    second_duplicate.parent.mkdir()
    first_duplicate.write_text("old")
    second_duplicate.write_text("old")
    duplicate_ops = [
        TransferOp(OP_DOWNLOAD, dest=str(first_duplicate)),
        TransferOp(OP_DOWNLOAD, dest=str(second_duplicate)),
    ]
    assert local_download_conflicts(duplicate_ops) == [
        str(first_duplicate),
        str(second_duplicate),
    ]

    blocked_folder = dest / "tools"
    blocked_folder.write_text("not a directory")
    local_root, mkdir_op = plan_download_folder_root("/sd/gcodes/tools", str(dest))
    assert mkdir_op.kind == OP_MKDIR_LOCAL
    assert local_root == str(dest / "tools")
    assert local_destination_requires_replacement(mkdir_op) is True
    assert local_download_conflicts([mkdir_op]) == [str(blocked_folder)]
    level_ops, recurse = plan_download_dir_level(
        [
            {"name": "inner", "path": "/sd/gcodes/tools/inner", "is_dir": True, "size": 0},
            {"name": "a.nc", "path": "/sd/gcodes/tools/a.nc", "is_dir": False, "size": 12},
            {"name": ".hidden", "path": "/sd/gcodes/tools/.hidden", "is_dir": False, "size": 1},
        ],
        local_root,
    )
    assert any(op.kind == OP_MKDIR_LOCAL and op.dest == str(dest / "tools" / "inner") for op in level_ops)
    assert any(op.kind == OP_DOWNLOAD and op.size == 12 for op in level_ops)
    assert recurse == [("/sd/gcodes/tools/inner", str(dest / "tools" / "inner"))]
    assert all(".hidden" not in op.dest for op in level_ops)
