"""Unit tests for StockSettingsPopup session reset (no Kivy widget tree)."""

import pytest

from carveracontroller.addons.stock.ui.StockSettingsPopup import (
    StockSettingsPopup,
    _parse_finite_float,
    _parse_positive_float,
)


def test_reset_for_loaded_file_uses_snapshot_not_ui():
    popup = StockSettingsPopup.__new__(StockSettingsPopup)
    custom = {
        "shape": {"kind": "rectangular", "width_mm": 150, "length_mm": 80, "height_mm": 20},
        "origin": {
            "xy_corner": "BL",
            "z_reference": "top",
            "offset_x_mm": 1.5,
            "offset_y_mm": -2.0,
            "offset_z_mm": 0.25,
        },
        "show_stock": True,
        "simulate_cut": True,
        "mesh_while_playing": False,
        "voxel_resolution": "high",
        "checkpoint_level": "medium",
    }
    popup._settings_snapshot = dict(custom)
    written = []
    popup._write_settings_to_ui = lambda s: written.append(dict(s))

    out = popup.reset_for_loaded_file()

    assert out["shape"]["width_mm"] == 150
    assert out["origin"]["xy_corner"] == "BL"
    assert out["origin"]["offset_x_mm"] == 1.5
    assert out["origin"]["offset_y_mm"] == -2.0
    assert out["origin"]["offset_z_mm"] == 0.25
    assert out["voxel_resolution"] == "high"
    assert out["checkpoint_level"] == "medium"
    assert out["show_stock"] is False
    assert out["simulate_cut"] is False
    assert out["mesh_while_playing"] is False
    assert popup._settings_snapshot == out
    assert written == [out]


def test_reset_for_loaded_file_preserves_cylindrical_shape():
    popup = StockSettingsPopup.__new__(StockSettingsPopup)
    custom = {
        "shape": {"kind": "cylindrical", "diameter_mm": 75, "height_mm": 15},
        "origin": {"xy_corner": "center", "z_reference": "bottom"},
        "show_stock": True,
        "simulate_cut": True,
        "voxel_resolution": "medium",
        "checkpoint_level": "high",
    }
    popup._settings_snapshot = dict(custom)
    written = []
    popup._write_settings_to_ui = lambda s: written.append(dict(s))

    out = popup.reset_for_loaded_file()

    assert out["shape"]["kind"] == "cylindrical"
    assert out["shape"]["diameter_mm"] == 75
    assert out["shape"]["height_mm"] == 15
    assert out["show_stock"] is False
    assert out["simulate_cut"] is False
    assert written == [out]


def test_reset_for_loaded_file_preserves_mesh_while_playing():
    popup = StockSettingsPopup.__new__(StockSettingsPopup)
    custom = {
        "shape": {"kind": "rectangular", "width_mm": 50, "length_mm": 40, "height_mm": 5},
        "origin": {"xy_corner": "BL", "z_reference": "top"},
        "show_stock": True,
        "simulate_cut": True,
        "mesh_while_playing": True,
        "voxel_resolution": "low",
        "checkpoint_level": "low",
    }
    popup._settings_snapshot = dict(custom)
    popup._write_settings_to_ui = lambda _s: None

    out = popup.reset_for_loaded_file()

    assert out["mesh_while_playing"] is True
    assert out["show_stock"] is False
    assert out["simulate_cut"] is False


def test_reset_for_loaded_file_when_toggles_already_off():
    popup = StockSettingsPopup.__new__(StockSettingsPopup)
    custom = {
        "shape": {"kind": "rectangular", "width_mm": 50, "length_mm": 40, "height_mm": 5},
        "origin": {"xy_corner": "TR", "z_reference": "bottom"},
        "show_stock": False,
        "simulate_cut": False,
        "voxel_resolution": "low",
        "checkpoint_level": "low",
    }
    popup._settings_snapshot = dict(custom)
    written = []
    popup._write_settings_to_ui = lambda s: written.append(dict(s))

    out = popup.reset_for_loaded_file()

    assert out["shape"]["length_mm"] == 40
    assert out["show_stock"] is False
    assert out["simulate_cut"] is False
    assert written == [out]
    assert popup._settings_snapshot == out


@pytest.mark.parametrize("text", ["nan", "NaN", "inf", "Infinity", "1e999"])
def test_parse_positive_float_rejects_non_finite(text):
    with pytest.raises(ValueError, match="finite positive"):
        _parse_positive_float(text, "Width")


def test_parse_positive_float_rejects_non_positive():
    with pytest.raises(ValueError, match="finite positive"):
        _parse_positive_float("0", "Width")
    with pytest.raises(ValueError, match="finite positive"):
        _parse_positive_float("-1", "Width")


def test_parse_positive_float_accepts_normal_values():
    assert _parse_positive_float("12.5", "Width") == 12.5
    assert _parse_positive_float("1,5", "Width") == 1.5


@pytest.mark.parametrize("text", ["nan", "NaN", "inf", "Infinity", "1e999"])
def test_parse_finite_float_rejects_non_finite(text):
    with pytest.raises(ValueError, match="finite number"):
        _parse_finite_float(text, "Offset X")


def test_parse_finite_float_accepts_zero_and_negative():
    assert _parse_finite_float("0", "Offset X") == 0.0
    assert _parse_finite_float("-3.25", "Offset Y") == -3.25
    assert _parse_finite_float("1,5", "Offset Z") == 1.5


def test_parse_finite_float_rejects_empty():
    with pytest.raises(ValueError, match="required"):
        _parse_finite_float("  ", "Offset X")


class _ChildList(list):
    """List that mimics Kivy ``children`` membership for remove/insert tests."""


class _RowStub:
    def __init__(self):
        self.parent = None
        self.height = 40
        self.opacity = 1
        self.disabled = False


class _FormStub:
    def __init__(self, children):
        self.children = _ChildList(children)

    def remove_widget(self, child):
        self.children.remove(child)
        child.parent = None

    def add_widget(self, child, index=0):
        self.children.insert(index, child)
        child.parent = self


def test_update_dimension_field_visibility_removes_length_for_cylinder():
    popup = StockSettingsPopup.__new__(StockSettingsPopup)
    popup._shape_pairs = [("Rectangular", "rectangular"), ("Cylinder", "cylindrical")]
    popup._fit_event = None
    popup._schedule_fit_popup_height = lambda *_a: None

    row_length = _RowStub()
    row_height = _RowStub()
    form = _FormStub([row_height, row_length])
    row_length.parent = form
    row_height.parent = form
    txt_length = type("T", (), {"disabled": False})()
    lbl_width = type("L", (), {"text": "Width (mm)"})()
    spn_shape = type("S", (), {"text": "Cylinder"})()

    popup.ids = {
        "dim_form": form,
        "row_length": row_length,
        "row_height": row_height,
        "txt_length": txt_length,
        "lbl_width": lbl_width,
        "spn_shape": spn_shape,
    }

    popup._update_dimension_field_visibility()

    assert row_length not in form.children
    assert row_length.parent is None
    assert txt_length.disabled is True
    assert lbl_width.text == "Diameter (mm)"
    assert form.children == [row_height]


def test_update_dimension_field_visibility_restores_length_for_rectangular():
    popup = StockSettingsPopup.__new__(StockSettingsPopup)
    popup._shape_pairs = [("Rectangular", "rectangular"), ("Cylinder", "cylindrical")]
    popup._fit_event = None
    popup._schedule_fit_popup_height = lambda *_a: None

    row_length = _RowStub()
    row_height = _RowStub()
    form = _FormStub([row_height])
    row_height.parent = form
    txt_length = type("T", (), {"disabled": True})()
    lbl_width = type("L", (), {"text": "Diameter (mm)"})()
    spn_shape = type("S", (), {"text": "Rectangular"})()

    popup.ids = {
        "dim_form": form,
        "row_length": row_length,
        "row_height": row_height,
        "txt_length": txt_length,
        "lbl_width": lbl_width,
        "spn_shape": spn_shape,
    }

    popup._update_dimension_field_visibility()

    assert row_length in form.children
    assert form.children.index(row_length) == form.children.index(row_height) + 1
    assert txt_length.disabled is False
    assert lbl_width.text == "Width (mm)"


class _SizedStub:
    def __init__(self, height: float, **extra):
        self.height = height
        for key, value in extra.items():
            setattr(self, key, value)


def test_fit_popup_height_clamps_to_window_fraction(monkeypatch):
    popup = StockSettingsPopup.__new__(StockSettingsPopup)
    popup._fit_event = None
    popup.ids = {
        "content_box": _SizedStub(0.0, padding=(16, 16, 16, 16), spacing=10),
        "form_body": _SizedStub(500.0, minimum_height=500.0),
        "lbl_title": _SizedStub(36.0),
        "lbl_help": _SizedStub(48.0),
        "btn_row": _SizedStub(44.0),
    }
    monkeypatch.setattr(
        "carveracontroller.addons.stock.ui.StockSettingsPopup.Window",
        type("W", (), {"height": 400.0}),
    )
    monkeypatch.setattr(
        "carveracontroller.addons.stock.ui.StockSettingsPopup.dp",
        lambda value: float(value),
    )

    popup._fit_popup_height()

    assert popup.height == pytest.approx(400.0 * 0.72)


def test_fit_popup_height_uses_content_when_short(monkeypatch):
    popup = StockSettingsPopup.__new__(StockSettingsPopup)
    popup._fit_event = None
    popup.ids = {
        "content_box": _SizedStub(0.0, padding=(16, 16, 16, 16), spacing=10),
        "form_body": _SizedStub(200.0, minimum_height=200.0),
        "lbl_title": _SizedStub(36.0),
        "lbl_help": _SizedStub(48.0),
        "btn_row": _SizedStub(44.0),
    }
    monkeypatch.setattr(
        "carveracontroller.addons.stock.ui.StockSettingsPopup.Window",
        type("W", (), {"height": 900.0}),
    )
    monkeypatch.setattr(
        "carveracontroller.addons.stock.ui.StockSettingsPopup.dp",
        lambda value: float(value),
    )

    popup._fit_popup_height()

    # padding 32 + title 36 + help 48 + buttons 44 + 3*spacing 30 + form 200
    assert popup.height == pytest.approx(390.0)


def test_on_apply_and_close_commits_snapshot_only_on_success(monkeypatch):
    popup = StockSettingsPopup.__new__(StockSettingsPopup)
    old = {
        "shape": {"kind": "rectangular", "width_mm": 50, "length_mm": 40, "height_mm": 5},
        "origin": {"xy_corner": "bl", "z_reference": "top"},
        "show_stock": False,
        "simulate_cut": False,
        "voxel_resolution": "low",
        "checkpoint_level": "low",
    }
    new = dict(old)
    new["show_stock"] = True
    popup._settings_snapshot = dict(old)
    popup.read_settings_from_ui = lambda: dict(new)
    dismissed = []
    popup.dismiss = lambda: dismissed.append(True)

    class Root:
        def __init__(self):
            self.messages = []
            self.apply_result = False

        def apply_stock_settings(self, settings):
            return self.apply_result

        def show_message_popup(self, message, ok):
            self.messages.append((message, ok))

    root = Root()
    monkeypatch.setattr(
        "carveracontroller.addons.stock.ui.StockSettingsPopup.App.get_running_app",
        lambda: type("A", (), {"root": root})(),
    )

    root.apply_result = False
    popup.on_apply_and_close()
    assert popup._settings_snapshot == old
    assert dismissed == []
    assert root.messages

    root.messages.clear()
    root.apply_result = True
    popup.on_apply_and_close()
    assert popup._settings_snapshot == new
    assert dismissed == [True]
    assert root.messages == []


def test_on_apply_and_close_keeps_snapshot_when_viewer_missing(monkeypatch):
    popup = StockSettingsPopup.__new__(StockSettingsPopup)
    old = {
        "shape": {"kind": "rectangular", "width_mm": 50, "length_mm": 40, "height_mm": 5},
        "origin": {"xy_corner": "bl", "z_reference": "top"},
        "show_stock": False,
        "simulate_cut": False,
        "voxel_resolution": "low",
        "checkpoint_level": "low",
    }
    popup._settings_snapshot = dict(old)
    popup.read_settings_from_ui = lambda: {**old, "show_stock": True}
    dismissed = []
    popup.dismiss = lambda: dismissed.append(True)

    monkeypatch.setattr(
        "carveracontroller.addons.stock.ui.StockSettingsPopup.App.get_running_app",
        lambda: None,
    )

    popup.on_apply_and_close()
    assert popup._settings_snapshot == old
    assert dismissed == []
