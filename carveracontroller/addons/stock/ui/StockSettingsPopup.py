"""Stock settings popup — configure stock shape and cut simulation."""

from __future__ import annotations

import math

from kivy.app import App
from kivy.clock import Clock
from kivy.core.window import Window
from kivy.metrics import dp
from kivy.properties import BooleanProperty, StringProperty
from kivy.uix.modalview import ModalView

from carveracontroller.addons.facing.stock_geometry import (
    CORNER_BL,
    CORNER_BR,
    CORNER_CENTER,
    CORNER_TL,
    CORNER_TR,
)
from carveracontroller.addons.stock.simulation_quality import (
    DEFAULT_CHECKPOINT_LEVEL,
    DEFAULT_VOXEL_RESOLUTION,
)
from carveracontroller.addons.stock.stock_defaults import (
    DEFAULT_DIAMETER_MM,
    DEFAULT_HEIGHT_MM,
    DEFAULT_LENGTH_MM,
    DEFAULT_WIDTH_MM,
    default_settings,
)
from carveracontroller.addons.stock.stock_origin import Z_BOTTOM, Z_TOP, StockOrigin
from carveracontroller.addons.stock.stock_shape import CylindricalStock, RectangularStock
from carveracontroller.translation import tr


def _stock_corner_pairs():
    return [
        (tr._("Bottom-left (+X / +Y extents)"), CORNER_BL),
        (tr._("Bottom-right (-X / +Y extents)"), CORNER_BR),
        (tr._("Top-left (+X / -Y extents)"), CORNER_TL),
        (tr._("Top-right (-X / -Y extents)"), CORNER_TR),
        (tr._("Center (±½ X / ±½ Y extents)"), CORNER_CENTER),
    ]


def _z_reference_pairs():
    return [
        (tr._("Top of stock (Z0 = top)"), Z_TOP),
        (tr._("Bottom of stock (Z0 = bottom)"), Z_BOTTOM),
    ]


def _shape_kind_pairs():
    return [
        (tr._("Rectangular"), "rectangular"),
        (tr._("Cylinder"), "cylindrical"),
    ]


def _voxel_resolution_pairs():
    return [
        (tr._("Low (faster, coarser detail)"), "low"),
        (tr._("Medium"), "medium"),
        (tr._("High (slower, finer detail)"), "high"),
    ]


def _checkpoint_level_pairs():
    return [
        (tr._("Low (less memory, coarser scrubbing)"), "low"),
        (tr._("Medium"), "medium"),
        (tr._("High (more memory, smoother scrubbing)"), "high"),
    ]


def _parse_positive_float(text: str, label: str) -> float:
    t = (text or "").strip().replace(",", ".")
    if not t:
        raise ValueError(tr._("%s is required") % label)
    value = float(t)
    if not math.isfinite(value) or value <= 0:
        raise ValueError(tr._("%s must be a finite positive number") % label)
    return value


def _parse_finite_float(text: str, label: str) -> float:
    t = (text or "").strip().replace(",", ".")
    if not t:
        raise ValueError(tr._("%s is required") % label)
    value = float(t)
    if not math.isfinite(value):
        raise ValueError(tr._("%s must be a finite number") % label)
    return value


def _format_mm(value) -> str:
    """Render a millimetre value without a spurious trailing ``.0``."""
    v = float(value)
    if v == int(v):
        return str(int(v))
    return str(v)


# Cap modal height on small screens (Facing preset-list pattern).
_MAX_HEIGHT_FRAC = 0.72
_MIN_SCROLL_HEIGHT = 120


class StockSettingsPopup(ModalView):
    """Configure stock shape, origin, and cut-simulation toggles (session-only)."""

    simulation_available = BooleanProperty(True)
    simulation_hint = StringProperty("")

    def __init__(self, **kwargs):
        self._corner_pairs = None
        self._z_pairs = None
        self._shape_pairs = None
        self._voxel_resolution_pairs = None
        self._checkpoint_level_pairs = None
        self._fit_event = None
        # Snapshot seeds the form and is the Cancel/dismiss restore baseline.
        self._settings_snapshot: dict = default_settings()
        super().__init__(**kwargs)
        self.bind(on_open=self._on_open, on_dismiss=self._on_dismiss)
        self.bind(simulation_available=self._schedule_fit_popup_height)
        self.simulation_hint = tr._("Cut simulation needs tool geometry from CAM comments in the loaded G-code file.")
        self._ensure_lists()
        self._populate_spinner_choices()
        self._write_settings_to_ui(self._settings_snapshot)
        self.ids.chk_show_stock.bind(active=self._on_show_stock_active)
        self.ids.spn_shape.bind(text=self._on_shape_text)
        if "form_body" in self.ids:
            self.ids.form_body.bind(minimum_height=self._schedule_fit_popup_height)
        if "lbl_help" in self.ids:
            self.ids.lbl_help.bind(height=self._schedule_fit_popup_height)

    def _ensure_lists(self):
        if self._corner_pairs is None:
            self._corner_pairs = _stock_corner_pairs()
        if self._z_pairs is None:
            self._z_pairs = _z_reference_pairs()
        if self._shape_pairs is None:
            self._shape_pairs = _shape_kind_pairs()
        if self._voxel_resolution_pairs is None:
            self._voxel_resolution_pairs = _voxel_resolution_pairs()
        if self._checkpoint_level_pairs is None:
            self._checkpoint_level_pairs = _checkpoint_level_pairs()

    def _on_open(self, *_args):
        self._refresh_simulation_availability()
        Window.bind(size=self._schedule_fit_popup_height)
        self._schedule_fit_popup_height()

    def _on_dismiss(self, *_args):
        """Discard any unapplied edits.

        Runs whether the popup was closed via Apply (a no-op — the snapshot
        was just updated to match), Cancel, or the hardware back button —
        so the form always reflects the last *applied* settings next time
        it's opened.
        """
        Window.unbind(size=self._schedule_fit_popup_height)
        if self._fit_event is not None:
            self._fit_event.cancel()
            self._fit_event = None
        self._write_settings_to_ui(self._settings_snapshot)
        try:
            App.get_running_app().root.restore_keyboard_jog_control()
        except Exception:
            pass

    def _schedule_fit_popup_height(self, *_args) -> None:
        if self._fit_event is not None:
            self._fit_event.cancel()
        self._fit_event = Clock.schedule_once(self._fit_popup_height, 0)

    def _fit_popup_height(self, *_args) -> None:
        """Size to content, then clamp to a fraction of the window with scroll.

        ``content_box`` fills the modal (``size_hint: 1, 1``) so the ModalView
        background always covers title, form, and buttons. The ScrollView takes
        the remaining space between help and the footer.
        """
        self._fit_event = None
        ids = self.ids
        if "content_box" not in ids or "form_body" not in ids:
            return
        if "lbl_title" not in ids or "lbl_help" not in ids or "btn_row" not in ids:
            return
        content_box = ids.content_box
        pad_top, pad_bottom = content_box.padding[1], content_box.padding[3]
        # Fixed frame around the scrolling form: padding, title, help, buttons,
        # and the three spacings between those four content_box children.
        frame_h = (
            float(pad_top)
            + float(pad_bottom)
            + float(ids.lbl_title.height)
            + float(ids.lbl_help.height)
            + float(ids.btn_row.height)
            + float(content_box.spacing) * 3
        )
        form_h = max(
            float(ids.form_body.minimum_height),
            float(ids.form_body.height),
            float(dp(_MIN_SCROLL_HEIGHT)),
        )
        max_h = float(Window.height) * _MAX_HEIGHT_FRAC
        self.height = min(frame_h + form_h, max_h)

    def _on_show_stock_active(self, _instance, active: bool) -> None:
        """Cut simulation requires visible stock — clear simulate when show is off."""
        if not active and "chk_simulate" in self.ids:
            self.ids.chk_simulate.active = False

    def _on_shape_text(self, *_args) -> None:
        self._update_dimension_field_visibility()

    def _shape_kind_from_ui(self) -> str:
        text = self.ids.spn_shape.text
        for lab, val in self._shape_pairs:
            if lab == text:
                return val
        return "rectangular"

    def _update_dimension_field_visibility(self) -> None:
        """Show Width/Length for rectangular; Diameter for cylindrical.

        Length is removed from the layout (not merely disabled) so the parent
        spacing collapses and no blank row remains.
        """
        ids = self.ids
        cylindrical = self._shape_kind_from_ui() == "cylindrical"
        if "lbl_width" in ids:
            ids.lbl_width.text = tr._("Diameter (mm)") if cylindrical else tr._("Width (mm)")
        if "row_length" not in ids or "dim_form" not in ids:
            return
        row = ids.row_length
        form = ids.dim_form
        if cylindrical:
            # Remove the layout from the row so it doesn't leave a blank gap.
            if row in form.children:
                form.remove_widget(row)
            if "txt_length" in ids:
                ids.txt_length.disabled = True
        else:
            if row.parent is None:
                # Insert Length just above Height (Kivy children are reverse-order).
                height_row = ids.get("row_height")
                if height_row is not None and height_row in form.children:
                    idx = form.children.index(height_row)
                    form.add_widget(row, index=idx + 1)
                else:
                    form.add_widget(row)
            row.height = dp(40)
            row.opacity = 1
            row.disabled = False
            if "txt_length" in ids:
                ids.txt_length.disabled = False
        self._schedule_fit_popup_height()

    def reset_for_loaded_file(self) -> dict:
        """Keep applied shape/origin/quality/playback-mesh; force show/sim off; rewrite UI.

        Used when a new G-code file loads. Must not read the live form — dirty
        or invalid mid-edit fields would otherwise be committed (or wipe the
        applied baseline via a defaults fallback).
        """
        settings = dict(self._settings_snapshot)
        settings["show_stock"] = False
        settings["simulate_cut"] = False
        self._write_settings_to_ui(settings)
        self._settings_snapshot = dict(settings)
        return settings

    def _refresh_simulation_availability(self):
        app = App.get_running_app()
        tool_table = {}
        if app is not None and getattr(app, "root", None) is not None:
            viewer = getattr(app.root, "gcode_viewer", None)
            if viewer is not None:
                tool_table = getattr(viewer, "tool_table", None) or {}
        self.simulation_available = bool(tool_table)

    def _populate_spinner_choices(self):
        self._ensure_lists()
        ids = self.ids
        ids.spn_shape.values = [lab for lab, _ in self._shape_pairs]
        ids.spn_corner.values = [lab for lab, _ in self._corner_pairs]
        ids.spn_z_ref.values = [lab for lab, _ in self._z_pairs]
        ids.spn_voxel_resolution.values = [lab for lab, _ in self._voxel_resolution_pairs]
        ids.spn_checkpoint_level.values = [lab for lab, _ in self._checkpoint_level_pairs]

    def _label_for_value(self, pairs, value, fallback: str | None = None) -> str:
        for label, val in pairs:
            if val == value:
                return label
        return fallback if fallback is not None else pairs[0][0]

    def _write_settings_to_ui(self, settings: dict) -> None:
        """Populate the form fields from a settings dict (inverse of :meth:`read_settings_from_ui`)."""
        self._ensure_lists()
        ids = self.ids
        shape = settings.get("shape") or {}
        origin = settings.get("origin") or {}
        kind = shape.get("kind", "rectangular")

        ids.spn_shape.text = self._label_for_value(self._shape_pairs, kind)
        if kind == "cylindrical":
            ids.txt_width.text = _format_mm(shape.get("diameter_mm", DEFAULT_DIAMETER_MM))
            ids.txt_length.text = _format_mm(DEFAULT_LENGTH_MM)
            ids.txt_height.text = _format_mm(shape.get("height_mm", DEFAULT_HEIGHT_MM))
        else:
            ids.txt_width.text = _format_mm(shape.get("width_mm", DEFAULT_WIDTH_MM))
            ids.txt_length.text = _format_mm(shape.get("length_mm", DEFAULT_LENGTH_MM))
            ids.txt_height.text = _format_mm(shape.get("height_mm", DEFAULT_HEIGHT_MM))
        ids.spn_corner.text = self._label_for_value(self._corner_pairs, origin.get("xy_corner"))
        ids.spn_z_ref.text = self._label_for_value(self._z_pairs, origin.get("z_reference"))
        ids.txt_offset_x.text = _format_mm(origin.get("offset_x_mm", 0.0))
        ids.txt_offset_y.text = _format_mm(origin.get("offset_y_mm", 0.0))
        ids.txt_offset_z.text = _format_mm(origin.get("offset_z_mm", 0.0))
        ids.chk_show_stock.active = bool(settings.get("show_stock", False))
        ids.chk_simulate.active = bool(settings.get("simulate_cut", False))
        if "chk_mesh_while_playing" in ids:
            ids.chk_mesh_while_playing.active = bool(settings.get("mesh_while_playing", False))
        ids.spn_voxel_resolution.text = self._label_for_value(
            self._voxel_resolution_pairs,
            settings.get("voxel_resolution", DEFAULT_VOXEL_RESOLUTION),
        )
        ids.spn_checkpoint_level.text = self._label_for_value(
            self._checkpoint_level_pairs,
            settings.get("checkpoint_level", DEFAULT_CHECKPOINT_LEVEL),
        )
        self._update_dimension_field_visibility()

    def _corner_from_ui(self) -> str:
        text = self.ids.spn_corner.text
        for lab, val in self._corner_pairs:
            if lab == text:
                return val
        return CORNER_BL

    def _z_from_ui(self) -> str:
        text = self.ids.spn_z_ref.text
        for lab, val in self._z_pairs:
            if lab == text:
                return val
        return Z_TOP

    def _voxel_resolution_from_ui(self) -> str:
        text = self.ids.spn_voxel_resolution.text
        for lab, val in self._voxel_resolution_pairs:
            if lab == text:
                return val
        return DEFAULT_VOXEL_RESOLUTION

    def _checkpoint_level_from_ui(self) -> str:
        text = self.ids.spn_checkpoint_level.text
        for lab, val in self._checkpoint_level_pairs:
            if lab == text:
                return val
        return DEFAULT_CHECKPOINT_LEVEL

    def read_settings_from_ui(self) -> dict:
        height = _parse_positive_float(self.ids.txt_height.text, tr._("Height"))
        kind = self._shape_kind_from_ui()
        if kind == "cylindrical":
            diameter = _parse_positive_float(self.ids.txt_width.text, tr._("Diameter"))
            shape = CylindricalStock(diameter_mm=diameter, height_mm=height)
        else:
            width = _parse_positive_float(self.ids.txt_width.text, tr._("Width"))
            length = _parse_positive_float(self.ids.txt_length.text, tr._("Length"))
            shape = RectangularStock(width_mm=width, length_mm=length, height_mm=height)
        origin = StockOrigin(
            xy_corner=self._corner_from_ui(),
            z_reference=self._z_from_ui(),
            offset_x_mm=_parse_finite_float(self.ids.txt_offset_x.text, tr._("Offset X")),
            offset_y_mm=_parse_finite_float(self.ids.txt_offset_y.text, tr._("Offset Y")),
            offset_z_mm=_parse_finite_float(self.ids.txt_offset_z.text, tr._("Offset Z")),
        )
        simulate = bool(self.ids.chk_simulate.active) and self.simulation_available
        mesh_while_playing = False
        if "chk_mesh_while_playing" in self.ids:
            mesh_while_playing = bool(self.ids.chk_mesh_while_playing.active)
        return {
            "shape": shape.to_dict(),
            "origin": origin.to_dict(),
            "show_stock": bool(self.ids.chk_show_stock.active),
            "simulate_cut": simulate,
            "mesh_while_playing": mesh_while_playing,
            "voxel_resolution": self._voxel_resolution_from_ui(),
            "checkpoint_level": self._checkpoint_level_from_ui(),
        }

    def on_apply_and_close(self):
        try:
            settings = self.read_settings_from_ui()
        except (TypeError, ValueError) as exc:
            app = App.get_running_app()
            if app is not None and getattr(app, "root", None) is not None:
                app.root.show_message_popup(str(exc), False)
            return
        app = App.get_running_app()
        applied = False
        if app is not None and getattr(app, "root", None) is not None:
            root = app.root
            if hasattr(root, "apply_stock_settings"):
                applied = bool(root.apply_stock_settings(settings))
        if not applied:
            if app is not None and getattr(app, "root", None) is not None:
                app.root.show_message_popup(tr._("Failed to apply stock settings."), False)
            return
        self._settings_snapshot = settings
        self.dismiss()
