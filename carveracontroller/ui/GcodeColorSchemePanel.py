"""Floating color-scheme picker and legend for the G-code viewer."""

from kivy.clock import Clock
from kivy.uix.anchorlayout import AnchorLayout
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.label import Label
from kivy.uix.widget import Widget
from kivy.graphics import Color, Rectangle
from kivy.metrics import dp
from kivy.properties import ListProperty

from carveracontroller.translation import tr
from carveracontroller.GcodeViewer import (
    COLOR_SCHEME_BY_TYPE,
    COLOR_SCHEME_BY_TOOL,
    COLOR_SCHEME_BY_SPEED,
    speed_colormap_rgb,
    tool_palette_rgb,
)

LEGEND_PANEL_MAX_HEIGHT = dp(250)
HEADER_HEIGHT = dp(36)
PANEL_BOTTOM_PADDING = dp(6)
LEGEND_ROW_HEIGHT = dp(24)
LEGEND_ROW_SPACING = dp(2)
LEGEND_GRID_PAD_Y = dp(4) + dp(2)


def build_legend_entries(color_scheme, gcode_viewer, used_tools):
    """Return list of {'label': str, 'color': (r,g,b,a)} for the active scheme."""
    entries = []
    if color_scheme == COLOR_SCHEME_BY_TYPE:
        entries.append({
            'label': tr._('Rapid (G0)'),
            'color': (1.0, 0.0, 0.0, 1.0),
        })
        entries.append({
            'label': tr._('Feed (G1)'),
            'color': (0.0, 1.0, 0.0, 1.0),
        })
        return entries

    if color_scheme == COLOR_SCHEME_BY_TOOL:
        tools = sorted(t for t in set(used_tools or []) if int(t) >= 1)
        if not tools:
            tools = list(range(1, 7))
        for tool in tools:
            label = f'T{int(tool)}'
            rgb = tool_palette_rgb(tool)
            entries.append({'label': label, 'color': (*rgb, 1.0)})
        return entries

    # Speed
    feed_min = float(getattr(gcode_viewer, 'feed_min', 0.0) or 0.0)
    feed_max = float(getattr(gcode_viewer, 'feed_max', 0.0) or 0.0)
    if feed_max <= feed_min:
        feed_max = feed_min + 1.0

    for step in range(0, 11):
        t = step / 10.0
        feed = feed_min + t * (feed_max - feed_min)
        rgb = speed_colormap_rgb(t)
        entries.append({
            'label': f'{feed:.0f} mm/min',
            'color': (*rgb, 1.0),
        })
    entries.append({
        'label': tr._('Rapid (G0)'),
        'color': (1.0, 0.0, 0.0, 1.0),
    })
    return entries


class _ColorSwatch(Widget):
    color = ListProperty([1.0, 1.0, 1.0, 1.0])

    def __init__(self, **kwargs):
        super().__init__(size_hint=(None, None), size=(dp(14), dp(14)), **kwargs)
        with self.canvas:
            self._color = Color(1, 1, 1, 1)
            self._rect = Rectangle(pos=self.pos, size=self.size)
        self.bind(pos=self._sync_rect, size=self._sync_rect, color=self._sync_color)
        self._sync_color()

    def _sync_rect(self, *_args):
        self._rect.pos = self.pos
        self._rect.size = self.size

    def _sync_color(self, *_args):
        c = self.color
        a = c[3] if len(c) > 3 else 1.0
        self._color.rgba = (float(c[0]), float(c[1]), float(c[2]), float(a))


def _legend_content_height(entry_count):
    if entry_count <= 0:
        return 0
    return (
        entry_count * LEGEND_ROW_HEIGHT
        + max(0, entry_count - 1) * LEGEND_ROW_SPACING
        + LEGEND_GRID_PAD_Y
    )


def _make_legend_row(entry):
    row = BoxLayout(
        orientation='horizontal',
        size_hint_y=None,
        height=LEGEND_ROW_HEIGHT,
        spacing=dp(8),
    )
    swatch_holder = AnchorLayout(
        size_hint=(None, None),
        size=(dp(14), LEGEND_ROW_HEIGHT),
        anchor_x='left',
        anchor_y='center',
    )
    swatch_holder.add_widget(_ColorSwatch(color=list(entry['color'])))
    label = Label(
        text=entry['label'],
        halign='left',
        valign='middle',
        font_size='12sp',
        color=(0.85, 0.85, 0.85, 1),
        size_hint_x=1,
    )
    label.bind(size=lambda inst, val: setattr(inst, 'text_size', val))
    row.add_widget(swatch_holder)
    row.add_widget(label)
    return row


class GcodeColorSchemePanel(BoxLayout):
    """Dropdown plus scrollable legend for the active toolpath color scheme."""

    def refresh(self, makera_root):
        legend_box = self.ids.legend_box
        legend_box.clear_widgets()
        viewer = makera_root.gcode_viewer
        if viewer is None:
            return
        scheme = getattr(viewer, 'color_scheme', COLOR_SCHEME_BY_TYPE)
        used_tools = getattr(makera_root, 'used_tools', None)
        entries = build_legend_entries(scheme, viewer, used_tools)
        for entry in entries:
            legend_box.add_widget(_make_legend_row(entry))
        self._entry_count = len(entries)
        Clock.schedule_once(self._finish_layout, 0)

    def _finish_layout(self, _dt=None):
        legend_box = self.ids.get('legend_box')
        scroll = self.ids.get('legend_scroll')
        if legend_box is None or scroll is None:
            return

        entry_count = getattr(self, '_entry_count', len(legend_box.children))
        content_h = _legend_content_height(entry_count)
        legend_box.height = content_h
        legend_box.do_layout()

        chrome_h = HEADER_HEIGHT + PANEL_BOTTOM_PADDING
        max_scroll_h = max(LEGEND_PANEL_MAX_HEIGHT - chrome_h, 0)
        needs_scroll = content_h > max_scroll_h + 0.5
        scroll_h = max_scroll_h if needs_scroll else content_h

        scroll.size_hint_y = None
        scroll.height = scroll_h
        scroll.do_scroll_y = needs_scroll

        self.do_layout()
        scroll.scroll_y = 1
