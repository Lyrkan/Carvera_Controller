"""Simple editor for G-code files stored on the machine."""

from __future__ import annotations

from kivy.app import App
from kivy.clock import Clock
from kivy.factory import Factory
from kivy.metrics import dp
from kivy.properties import BooleanProperty, StringProperty
from kivy.uix.modalview import ModalView


class MachineFileEditorPopup(ModalView):
    filename = StringProperty("")
    content_text = StringProperty("")
    content_focused = BooleanProperty(False)

    def open_editor(self, filename: str, content: str) -> None:
        self.filename = filename
        self.content_text = content
        self._apply_editor_content()
        self.open()

    def on_open(self) -> None:
        super().on_open()
        self._bind_editor()
        Clock.unschedule(self._sync_content_height)
        Clock.schedule_once(self._sync_content_height, 0)

    def _apply_editor_content(self) -> None:
        if "ti_filename" in self.ids:
            self.ids.ti_filename.text = self.filename
        ti = self.ids.get("ti_content")
        scroll = self.ids.get("content_scroll")
        if ti is not None:
            ti.unbind(
                text=self._sync_content_height,
                minimum_height=self._sync_content_height,
                cursor=self._ensure_cursor_visible,
            )
            ti.text = self.content_text
            ti.cursor = (0, 0)
        if scroll is not None:
            scroll.unbind(size=self._sync_content_height)
            scroll.scroll_y = 1
        self._bind_editor()
        Clock.unschedule(self._sync_content_height)
        Clock.schedule_once(self._sync_content_height, 0)

    def _bind_editor(self) -> None:
        ti = self.ids.get("ti_content")
        scroll = self.ids.get("content_scroll")
        if ti is not None:
            ti.unbind(
                text=self._sync_content_height,
                minimum_height=self._sync_content_height,
                cursor=self._ensure_cursor_visible,
            )
            ti.bind(
                text=self._sync_content_height,
                minimum_height=self._sync_content_height,
                cursor=self._ensure_cursor_visible,
            )
        if scroll is not None:
            scroll.unbind(size=self._sync_content_height)
            scroll.bind(size=self._sync_content_height)

    def _sync_content_height(self, *_args) -> None:
        ti = self.ids.get("ti_content")
        scroll = self.ids.get("content_scroll")
        if ti is None or scroll is None:
            return
        min_height = max(ti.minimum_height, scroll.height, dp(200))
        if ti.height != min_height:
            ti.height = min_height

    def _ensure_cursor_visible(self, *_args) -> None:
        ti = self.ids.get("ti_content")
        scroll = self.ids.get("content_scroll")
        if ti is None or scroll is None or not ti.focus:
            return
        span = ti.height - scroll.height
        if span <= 0:
            return
        margin = dp(8)
        caret_top = ti.cursor_pos[1] + margin
        caret_bottom = ti.cursor_pos[1] - ti.line_height - margin
        visible_bottom = scroll.scroll_y * span
        visible_top = visible_bottom + scroll.height
        if caret_bottom < visible_bottom:
            scroll.scroll_y = min(1.0, max(0.0, caret_bottom / span))
        elif caret_top > visible_top:
            scroll.scroll_y = min(1.0, max(0.0, (caret_top - scroll.height) / span))

    def _unbind_editor(self) -> None:
        Clock.unschedule(self._sync_content_height)
        ti = self.ids.get("ti_content")
        scroll = self.ids.get("content_scroll")
        if ti is not None:
            ti.unbind(
                text=self._sync_content_height,
                minimum_height=self._sync_content_height,
                cursor=self._ensure_cursor_visible,
            )
        if scroll is not None:
            scroll.unbind(size=self._sync_content_height)

    def _current_content(self) -> str:
        if "ti_content" in self.ids:
            return self.ids.ti_content.text
        return self.content_text

    def dismiss(self, *args, **kwargs):
        self._unbind_editor()
        return super().dismiss(*args, **kwargs)

    def on_dismiss(self):
        self._unbind_editor()
        return super().on_dismiss()

    def cancel(self) -> None:
        self.dismiss()

    def save(self) -> None:
        app = App.get_running_app()
        root = getattr(app, "root", None)
        if root is None:
            return
        filename = self.ids.ti_filename.text if "ti_filename" in self.ids else self.filename
        content = self._current_content()
        root.save_machine_file_editor(filename, content)


if "MachineFileEditorPopup" not in Factory.classes:
    Factory.register("MachineFileEditorPopup", cls=MachineFileEditorPopup)
