import re
from types import SimpleNamespace

import flet as ft


SCALE = 0.9


def S(v):
    """Scale a dimension literal by the global UI SCALE factor."""
    return round(v * SCALE)


FONT_SIZE = S(14)
LABEL_ALIGN_WIDTH = 0


def set_label_align_width(value):
    global LABEL_ALIGN_WIDTH
    LABEL_ALIGN_WIDTH = value


class CompactTextField(ft.Container):
    """Compact text input, 32px height, visually consistent with CompactDropdown."""

    def __init__(self, value="", width=S(100), read_only=False, on_change=None, tooltip=None, filter=None, max_length=None):
        super().__init__()
        self.height = S(32)
        self.width = width if width else None
        self.padding = 0
        self.bgcolor = None
        self.border = None
        self.border_radius = 0
        self._read_only = read_only
        self._on_change = on_change
        self._value = value
        self._label = ft.Text(value or "", size=FONT_SIZE)
        self._committed = False
        self._tooltip = tooltip or ""
        self._filter = filter
        self._max_length = max_length
        self._build_display()

    def _build_display(self):
        self.content = ft.Container(
            height=S(32), padding=ft.Padding(S(8), 0, S(8), 0),
            border=ft.Border(ft.BorderSide(1, ft.Colors.OUTLINE), ft.BorderSide(1, ft.Colors.OUTLINE), ft.BorderSide(1, ft.Colors.OUTLINE), ft.BorderSide(1, ft.Colors.OUTLINE)),
            border_radius=4,
            tooltip=self._tooltip,
            on_click=None if self._read_only else self._on_click,
            content=ft.Row([self._label], spacing=2, vertical_alignment=ft.CrossAxisAlignment.CENTER),
        )

    def _on_click(self, e):
        self._committed = False
        tf = ft.TextField(
            value=self._value, text_size=FONT_SIZE, dense=True,
            filled=False, border=ft.InputBorder.NONE,
            content_padding=ft.Padding(0, 0, 0, 0), height=S(28),
            autofocus=True, on_submit=self._on_submit, on_blur=self._on_submit,
            max_length=self._max_length,
            input_filter=ft.InputFilter(regex_string=self._filter, allow=True) if self._filter else None,
        )
        self.content = ft.Container(
            height=S(32), padding=ft.Padding(S(4), 0, S(4), 0),
            border=ft.Border(ft.BorderSide(1, ft.Colors.OUTLINE), ft.BorderSide(1, ft.Colors.OUTLINE), ft.BorderSide(1, ft.Colors.OUTLINE), ft.BorderSide(1, ft.Colors.OUTLINE)),
            border_radius=4, content=tf,
        )
        self.update()

    def set_tooltip(self, text):
        self._tooltip = text

    @property
    def value(self):
        return self._value

    @value.setter
    def value(self, val):
        self._value = val
        self._label.value = val
        try:
            self._label.update()
        except RuntimeError:
            pass

    def _on_submit(self, e):
        if self._committed:
            return
        self._committed = True
        raw = e.control.value
        if self._filter:
            raw = ''.join(c for c in raw if re.match(self._filter, c))
        self._value = raw
        if self._on_change:
            self._on_change(SimpleNamespace(control=SimpleNamespace(value=self._value)))
        self._label.value = self._value
        self._build_display()
        self.update()


class CompactDropdown(ft.Container):
    """Compact dropdown, PopupMenuButton with controllable width."""

    _instances = None

    def __init__(self, options=None, value="", on_select=None, expand=False,
                 dyna_width=None, width=None, min_width=None, max_width=None,
                 tooltip=None, _instances_list=None):
        super().__init__()
        self._options = options or []
        self._on_select_cb = on_select
        self._dyna = dyna_width
        self._fixed = width
        self._min = min_width or 0
        self._max = max_width or 0
        self._tooltip = tooltip
        self.height = S(32)
        self.padding = 0
        self.bgcolor = None
        self.border = None
        self.border_radius = 0
        self.expand_loose = expand

        self._label = ft.Text(value or "", size=FONT_SIZE)
        self._build_menu()
        self._apply_width()
        if _instances_list is not None:
            _instances_list.append(self)
        elif CompactDropdown._instances is not None:
            CompactDropdown._instances.append(self)

    def reapply_width(self):
        self._apply_width()

    def _calc_auto_width(self):
        txt = self._label.value or ""
        if not txt:
            return S(100)
        w = sum(FONT_SIZE * (1.2 if ord(c) > 127 else 0.6) for c in txt)
        return int(w) + S(34)

    def _apply_width(self):
        if self._dyna:
            self.width = LABEL_ALIGN_WIDTH or self._calc_auto_width()
        elif self._fixed is not None:
            self.width = self._fixed
        else:
            auto = self._calc_auto_width()
            if self._min and auto < self._min:
                self.width = self._min
            elif self._max and auto > self._max:
                self.width = self._max
            else:
                self.width = None

    def _build_menu(self):
        def on_item_click(e):
            val = e.control.data
            self._value = val
            self._label.value = val
            self._apply_width()
            try:
                self._label.update()
                self.update()
            except RuntimeError:
                pass
            if self._on_select_cb:
                ev = SimpleNamespace(control=SimpleNamespace(value=val))
                self._on_select_cb(ev)

        items = [
            ft.PopupMenuItem(content=ft.Container(ft.Text(o, size=FONT_SIZE), padding=ft.Padding(8, 0, 8, 0)), data=o, height=S(32), padding=0, on_click=on_item_click)
            for o in self._options
        ]

        has_limit = self._min or self._max
        align = ft.MainAxisAlignment.SPACE_BETWEEN if (self._fixed is not None or self._dyna or has_limit) else ft.MainAxisAlignment.START

        self.content = ft.PopupMenuButton(
            items=items,
            menu_position=ft.PopupMenuPosition.UNDER,
            enable_feedback=False,
            padding=0, menu_padding=0,
            tooltip=self._tooltip or "",
            content=ft.Container(
                height=S(32),
                padding=ft.Padding(S(8), 0, S(8), 0),
                tooltip=self._tooltip or "",
                border=ft.Border(
                    ft.BorderSide(1, ft.Colors.OUTLINE),
                    ft.BorderSide(1, ft.Colors.OUTLINE),
                    ft.BorderSide(1, ft.Colors.OUTLINE),
                    ft.BorderSide(1, ft.Colors.OUTLINE),
                ),
                border_radius=4,
                content=ft.Row([
                    self._label,
                    ft.Icon(ft.Icons.ARROW_DROP_DOWN, size=S(16)),
                ], spacing=2, alignment=align,
                   vertical_alignment=ft.CrossAxisAlignment.CENTER),
            ),
        )

    def set_tooltip(self, text):
        self._tooltip = text
        try:
            self._build_menu()
            self.update()
        except RuntimeError:
            pass

    @property
    def value(self):
        return self._label.value

    @value.setter
    def value(self, val):
        self._label.value = val
        self._apply_width()
        try:
            self._build_menu()
            self._label.update()
            self.update()
        except RuntimeError:
            pass

    @property
    def options(self):
        return self._options

    @options.setter
    def options(self, opts):
        self._options = opts
        try:
            self._build_menu()
            self.update()
        except RuntimeError:
            pass
