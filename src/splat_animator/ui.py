from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSlider,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from .gui_assets import app_icon as _app_icon
from .preview import PreviewCanvas
from .theme import COLORS


class MainWindowUiMixin:
    @staticmethod
    def _label(text: str, object_name: str | None = None, *, muted: bool = False) -> QLabel:
        label = QLabel(text)
        if object_name:
            label.setObjectName(object_name)
        if muted:
            label.setProperty("muted", True)
        label.setWordWrap(True)
        return label

    @staticmethod
    def _combo(items: list[tuple[str, object]]) -> QComboBox:
        combo = QComboBox()
        combo.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
        combo.setMinimumContentsLength(6)
        combo.setMinimumWidth(0)
        combo.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Fixed)
        for label, data in items:
            combo.addItem(label, data)
        return combo

    @staticmethod
    def _double_spin(
        minimum: float,
        maximum: float,
        value: float,
        *,
        decimals: int = 2,
        step: float = 0.1,
        suffix: str = "",
    ) -> QDoubleSpinBox:
        spin = QDoubleSpinBox()
        spin.setRange(minimum, maximum)
        spin.setDecimals(decimals)
        spin.setSingleStep(step)
        spin.setValue(value)
        spin.setSuffix(suffix)
        spin.setMinimumWidth(0)
        spin.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Fixed)
        return spin

    @staticmethod
    def _spin(
        minimum: int, maximum: int, value: int, *, step: int = 1, suffix: str = ""
    ) -> QSpinBox:
        spin = QSpinBox()
        spin.setRange(minimum, maximum)
        spin.setSingleStep(step)
        spin.setValue(value)
        spin.setSuffix(suffix)
        spin.setMinimumWidth(0)
        spin.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Fixed)
        return spin

    def _card(self, title: str, subtitle: str = "") -> tuple[QFrame, QVBoxLayout]:
        frame = QFrame()
        frame.setObjectName("card")
        layout = QVBoxLayout(frame)
        layout.setContentsMargins(17, 15, 17, 17)
        layout.setSpacing(10)
        layout.addWidget(self._label(title.upper(), "sectionLabel"))
        if subtitle:
            layout.addWidget(self._label(subtitle, "hint"))
        return frame, layout

    def _field(self, layout: QVBoxLayout, title: str, widget: QWidget, hint: str = "") -> None:
        layout.addWidget(self._label(title, "fieldLabel"))
        layout.addWidget(widget)
        if hint:
            layout.addWidget(self._label(hint, "hint"))

    @staticmethod
    def _path_row(line: QLineEdit, button: QPushButton) -> QWidget:
        row = QWidget()
        layout = QHBoxLayout(row)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(7)
        layout.addWidget(line, 1)
        layout.addWidget(button)
        return row

    def _build_ui(self) -> None:
        root = QWidget()
        root.setObjectName("root")
        self.setCentralWidget(root)
        outer = QHBoxLayout(root)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        self._build_rail(outer)

        workspace = QWidget()
        workspace_layout = QVBoxLayout(workspace)
        workspace_layout.setContentsMargins(24, 18, 18, 16)
        workspace_layout.setSpacing(14)

        header = QHBoxLayout()
        header_text = QVBoxLayout()
        header_text.setSpacing(2)
        header_text.addWidget(self._label("Splat animation", "pageTitle"))
        header.addLayout(header_text)
        header.addStretch(1)
        self.load_preset_button = QPushButton("Load preset")
        self.save_preset_button = QPushButton("Save preset")
        header.addWidget(self.load_preset_button)
        header.addWidget(self.save_preset_button)
        workspace_layout.addLayout(header)

        preview_frame = QFrame()
        preview_frame.setObjectName("previewFrame")
        preview_layout = QVBoxLayout(preview_frame)
        preview_layout.setContentsMargins(1, 1, 1, 10)
        preview_layout.setSpacing(7)
        self.preview = PreviewCanvas()
        preview_layout.addWidget(self.preview, 1)

        player = QHBoxLayout()
        player.setContentsMargins(12, 0, 12, 0)
        self.play_button = QPushButton("▶")
        self.play_button.setFixedWidth(42)
        self.play_button.setToolTip("Play or pause the preview timeline")
        self.timeline = QSlider(Qt.Horizontal)
        self.timeline.setRange(0, 10_000)
        self.time_label = QLabel("0:00.0 / 0:12.0")
        self.time_label.setFixedWidth(110)
        player.addWidget(self.play_button)
        player.addWidget(self.timeline, 1)
        player.addWidget(self.time_label)
        preview_layout.addLayout(player)
        navigation_hint = self._label(
            "Left drag to orbit | Right or middle drag to pan | Scroll to zoom | "
            "WASD and Q/E to move",
            "hint",
        )
        navigation_hint.setAlignment(Qt.AlignCenter)
        preview_layout.addWidget(navigation_hint)
        workspace_layout.addWidget(preview_frame, 1)

        status_row = QHBoxLayout()
        self.status_dot = QLabel("●")
        self.status_dot.setStyleSheet(f"color: {COLORS['blue']};")
        self.status_label = self._label("Choose a source to begin", muted=True)
        self.loaded_label = self._label("No scene loaded", "hint")
        # Preview progress text must never alter the row height: doing so
        # resizes the canvas, which itself schedules another preview.
        self.status_label.setWordWrap(False)
        self.status_label.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Fixed)
        self.loaded_label.setWordWrap(False)
        self.loaded_label.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Fixed)
        status_row.addWidget(self.status_dot)
        status_row.addWidget(self.status_label, 1)
        status_row.addWidget(self.loaded_label)
        workspace_layout.addLayout(status_row)
        outer.addWidget(workspace, 1)

        self._build_inspector(outer)

    def _build_rail(self, outer: QHBoxLayout) -> None:
        rail = QWidget()
        rail.setObjectName("rail")
        rail.setFixedWidth(218)
        layout = QVBoxLayout(rail)
        layout.setContentsMargins(14, 20, 14, 13)
        layout.setSpacing(8)

        brand = QHBoxLayout()
        icon = QLabel()
        icon.setPixmap(_app_icon(38).pixmap(38, 38))
        brand_text = QVBoxLayout()
        brand_text.setSpacing(0)
        brand_text.addWidget(self._label("Reconstrura", "appName"))
        brand_text.addWidget(self._label("SPLAT ANIMATOR", "appTag"))
        brand.addWidget(icon)
        brand.addLayout(brand_text)
        brand.addStretch(1)
        layout.addLayout(brand)
        layout.addSpacing(25)

        layout.addWidget(self._label("SECTIONS", "sectionLabel"))
        self.nav_source = self._nav_button("Source")
        self.nav_motion = self._nav_button("Animation")
        self.nav_look = self._nav_button("Camera && appearance")
        self.nav_export = self._nav_button("Export")
        for button in (self.nav_source, self.nav_motion, self.nav_look, self.nav_export):
            layout.addWidget(button)
        self.nav_source.setChecked(True)
        layout.addStretch(1)

        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        line.setStyleSheet(f"color: {COLORS['line']};")
        layout.addWidget(line)
        gpu_row = QHBoxLayout()
        gpu_row.setSpacing(6)
        self.gpu_name_label = self._label("GPU", "fieldLabel")
        self.gpu_usage_label = self._label("Detecting...", "hint")
        self.gpu_usage_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        gpu_row.addWidget(self.gpu_name_label, 1)
        gpu_row.addWidget(self.gpu_usage_label)
        layout.addLayout(gpu_row)
        self.gpu_meter = QProgressBar()
        self.gpu_meter.setObjectName("gpuMeter")
        self.gpu_meter.setRange(0, 1000)
        self.gpu_meter.setValue(0)
        self.gpu_meter.setTextVisible(False)
        self.gpu_meter.setFixedHeight(4)
        layout.addWidget(self.gpu_meter)
        outer.addWidget(rail)

    def _nav_button(self, text: str) -> QPushButton:
        button = QPushButton(text)
        button.setObjectName("navButton")
        button.setCheckable(True)
        button.setAutoExclusive(True)
        return button

    def _build_inspector(self, outer: QHBoxLayout) -> None:
        self.inspector = QScrollArea()
        self.inspector.setWidgetResizable(True)
        self.inspector.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.inspector.setFixedWidth(438)
        content = QWidget()
        content.setMinimumWidth(0)
        content.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        self.inspector_layout = QVBoxLayout(content)
        self.inspector_layout.setContentsMargins(10, 18, 17, 18)
        self.inspector_layout.setSpacing(12)
        self.inspector.setWidget(content)

        self._build_source_card()
        self._build_motion_card()
        self._build_camera_card()
        self._build_look_card()
        self._build_output_card()
        self.inspector_layout.addStretch(1)
        outer.addWidget(self.inspector)

        self.nav_source.clicked.connect(
            lambda: self.inspector.ensureWidgetVisible(self.source_card, 0, 12)
        )
        self.nav_motion.clicked.connect(
            lambda: self.inspector.ensureWidgetVisible(self.motion_card, 0, 12)
        )
        self.nav_look.clicked.connect(
            lambda: self.inspector.ensureWidgetVisible(self.camera_card, 0, 12)
        )
        self.nav_export.clicked.connect(
            lambda: self.inspector.ensureWidgetVisible(self.output_card, 0, 12)
        )

    def _build_source_card(self) -> None:
        self.source_card, layout = self._card("Source")
        self.project_line = QLineEdit()
        self.project_line.setPlaceholderText("Reconstrura or 3DGS project folder")
        self.project_browse = QPushButton("Browse")
        self.project_scan = QPushButton("Find files")
        project_row = QWidget()
        project_layout = QHBoxLayout(project_row)
        project_layout.setContentsMargins(0, 0, 0, 0)
        project_layout.setSpacing(7)
        project_layout.addWidget(self.project_line, 1)
        project_layout.addWidget(self.project_browse)
        project_layout.addWidget(self.project_scan)
        self._field(layout, "Project folder", project_row)

        self.candidate_combo = QComboBox()
        self.candidate_combo.setPlaceholderText("No files found")
        self._field(layout, "Detected file", self.candidate_combo)

        self.source_line = QLineEdit()
        self.source_line.setPlaceholderText(".ply or .splat")
        self.source_browse = QPushButton("Choose file")
        self._field(layout, "Splat file", self._path_row(self.source_line, self.source_browse))

        self.budget_spin = self._spin(0, 5_000_000, 0, step=50_000)
        self.budget_spin.setSpecialValueText("No limit")
        self._field(
            layout,
            "Point limit",
            self.budget_spin,
            "Use 0 for the full scene. A lower limit speeds up previews.",
        )
        self.source_info = self._label("No file selected", "hint")
        layout.addWidget(self.source_info)
        self.load_scene_button = QPushButton("Load file")
        self.load_scene_button.setProperty("primary", True)
        layout.addWidget(self.load_scene_button)
        self.inspector_layout.addWidget(self.source_card)

    def _build_motion_card(self) -> None:
        self.motion_card, layout = self._card("Animation")
        self.duration_spin = self._double_spin(0.5, 300.0, 12.0, suffix=" s", step=0.5)
        self._field(layout, "Duration", self.duration_spin)

        self.loop_toggle = QCheckBox("Seamless loop")
        self._field(
            layout,
            "Looping",
            self.loop_toggle,
            (
                "Adjusts the duration and spin speed to complete a full turn. "
                "Transitions return to the starting view."
            ),
        )

        self.start_representation = self._combo(
            [("Gaussian splat", "splat"), ("Point cloud", "point")]
        )
        self._field(layout, "Start with", self.start_representation)

        self.trip_mode = self._combo(
            [
                ("Return to start", True),
                ("One-way", False),
                ("Off", None),
            ]
        )
        self._field(layout, "Transition", self.trip_mode)

        timing_row = QWidget()
        timing_layout = QHBoxLayout(timing_row)
        timing_layout.setContentsMargins(0, 0, 0, 0)
        timing_layout.setSpacing(7)
        self.transition_start_spin = self._double_spin(0.0, 300.0, 2.0, suffix=" s")
        self.transition_duration_spin = self._double_spin(0.1, 120.0, 2.2, suffix=" s")
        timing_layout.addWidget(self.transition_start_spin)
        timing_layout.addWidget(self.transition_duration_spin)
        self._field(
            layout,
            "Start time / duration",
            timing_row,
        )

        self.return_hold_spin = self._double_spin(0.0, 120.0, 2.6, suffix=" s")
        self._field(layout, "Hold before return", self.return_hold_spin)

        self.easing_combo = self._combo(
            [("Ease in/out", "cinematic"), ("Smooth", "smooth"), ("Linear", "linear")]
        )
        self._field(layout, "Easing", self.easing_combo)

        self.transition_effect = self._combo(
            [
                ("Sweep", "sweep"),
                ("Radial", "radial"),
                ("Wave", "wave"),
                ("Spiral", "spiral"),
                ("Dissolve", "dissolve"),
            ]
        )
        self._field(layout, "Effect", self.transition_effect)

        scan_row = QWidget()
        scan_layout = QHBoxLayout(scan_row)
        scan_layout.setContentsMargins(0, 0, 0, 0)
        scan_layout.setSpacing(7)
        self.scan_direction = self._combo(
            [
                ("Top to bottom", "top_to_bottom"),
                ("Bottom to top", "bottom_to_top"),
            ]
        )
        self.up_axis = self._combo(
            [
                ("-Y up", "-y"),
                ("Y up", "y"),
                ("Z up", "z"),
                ("-Z up", "-z"),
                ("X up", "x"),
                ("-X up", "-x"),
            ]
        )
        scan_layout.addWidget(self.scan_direction, 2)
        scan_layout.addWidget(self.up_axis, 1)
        self._field(layout, "Direction / up axis", scan_row)

        self.scan_feather_spin = self._double_spin(0.005, 0.35, 0.055, decimals=3, step=0.01)
        self._field(layout, "Edge softness", self.scan_feather_spin)
        self.timeline_summary = self._label("", "hint")
        layout.addWidget(self.timeline_summary)
        self.inspector_layout.addWidget(self.motion_card)

    def _build_camera_card(self) -> None:
        self.camera_card, layout = self._card("Camera")
        self.rotation_mode = self._combo(
            [("Orbit camera", "camera_orbit"), ("Spin object", "object_spin")]
        )
        self._field(layout, "Rotation mode", self.rotation_mode)

        spin_row = QWidget()
        spin_layout = QHBoxLayout(spin_row)
        spin_layout.setContentsMargins(0, 0, 0, 0)
        spin_layout.setSpacing(7)
        self.spin_speed = self._double_spin(0.0, 720.0, 30.0, suffix="°/s", step=5.0)
        self.spin_direction = self._combo(
            [("Clockwise", "clockwise"), ("Counterclockwise", "counter_clockwise")]
        )
        spin_layout.addWidget(self.spin_speed)
        spin_layout.addWidget(self.spin_direction)
        self._field(layout, "Spin speed / direction", spin_row)

        angle_row = QWidget()
        angle_layout = QHBoxLayout(angle_row)
        angle_layout.setContentsMargins(0, 0, 0, 0)
        angle_layout.setSpacing(7)
        self.start_angle = self._double_spin(-360.0, 360.0, 0.0, suffix="°", step=5.0)
        self.elevation = self._double_spin(-80.0, 80.0, 12.0, suffix="°", step=2.0)
        angle_layout.addWidget(self.start_angle)
        angle_layout.addWidget(self.elevation)
        self._field(layout, "Start angle / elevation", angle_row)

        camera_row = QWidget()
        camera_layout = QHBoxLayout(camera_row)
        camera_layout.setContentsMargins(0, 0, 0, 0)
        camera_layout.setSpacing(7)
        self.distance_scale = self._double_spin(
            0.0001, 100.0, 2.65, suffix="×", decimals=6, step=0.1
        )
        self.fov = self._double_spin(1.0, 140.0, 48.0, suffix="°", step=2.0)
        camera_layout.addWidget(self.distance_scale)
        camera_layout.addWidget(self.fov)
        self._field(layout, "Distance / field of view", camera_row)

        pan_row = QWidget()
        pan_layout = QHBoxLayout(pan_row)
        pan_layout.setContentsMargins(0, 0, 0, 0)
        pan_layout.setSpacing(7)
        self.camera_target_x = self._double_spin(
            -1000.0, 1000.0, 0.0, suffix="×", decimals=8, step=0.05
        )
        self.camera_target_y = self._double_spin(
            -1000.0, 1000.0, 0.0, suffix="×", decimals=8, step=0.05
        )
        self.camera_target_z = self._double_spin(
            -1000.0, 1000.0, 0.0, suffix="×", decimals=8, step=0.05
        )
        pan_layout.addWidget(self.camera_target_x, 1)
        pan_layout.addWidget(self.camera_target_y, 1)
        pan_layout.addWidget(self.camera_target_z, 1)
        self._field(
            layout,
            "Orbit target X / Y / Z",
            pan_row,
            "Offsets use the scene radius as one unit.",
        )

        self.rotation_center_mode = self._combo(
            [
                ("Automatic", "automatic"),
                ("Scene center", "scene"),
                ("Orbit target", "target"),
                ("Custom", "custom"),
            ]
        )
        self._field(layout, "Rotation center", self.rotation_center_mode)

        rotation_center_row = QWidget()
        rotation_center_layout = QHBoxLayout(rotation_center_row)
        rotation_center_layout.setContentsMargins(0, 0, 0, 0)
        rotation_center_layout.setSpacing(7)
        self.rotation_center_x = self._double_spin(
            -1000.0, 1000.0, 0.0, suffix="×", decimals=8, step=0.05
        )
        self.rotation_center_y = self._double_spin(
            -1000.0, 1000.0, 0.0, suffix="×", decimals=8, step=0.05
        )
        self.rotation_center_z = self._double_spin(
            -1000.0, 1000.0, 0.0, suffix="×", decimals=8, step=0.05
        )
        rotation_center_layout.addWidget(self.rotation_center_x, 1)
        rotation_center_layout.addWidget(self.rotation_center_y, 1)
        rotation_center_layout.addWidget(self.rotation_center_z, 1)
        self._field(layout, "Custom center X / Y / Z", rotation_center_row)

        camera_actions = QWidget()
        camera_actions_layout = QHBoxLayout(camera_actions)
        camera_actions_layout.setContentsMargins(0, 0, 0, 0)
        camera_actions_layout.setSpacing(7)
        self.opening_frame_button = QPushButton("Go to start")
        self.reset_camera_button = QPushButton("Reset view")
        camera_actions_layout.addWidget(self.opening_frame_button)
        camera_actions_layout.addWidget(self.reset_camera_button)
        layout.addWidget(camera_actions)
        self.inspector_layout.addWidget(self.camera_card)

    def _build_look_card(self) -> None:
        self.look_card, layout = self._card("Appearance")
        point_row = QWidget()
        point_layout = QHBoxLayout(point_row)
        point_layout.setContentsMargins(0, 0, 0, 0)
        point_layout.setSpacing(7)
        self.point_radius = self._double_spin(0.25, 12.0, 0.75, suffix=" px", step=0.1)
        self.point_opacity = self._double_spin(0.01, 1.0, 0.55, step=0.05)
        point_layout.addWidget(self.point_radius)
        point_layout.addWidget(self.point_opacity)
        self._field(
            layout,
            "Point size / opacity",
            point_row,
            "Point size is based on 1080p and scales with the output resolution.",
        )

        splat_row = QWidget()
        splat_layout = QHBoxLayout(splat_row)
        splat_layout.setContentsMargins(0, 0, 0, 0)
        splat_layout.setSpacing(7)
        self.splat_scale = self._double_spin(0.1, 5.0, 1.0, suffix="×", step=0.05)
        self.splat_opacity = self._double_spin(0.01, 2.0, 1.0, suffix="×", step=0.05)
        splat_layout.addWidget(self.splat_scale)
        splat_layout.addWidget(self.splat_opacity)
        self._field(layout, "Splat scale / opacity", splat_row)

        self.exposure = self._double_spin(-4.0, 4.0, 0.0, suffix=" EV", step=0.1)
        self._field(layout, "Exposure", self.exposure)

        background_row = QWidget()
        background_layout = QHBoxLayout(background_row)
        background_layout.setContentsMargins(0, 0, 0, 0)
        background_layout.setSpacing(7)
        self.background_button = QPushButton("Choose color")
        self.gradient = self._double_spin(0.0, 0.75, 0.16, step=0.05)
        background_layout.addWidget(self.background_button, 1)
        background_layout.addWidget(self.gradient)
        self._field(layout, "Background color / gradient", background_row)
        self.transparent_background = QCheckBox("Transparent background")
        self.transparent_background.setToolTip(
            "Exports an alpha channel with VP9 WebM. "
            "Enabling this selects VP9 automatically. "
            "VLC, Dragon Player, and some other players ignore WebM alpha and may "
            "show fuzzy edges; use an alpha-aware editor or compositor to verify it."
        )
        layout.addWidget(self.transparent_background)
        self.premultiplied_alpha = QCheckBox("Black fallback for alpha-blind players")
        self.premultiplied_alpha.setToolTip(
            "Switches on with Transparent background so VLC and similar players "
            "display clean edges on black. Trade-off: it stores premultiplied RGB, "
            "so editors expecting straight alpha may multiply it again and produce "
            "dark edges. Disable this option for standard alpha compositing."
        )
        self.premultiplied_alpha.setEnabled(False)
        layout.addWidget(self.premultiplied_alpha)
        self.inspector_layout.addWidget(self.look_card)

    def _build_output_card(self) -> None:
        self.output_card, layout = self._card("Export")
        self.resolution_preset = self._combo(
            [
                ("Full HD (1920 × 1080)", (1920, 1080)),
                ("4K UHD (3840 × 2160)", (3840, 2160)),
                ("Square (1080 × 1080)", (1080, 1080)),
                ("Vertical (1080 × 1920)", (1080, 1920)),
                ("Vertical 4K (2160 × 3840)", (2160, 3840)),
                ("Custom", None),
            ]
        )
        self._field(layout, "Resolution preset", self.resolution_preset)

        resolution_row = QWidget()
        resolution_layout = QHBoxLayout(resolution_row)
        resolution_layout.setContentsMargins(0, 0, 0, 0)
        resolution_layout.setSpacing(7)
        self.width_spin = self._spin(16, 7680, 1920, step=2)
        self.height_spin = self._spin(16, 7680, 1080, step=2)
        self.fps_spin = self._spin(1, 240, 30, suffix=" fps")
        resolution_layout.addWidget(self.width_spin)
        resolution_layout.addWidget(self.height_spin)
        resolution_layout.addWidget(self.fps_spin)
        self._field(layout, "Width / height / frame rate", resolution_row)

        codec_row = QWidget()
        codec_layout = QHBoxLayout(codec_row)
        codec_layout.setContentsMargins(0, 0, 0, 0)
        codec_layout.setSpacing(7)
        self.codec_combo = self._combo(
            [
                ("H.264 (MP4)", "h264"),
                ("H.265 (MP4)", "h265"),
                ("ProRes 422 HQ (MOV)", "prores"),
                ("VP9 (WebM)", "vp9"),
            ]
        )
        self.quality_spin = self._spin(0, 51, 18)
        codec_layout.addWidget(self.codec_combo, 2)
        codec_layout.addWidget(self.quality_spin, 1)
        self._field(
            layout,
            "Format / quality (CRF)",
            codec_row,
            "Lower CRF means higher quality and larger files.",
        )

        self.bitrate_spin = self._double_spin(
            0.0,
            500.0,
            0.0,
            decimals=1,
            step=1.0,
            suffix=" Mb/s",
        )
        self.bitrate_spin.setSpecialValueText("Use CRF")
        self._field(
            layout,
            "Target bitrate",
            self.bitrate_spin,
            "Set a bitrate for predictable file size, or use CRF for automatic quality.",
        )

        self.output_line = QLineEdit()
        self.output_line.setPlaceholderText("promo.mp4")
        self.output_browse = QPushButton("Browse")
        self._field(layout, "Output file", self._path_row(self.output_line, self.output_browse))

        self.render_estimate = self._label("360 frames", "hint")
        layout.addWidget(self.render_estimate)
        self.render_button = QPushButton("Render video")
        self.render_button.setProperty("primary", True)
        self.render_button.setEnabled(False)
        layout.addWidget(self.render_button)
        self.cancel_button = QPushButton("Cancel render")
        self.cancel_button.setEnabled(False)
        layout.addWidget(self.cancel_button)
        self.render_progress = QProgressBar()
        self.render_progress.setRange(0, 100)
        self.render_progress.setValue(0)
        self.render_progress.setTextVisible(False)
        layout.addWidget(self.render_progress)
        self.render_status = self._label("Ready", "hint")
        layout.addWidget(self.render_status)
        self.inspector_layout.addWidget(self.output_card)

    def _connect_changes(self) -> None:
        self.project_browse.clicked.connect(self._choose_project)
        self.project_scan.clicked.connect(lambda: self._scan_project(False))
        self.project_line.returnPressed.connect(lambda: self._scan_project(False))
        self.source_browse.clicked.connect(self._choose_source)
        self.candidate_combo.currentIndexChanged.connect(self._candidate_changed)
        self.source_line.editingFinished.connect(self._inspect_manual_source)
        self.load_scene_button.clicked.connect(self._load_scene)
        self.output_browse.clicked.connect(self._choose_output)
        self.background_button.clicked.connect(self._choose_background)
        self.transparent_background.toggled.connect(self._transparency_changed)
        self.premultiplied_alpha.toggled.connect(self._settings_changed)
        self.render_button.clicked.connect(self._start_render)
        self.cancel_button.clicked.connect(self._cancel_render)
        self.load_preset_button.clicked.connect(self._load_preset)
        self.save_preset_button.clicked.connect(self._save_preset)
        self.play_button.clicked.connect(self._toggle_play)
        self.timeline.valueChanged.connect(self._timeline_changed)
        self.timeline.sliderPressed.connect(self._preview_interaction_started)
        self.timeline.sliderReleased.connect(self._preview_interaction_settled)
        self.preview.dragged.connect(self._preview_dragged)
        self.preview.panned.connect(self._preview_panned)
        self.preview.zoomed.connect(self._preview_zoomed)
        self.preview.navigated.connect(self._preview_navigated)
        self.preview.resetRequested.connect(self._reset_camera)
        self.preview.resized.connect(self._preview_resized)
        self.opening_frame_button.clicked.connect(self._show_opening_frame)
        self.reset_camera_button.clicked.connect(self._reset_camera)
        self.resolution_preset.currentIndexChanged.connect(self._resolution_preset_changed)
        self.codec_combo.currentIndexChanged.connect(self._codec_changed)
        self.bitrate_spin.valueChanged.connect(self._bitrate_changed)
        self.trip_mode.currentIndexChanged.connect(self._trip_mode_changed)
        self.loop_toggle.toggled.connect(self._loop_toggled)
        self.rotation_center_mode.currentIndexChanged.connect(self._rotation_center_mode_changed)

        change_widgets = (
            self.duration_spin,
            self.transition_start_spin,
            self.transition_duration_spin,
            self.return_hold_spin,
            self.scan_feather_spin,
            self.spin_speed,
            self.start_angle,
            self.elevation,
            self.distance_scale,
            self.camera_target_x,
            self.camera_target_y,
            self.camera_target_z,
            self.rotation_center_x,
            self.rotation_center_y,
            self.rotation_center_z,
            self.fov,
            self.point_radius,
            self.point_opacity,
            self.splat_scale,
            self.splat_opacity,
            self.exposure,
            self.gradient,
            self.width_spin,
            self.height_spin,
            self.fps_spin,
            self.quality_spin,
        )
        for widget in change_widgets:
            widget.valueChanged.connect(self._settings_changed)
        combo_widgets = (
            self.start_representation,
            self.trip_mode,
            self.easing_combo,
            self.transition_effect,
            self.scan_direction,
            self.up_axis,
            self.rotation_mode,
            self.spin_direction,
            self.codec_combo,
        )
        for widget in combo_widgets:
            widget.currentIndexChanged.connect(self._settings_changed)
