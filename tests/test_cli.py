from splat_animator.cli import _parser, _settings_from_args


def test_direction_option_updates_spin_direction() -> None:
    arguments = _parser().parse_args(["--direction", "counter_clockwise"])

    settings = _settings_from_args(arguments)

    assert settings.spin_direction == "counter_clockwise"


def test_transparent_background_option_uses_alpha_capable_codec() -> None:
    arguments = _parser().parse_args(
        ["--transparent-background", "--codec", "vp9"]
    )

    settings = _settings_from_args(arguments)

    assert settings.transparent_background
    assert settings.premultiplied_alpha
    assert settings.codec == "vp9"


def test_bitrate_option_selects_average_bitrate_mode() -> None:
    arguments = _parser().parse_args(["--bitrate", "8.5"])

    settings = _settings_from_args(arguments)

    assert settings.bitrate_mbps == 8.5
