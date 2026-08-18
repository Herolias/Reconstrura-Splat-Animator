from splat_animator.cli import _parser, _settings_from_args


def test_direction_option_updates_spin_direction() -> None:
    arguments = _parser().parse_args(["--direction", "counter_clockwise"])

    settings = _settings_from_args(arguments)

    assert settings.spin_direction == "counter_clockwise"
