from dataclasses import replace

import pytest

from splat_animator.models import AnimationSettings, RepresentationFrame


def test_round_trip_timeline_and_continuous_spin() -> None:
    settings = AnimationSettings(
        transition_start=2.0,
        transition_duration=2.0,
        return_hold=1.0,
        spin_speed=30.0,
        start_angle=10.0,
    )
    assert settings.representation_at(0.0) == RepresentationFrame(1.0, 1.0, 0.0)
    assert settings.representation_at(3.0) == RepresentationFrame(1.0, 0.0, 0.5)
    assert settings.representation_at(4.5) == RepresentationFrame(0.0, 0.0, 0.0)
    assert settings.representation_at(6.0) == RepresentationFrame(0.0, 1.0, 0.5)
    assert settings.representation_at(7.1) == RepresentationFrame(1.0, 1.0, 0.0)
    assert settings.angle_at(2.0) == pytest.approx(-50.0)


def test_one_way_point_to_splat_and_json_round_trip(tmp_path) -> None:
    settings = AnimationSettings(
        start_representation="point",
        round_trip=False,
        easing="linear",
        transition_start=1.0,
        transition_duration=4.0,
        background="#123456",
        camera_target_x=0.35,
        camera_target_y=-0.2,
        camera_target_z=0.1,
        rotation_center_mode="custom",
        rotation_center_x=0.2,
        rotation_center_y=-0.4,
        rotation_center_z=0.6,
        transition_effect="wave",
        transparent_background=True,
        premultiplied_alpha=True,
        codec="vp9",
        bitrate_mbps=12.5,
    )
    assert settings.representation_at(0.5) == RepresentationFrame(0.0, 0.0, 0.0)
    assert settings.representation_at(3.0) == RepresentationFrame(0.0, 1.0, 0.5)
    assert settings.representation_at(6.0) == RepresentationFrame(1.0, 1.0, 0.0)
    path = tmp_path / "preset.json"
    settings.to_json(path)
    assert AnimationSettings.from_json(path) == settings


def test_no_transformation_holds_selected_representation_and_loops_rotation() -> None:
    settings = AnimationSettings(
        duration=10.0,
        fps=30,
        spin_speed=36.0,
        start_representation="point",
        transformation_enabled=False,
        round_trip=False,
        seamless_loop=True,
        transition_start=100.0,
    )

    assert settings.representation_at(0.0) == RepresentationFrame(0.0, 0.0, 0.0)
    assert settings.representation_at(500.0) == RepresentationFrame(0.0, 0.0, 0.0)
    assert settings.transition_end == 0.0

    resolved = settings.resolved_for_loop()
    assert not resolved.transformation_enabled
    assert not resolved.round_trip
    assert resolved.duration == pytest.approx(10.0)
    assert resolved.spin_speed == pytest.approx(36.0)


def test_validation_rejects_odd_video_dimensions() -> None:
    with pytest.raises(ValueError, match="even"):
        replace(AnimationSettings(), width=1919).validate()
    with pytest.raises(ValueError, match="distance"):
        replace(AnimationSettings(), distance_scale=0.0).validate()
    with pytest.raises(ValueError, match="transition effect"):
        replace(AnimationSettings(), transition_effect="teleport").validate()
    with pytest.raises(ValueError, match="rotation center mode"):
        replace(AnimationSettings(), rotation_center_mode="selection").validate()
    with pytest.raises(ValueError, match="require VP9"):
        replace(AnimationSettings(), transparent_background=True).validate()
    with pytest.raises(ValueError, match="require VP9"):
        replace(
            AnimationSettings(),
            transparent_background=True,
            codec="prores",
        ).validate()
    replace(
        AnimationSettings(),
        transparent_background=True,
        codec="vp9",
    ).validate()
    with pytest.raises(ValueError, match="requires a transparent background"):
        replace(AnimationSettings(), premultiplied_alpha=True).validate()
    with pytest.raises(ValueError, match="zero or greater"):
        replace(AnimationSettings(), bitrate_mbps=-1.0).validate()
    with pytest.raises(ValueError, match="zero or greater"):
        replace(AnimationSettings(), bitrate_mbps=True).validate()
    with pytest.raises(ValueError, match="not supported for ProRes"):
        replace(AnimationSettings(), codec="prores", bitrate_mbps=20.0).validate()
    assert not AnimationSettings().premultiplied_alpha


def test_validation_rejects_non_boolean_seamless_loop(tmp_path) -> None:
    path = tmp_path / "invalid-loop.json"
    path.write_text('{"seamless_loop": "false"}', encoding="utf-8")

    with pytest.raises(ValueError, match="Seamless loop must be true or false"):
        AnimationSettings.from_json(path)


def test_seamless_loop_resolves_to_complete_frame_aligned_cycle() -> None:
    selected = AnimationSettings(
        duration=10.0,
        fps=30,
        spin_speed=30.0,
        round_trip=False,
        seamless_loop=True,
    )
    resolved = selected.resolved_for_loop()

    assert resolved.round_trip
    assert resolved.duration * resolved.fps == pytest.approx(resolved.frame_count)
    assert resolved.spin_speed * resolved.duration / 360.0 == pytest.approx(1.0)
    assert resolved.transition_end <= resolved.duration
    assert resolved.representation_at(0.0) == resolved.representation_at(resolved.duration)
    assert selected.duration == 10.0
    assert not selected.round_trip


def test_seamless_loop_with_no_rotation_only_extends_for_round_trip() -> None:
    resolved = AnimationSettings(
        duration=7.0,
        spin_speed=0.0,
        seamless_loop=True,
    ).resolved_for_loop()
    assert resolved.duration == pytest.approx(resolved.transition_end)
    assert resolved.spin_speed == 0.0
