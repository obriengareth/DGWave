from copy import deepcopy
from pathlib import Path

import pytest

from dg_wave.config import load_config, parse_config, tomllib


@pytest.fixture
def data():
    with (Path(__file__).parents[1] / "stage0.toml").open("rb") as stream:
        return tomllib.load(stream)


@pytest.mark.parametrize(
    ("section", "key", "value"),
    [
        ("material", key, value)
        for key in ("density", "bulk_modulus", "shear_modulus")
        for value in (0, -1.0, float("nan"), float("inf"), True, "1000")
    ] + [
        ("discretization", "degree", value)
        for value in (-1, 1.5, True, "2")
    ] + [
        ("time", "dt", value)
        for value in (0, -0.1, float("nan"), float("inf"), True, "0.01")
    ] + [
        ("time", "steps", 0),
        ("source", "position", [0.0, 1.0]),
        ("source", "position", [0.0, 1.0, float("inf")]),
        ("source", "frequency", 0),
        ("output", "interval", 0),
        ("output", "directory", ""),
        ("mesh", "path", " "),
        ("runtime", "backend", "gpu"),
    ],
)
def test_invalid_parameter(data, section, key, value):
    data[section][key] = value
    with pytest.raises(ValueError, match=rf"{section}\.{key}"):
        parse_config(data)


@pytest.mark.parametrize("degree", [0, 1, 4])
def test_valid_config(data, degree, tmp_path):
    data["discretization"]["degree"] = degree
    config = parse_config(data, tmp_path)
    assert config.degree == degree
    assert config.mesh == tmp_path / "mesh.msh"
    assert config.output == tmp_path / "output"
    assert config.source.position == (0.0, 0.0, 0.0)
    assert config.material.density == 2500.0


@pytest.mark.parametrize("mutation", ["unknown_section", "unknown_key", "missing_key"])
def test_schema_errors(data, mutation):
    data = deepcopy(data)
    if mutation == "unknown_section":
        data["typo"] = {}
    elif mutation == "unknown_key":
        data["material"]["typo"] = 1
    else:
        del data["material"]["density"]
    with pytest.raises(ValueError, match="exactly"):
        parse_config(data)


def test_load_config_relative_paths(tmp_path):
    source = Path(__file__).parents[1] / "stage0.toml"
    target = tmp_path / "test.toml"
    target.write_text(source.read_text())
    assert load_config(target).mesh == tmp_path / "mesh.msh"


def test_invalid_toml(tmp_path):
    path = tmp_path / "invalid.toml"
    path.write_text("[broken")
    with pytest.raises(ValueError, match="invalid TOML"):
        load_config(path)


def test_optional_elastic_settings(data):
    data["boundary"] = {"types": ["absorbing", "periodic", "periodic"]}
    data["source"].update(width=0.1, delay=0.0, direction=[2.0, 0.0, 0.0])
    data["receivers"] = {"positions": [[0.2, 0.3, 0.4]]}
    config = parse_config(data)
    assert config.boundaries == ("absorbing", "periodic", "periodic")
    assert config.source.direction == (1.0, 0.0, 0.0)
    assert config.receivers == ((0.2, 0.3, 0.4),)


@pytest.mark.parametrize(("section", "key", "value"), [
    ("boundary", "types", ["periodic"]),
    ("boundary", "types", ["pml", "periodic", "periodic"]),
    ("source", "width", 0.0),
    ("source", "delay", -1.0),
    ("source", "ramp_time", -1.0),
    ("source", "direction", [0.0, 0.0, 0.0]),
    ("receivers", "positions", [[0.1, 0.2]]),
])
def test_invalid_elastic_settings(data, section, key, value):
    data.setdefault(section, {})[key] = value
    with pytest.raises(ValueError, match=rf"{section}\.{key}"):
        parse_config(data)
