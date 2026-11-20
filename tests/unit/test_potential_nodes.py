"""Tests for PotentialNodes: InputPotential and resolve_potential."""

import pandas as pd
import pytest

from aiflow_nodes.PotentialNodes import InputPotential, resolve_potential


def _input(**fields):
    return InputPotential._original_dataclass(**fields)


@pytest.fixture
def meam_files(tmp_path):
    library = tmp_path / "WP.library.meam"
    parameter = tmp_path / "WP.parameter.meam"
    library.write_text("library\n")
    parameter.write_text("parameter\n")
    return library, parameter


@pytest.mark.parametrize(
    "potential", ["1999--Mishin-Y--Al--LAMMPS--ipr1", pd.DataFrame({"Name": ["x"]})]
)
def test_names_and_dataframes_pass_through(potential):
    assert resolve_potential(potential) is potential


def test_a_table_name():
    assert resolve_potential(_input(potential_name="  Mishin-Al  ")) == "Mishin-Al"


def test_the_dataframe_route(meam_files):
    library, parameter = meam_files
    df = resolve_potential(
        _input(
            name="W-P_MEAM",
            filename=f"{library}, {parameter}",
            species="W, P",
            config="pair_style meam; pair_coeff * * WP.library.meam W P WP.parameter.meam W P",
        )
    )
    assert list(df.columns) == ["Name", "Filename", "Model", "Species", "Config"]
    row = df.iloc[0]
    assert (row["Name"], row["Model"]) == ("W-P_MEAM", "Custom")
    assert row["Species"] == ["W", "P"]
    assert row["Filename"] == [str(library), str(parameter)]
    # file basenames in the config become full paths, as for table potentials
    assert row["Config"] == [
        "pair_style meam\n",
        f"pair_coeff * * {library} W P {parameter} W P\n",
    ]


def test_config_lines_may_be_newline_separated_and_name_defaults_to_custom():
    df = resolve_potential(
        _input(species="Al", config="pair_style lj/cut 5.0\npair_coeff * * 0.4 2.6")
    )
    row = df.iloc[0]
    assert row["Name"] == "Custom"
    assert row["Filename"] == []
    assert row["Config"] == ["pair_style lj/cut 5.0\n", "pair_coeff * * 0.4 2.6\n"]


@pytest.mark.parametrize(
    "fields, error, match",
    [
        ({"potential_name": "x", "species": "W"}, ValueError, "not both"),
        ({}, ValueError, "is empty"),
        ({"species": "W"}, ValueError, "needs species and config"),
        ({"config": "pair_style meam"}, ValueError, "needs species and config"),
        (
            {"species": "W", "config": "pair_style meam", "filename": "/no/such/file"},
            FileNotFoundError,
            "not found",
        ),
    ],
)
def test_invalid_inputs(fields, error, match):
    with pytest.raises(error, match=match):
        resolve_potential(_input(**fields))


def test_rejects_another_type():
    with pytest.raises(TypeError, match="InputPotential"):
        resolve_potential(42)
