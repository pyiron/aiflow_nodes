"""Checks for recovering a melting temperature from a failed calphy run.

``MeltedError`` tells you a solid melted but not at which temperature.  calphy
dumps the reversible-scaling trajectory as it sweeps and only checks for melting
*afterwards*, so on failure the frames are all on disk;
``CalphyMeltingFromTrajectory`` maps them onto temperature via T = t0/lambda and
classifies each one.  What is covered here is that mapping, the crossing
interpolation and the refusal paths -- built from synthetic run directories, so
no LAMMPS and no calphy execution.
"""

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("calphy")
pytest.importorskip("pyscal3")

from aiflow_nodes.atomistic.property.calphy import (  # noqa: E402
    CalphyMeltingFromTrajectory,
)

N_SWITCHING_STEPS = 300
N_PRINT_STEPS = 100  # -> 4 dumped frames (steps 0, 100, 200, 300)


def _write_dump(path, frames):
    """Emit *frames* as a LAMMPS text dump, the format calphy's dump writes."""
    with open(path, "w") as fh:
        for step, atoms in enumerate(frames):
            cell = np.asarray(atoms.get_cell())
            fh.write(f"ITEM: TIMESTEP\n{step * N_PRINT_STEPS}\n")
            fh.write(f"ITEM: NUMBER OF ATOMS\n{len(atoms)}\n")
            fh.write("ITEM: BOX BOUNDS pp pp pp\n")
            for axis in range(3):
                fh.write(f"0.0 {cell[axis, axis]}\n")
            fh.write("ITEM: ATOMS id type mass x y z vx vy vz\n")
            for i, (x, y, z) in enumerate(atoms.get_positions(), start=1):
                fh.write(f"{i} 1 26.98 {x} {y} {z} 0.0 0.0 0.0\n")


def _make_simfolder(
    tmp_path, frames, t_start=300.0, t_stop=900.0, n_print=N_PRINT_STEPS
):
    """Build the minimum calphy run directory this node reads."""
    simfolder = tmp_path / "ts-abc.dat-solid-300-0"
    simfolder.mkdir()

    (simfolder / "input_file.yaml").write_text(
        "calculations:\n"
        f"- temperature: [{t_start}, {t_stop}]\n"
        "  pressure: 0.0\n"
        "  reference_phase: solid\n"
        f"  n_print_steps: {n_print}\n"
        f"  n_switching_steps: {N_SWITCHING_STEPS}\n"
    )

    _write_iteration(simfolder, frames, 1, t_start, t_stop)
    return str(simfolder)


def _write_iteration(simfolder, frames, iteration, t_start=300.0, t_stop=900.0):
    """Add one ``ts.forward_<iteration>`` sweep and its dumped trajectory."""
    simfolder = Path(simfolder)
    # calphy prints one row per MD step, and lambda is a LAMMPS ramp() from 1
    # down to t_start/t_stop, so the temperature axis is t_start/lambda.
    lam = np.linspace(1.0, t_start / t_stop, N_SWITCHING_STEPS + 1)
    np.savetxt(
        simfolder / f"ts.forward_{iteration}.dat",
        np.column_stack(
            [np.zeros_like(lam), np.zeros_like(lam), np.full_like(lam, 1e4), lam]
        ),
    )
    _write_dump(simfolder / f"traj.ts.forward_{iteration}.dat", frames)


def _crystal():
    from ase.build import bulk

    return bulk("Al", cubic=True).repeat(3)


def _scrambled(seed):
    atoms = _crystal()
    rng = np.random.default_rng(seed)
    atoms.set_positions(rng.uniform(0, atoms.get_cell()[0, 0], size=(len(atoms), 3)))
    return atoms


# ── the temperature mapping ────────────────────────────────────────────────


def test_frame_temperatures_follow_t0_over_lambda(tmp_path):
    """Frame k must be placed at t0/lambda_k, lambda read at step k*n_print."""
    simfolder = _make_simfolder(tmp_path, [_crystal()] * 4)
    _, sweep = CalphyMeltingFromTrajectory(simfolder=simfolder).run()

    assert list(sweep["step"]) == [0, 100, 200, 300]
    assert np.allclose(sweep["temperature"], 300.0 / sweep["lambda"])
    # The sweep spans the requested window end to end.
    assert sweep["temperature"].iloc[0] == pytest.approx(300.0)
    assert sweep["temperature"].iloc[-1] == pytest.approx(900.0)


# ── the crossing ───────────────────────────────────────────────────────────


def test_melting_lands_between_the_bracketing_frames(tmp_path):
    """The answer must sit between the last solid and first molten frame."""
    frames = [_crystal(), _crystal(), _scrambled(0), _scrambled(1)]
    simfolder = _make_simfolder(tmp_path, frames)
    t_melt, sweep = CalphyMeltingFromTrajectory(simfolder=simfolder).run()

    assert sweep["solid_fraction"].iloc[1] > 0.7 > sweep["solid_fraction"].iloc[2]
    assert sweep["temperature"].iloc[1] <= t_melt <= sweep["temperature"].iloc[2]


def test_a_crystalline_sweep_reports_nan_rather_than_a_number(tmp_path):
    """A clean run has no melting temperature to report, and must not invent one."""
    simfolder = _make_simfolder(tmp_path, [_crystal()] * 4)
    t_melt, sweep = CalphyMeltingFromTrajectory(simfolder=simfolder).run()

    assert np.isnan(t_melt)
    assert (sweep["solid_fraction"] > 0.7).all()


def test_molten_first_frame_reports_the_start_temperature(tmp_path):
    """Nothing can be interpolated if there was never a solid; report t_start."""
    simfolder = _make_simfolder(tmp_path, [_scrambled(i) for i in range(4)])
    t_melt, _ = CalphyMeltingFromTrajectory(simfolder=simfolder).run()
    assert t_melt == pytest.approx(300.0)


def test_threshold_choice_moves_the_answer_monotonically(tmp_path):
    """A stricter solid-fraction threshold must not report a later melting."""
    frames = [_crystal(), _crystal(), _scrambled(0), _scrambled(1)]
    simfolder = _make_simfolder(tmp_path, frames)
    lenient, _ = CalphyMeltingFromTrajectory(
        simfolder=simfolder, solid_fraction_threshold=0.3
    ).run()
    strict, _ = CalphyMeltingFromTrajectory(
        simfolder=simfolder, solid_fraction_threshold=0.9
    ).run()
    assert strict <= lenient


# ── which iteration is read ────────────────────────────────────────────────


def test_the_last_iteration_written_is_the_one_read(tmp_path):
    """A run that aborted did so in whichever iteration it had reached.

    ``CalphyMeltingTemperatureSearch`` calls this on a leg that raised, from inside a
    ``except`` block that has no way to know how many iterations got through, so
    the default has to work that out from what is on disk.
    """
    simfolder = _make_simfolder(tmp_path, [_crystal()] * 4)  # iteration 1: clean
    _write_iteration(  # iteration 2: the one that died
        simfolder, [_crystal(), _crystal(), _scrambled(0), _scrambled(1)], 2
    )
    assert np.isnan(
        CalphyMeltingFromTrajectory(simfolder=simfolder, iteration=1).run()[0]
    )
    t_melt, _ = CalphyMeltingFromTrajectory(simfolder=simfolder).run()
    assert np.isfinite(t_melt)


def test_a_run_that_never_reached_the_sweep_says_so(tmp_path):
    """Melting during equilibration leaves nothing dumped to place on a T axis.

    That is a different failure from a missing directory, and the caller acts on
    it differently: no sweep means the only bound available is the temperature
    the thermostat was held at.
    """
    simfolder = tmp_path / "ts-abc.dat-solid-300-0"
    simfolder.mkdir()
    (simfolder / "input_file.yaml").write_text(
        "calculations:\n- temperature: [300.0, 900.0]\n  n_print_steps: 100\n"
    )
    with pytest.raises(FileNotFoundError, match="never reached"):
        CalphyMeltingFromTrajectory(simfolder=str(simfolder)).run()


# ── the refusal paths ──────────────────────────────────────────────────────


def test_single_temperature_run_raises(tmp_path):
    """A 'fe'-mode run has no temperature axis to place a melting event on."""
    simfolder = tmp_path / "fe-abc.dat-solid-300-0"
    simfolder.mkdir()
    (simfolder / "input_file.yaml").write_text(
        "calculations:\n- temperature: 300.0\n  n_print_steps: 100\n"
    )
    with pytest.raises(ValueError, match="single-temperature"):
        CalphyMeltingFromTrajectory(simfolder=str(simfolder)).run()


def test_no_dumped_trajectory_raises(tmp_path):
    """n_print_steps = 0 is what makes this diagnosis impossible; say so."""
    simfolder = _make_simfolder(tmp_path, [_crystal()], n_print=0)
    with pytest.raises(ValueError, match="n_print_steps"):
        CalphyMeltingFromTrajectory(simfolder=simfolder).run()


def test_missing_iteration_raises_and_lists_what_exists(tmp_path):
    """Asking for iteration 3 of a 1-iteration run should show what is there."""
    simfolder = _make_simfolder(tmp_path, [_crystal()] * 4)
    with pytest.raises(FileNotFoundError, match="ts.forward_1.dat"):
        CalphyMeltingFromTrajectory(simfolder=simfolder, iteration=3).run()


def test_non_calphy_directory_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match="input_file.yaml"):
        CalphyMeltingFromTrajectory(simfolder=str(tmp_path)).run()


def test_bad_direction_raises(tmp_path):
    simfolder = _make_simfolder(tmp_path, [_crystal()] * 4)
    with pytest.raises(ValueError, match="direction"):
        CalphyMeltingFromTrajectory(simfolder=simfolder, direction="sideways").run()
