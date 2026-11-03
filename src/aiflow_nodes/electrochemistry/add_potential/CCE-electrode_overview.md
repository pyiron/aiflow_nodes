# VASP Electrochemistry (CCE / CDCE) Workflow

## Overview

Performs electrochemical VASP molecular dynamics simulations at constant
electrode potential using a VASP Python plugin (thermopotentiostat). The
electrode is held at a target potential during MD by adjusting the number of
electrons against a counter-charge.

Two variants live in `vasp.py` in this directory:

| node | counter-charge | plugin template |
|------|----------------|-----------------|
| `CCESetup` | Ne atoms in the structure act as the charge reservoir | `vasp_plugin-CCE.plugin` |
| `CDCESetup` | Gaussian counter-charge behind a wall potential; no Ne atoms, POTCAR untouched | `vasp_plugin-CDCE_MD.plugin` |

Both setup nodes only compute and render; nothing is written to disk until
their `plugin_data` output is passed to `CreateVaspInputResources`
(`atomistic/engine/vasp_new.py`), which adds the extra INCAR tags, applies the
POTCAR override and writes `vasp_plugin.py` into the working directory.

## CCESetup

### Inputs
| Input | Description |
|-------|-------------|
| `structure` | Full cell (`Atoms`), electrode + Ne counter-charge atoms |
| `electrode` | Electrode `Atoms`; its elements identify electrode atoms in `structure`, used to compute `d_electrode` |
| `calc` | `VaspInput` from `MergeVaspInput`; must carry MD settings (`calc.md`) |
| `potcar_lib_path` | POTCAR library used to read each element's `ZVAL` |
| `path_to_plugin` | Path to the plugin template |
| `potential` | Target electrode potential in V (`phi0` in the plugin) |
| `Q0` | Extra electrode charge (electrons), spread over the Ne atoms |
| `grid_position_frac`, `grid_roll_frac` | Where (fractional z) the potential is measured / grid roll |
| `tau` | Thermostat time constant, in MD time steps |

### Outputs
`structure` and `calc` unchanged, plus `plugin_data` (`VaspPlugin`) holding:
- the rendered plugin content (→ `vasp_plugin.py`)
- the Ne `ZVAL` override for the POTCAR, so that `Q0` is carried by the Ne atoms
- extra INCAR tags: adjusted `NELECT`, `PLUGINS/LOCAL_POTENTIAL`, `PLUGINS/OCCUPANCIES`

### Example
```python
wf.cce = CCESetup(
    structure=wf.structure.outputs.structure,
    electrode=wf.electrode.outputs.structure,
    calc=wf.calc.outputs.calc,  # MergeVaspInput(..., md=...)
    potential=0.5,
)
wf.resources = CreateVaspInputResources(
    structure=wf.cce.outputs.structure,
    calc=wf.cce.outputs.calc,
    plugin_data=wf.cce.outputs.plugin_data,
)
```

### Plugin templates
The templates are based on the original CCE implementation from
[eisenforschung/VASP-Python](https://github.com/eisenforschung/VASP-Python/tree/main),
**modified** so that all per-simulation parameters (e.g. `phi0`, `Q0`, `tau`,
cell dimensions) are `{placeholder}` format strings filled in by the setup node
via `str.format()` instead of hardcoded numbers.

### ⚠️ Be careful
- **VASP ≥ 6.5 required** — earlier versions do not support Python plugins
- `calc` must contain MD settings (`InputMDVASP` passed to `MergeVaspInput`)
- For `CCESetup` the structure **must contain Ne atoms**
- Cell **must be orthogonal** — the plugin requires `a3 ⊥ a1, a2`
- The default `path_to_plugin` is relative; pass an absolute path if the
  template cannot be found from your working directory

---

## ParsePotential

Reads the three files the plugin writes during the VASP run: `el_pot_z.dat`,
`Q.dat`, `phi.dat`. Takes the `io_bundle` from `RunVaspCalculation`; the
number of MD steps is taken from `NSW` in `extra_incar` or from
`calc.md.n_ionic_steps`.

### Outputs
| Output | Shape | Description |
|--------|-------|-------------|
| `electrostatic_potential_z_2d` | `[n_steps, nz]` | Planar-averaged potential along z per MD step |
| `charge_list` | `[n_steps]` | Electrode charge per step |
| `pot_list` | `[n_steps]` | Electrode potential per step |
