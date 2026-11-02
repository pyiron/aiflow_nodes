# aiflow_nodes — finding and reusing nodes (LLM reference)

Companion to the aiflow workflow guide (`docs/llm_workflow_guide.md`), which covers the aiflow API itself.

## ALWAYS reuse nodes from aiflow_nodes — do not reinvent them

**Before defining any new node, search `aiflow_nodes` for an existing one.**
Reinventing nodes that already exist is the most common and most costly
mistake when constructing aiflow workflows.  `aiflow_nodes` contains
ready-to-use, tested nodes for maths, array manipulation, I/O, plotting,
control flow, atomistic structure building, DFT/MD engines, executors, and
more.  Only write a new `@as_function_node` (or dataclass node) when a
thorough search confirms that nothing equivalent exists.

**Sweeps, control flow, executors and list helpers: use `aiflow.core.stdlib`.**
`IterToDataFrame`, `iterate`, `branch`, `loop_until`, the executors, `ListOf…`,
`Range`, `Linspace` and friends exist in `aiflow_nodes` too, but the copies in
`aiflow.core.stdlib.controls` / `.executors` / `.utilities` are the ones to use.

### Package layout

```
aiflow_nodes/
  controls.py          IterToDataFrame, loop_until, iterate, pick_element,
                       ExtractList, Filter, GetAttribute, SetAttribute,
                       InputVector, Slice, Code, Print, Sleep
  math_utils.py        Add, Subtract, Multiply, Divide, Linspace, Arange,
                       Range, Sin, Cos, Mean, Sum, Array, Reshape, SVD,
                       DotProduct, Transpose, BSpline, WeightedHistogram, …
  utilities.py         Index, Prepend, Append, GetItem, ListOf,
                       ListOfStrings, ListOfIntegers, ListOfFloats,
                       ListOfBooleans
  dataframe.py         ReadDataFrame, ReadCSV, GetColumnFromDataFrame,
                       GetRowsFromDataFrame, MergeDataFrames,
                       ApplyFunctionToSeries, DisplayDataFrame
  plotting.py          Plot, MultiPlot, Scatter, Histogram, PlotDataFrame,
                       PlotDataFrameXY, ShowArray, LinearFittingCurve,
                       Subplot, Title, MergePlots
  executors.py         ThreadPoolExecutor, ProcessPoolExecutor, ForkExecutor,
                       SingleNodeExecutor, FluxClusterExecutor,
                       SlurmExecutor
  databases/
    node_hash_db.py    CreateDB, ShowTable, GetNode, GetUpstreamGraph,
                       GetHash, DeleteDB
  atomistic/
    structure/build.py       Bulk, Surface, CubicBulkCell, HighIndexSurface
    structure/transform.py   Repeat, RepeatXYZ, ApplyStrain, CreateVacancy,
                             RotateAxisAngle, FixAtoms, FixSpecies,
                             LayerShift, Stack, Rattle, RattleLoop, Stretch,
                             StretchLoop, GenerateStrainedStructures
    engine/ase.py            EMT, GRACE, Ace, LammpsEngine
    engine/lammps.py         ListPotentials, CreateLammpsStructure,
                             CreateLammpsMDInput, RunLammpsCalculation,
                             ParseLammpsOutput
    engine/quantumespresso.py  calculate_qe, converge_energy_cutoff
    calculator/ase.py        Static, Relax, Minimize, StaticEnergy,
                             MapCalculatorOnStructures,
                             GenericOptimizerSettings
    calculator/data.py       (inp/out dataclass nodes for calculator I/O)
    property/elastic.py      ComputeElasticConstants,
                             StressStrainElasticConstants, ApplyVoigtStrain
    property/bulk.py         CalculateEVCurve, FitBirchMurnaghanEOS,
                             PlotEVCurve
    property/phonons.py      GenerateSupercells, phonopy, GetTotalDos,
                             GetThermalProperties, HasImaginaryModes, …
    property/calphy.py       SolidFreeEnergyWithTemp,
                             LiquidFreeEnergyWithTemp, FindMeltingTemperature,
                             PlotSolidLiquidFreeEnergy, InputClass,
                             CalphyDiagnostics, CalphyDiagnosticsTable,
                             PlotCalphyHysteresis,
                             EstimateCalphyTemperatureRange,
                             MeltingTemperature [group],
                             CalphyMeltingTemperatureSearch,
                             SuggestTemperatureWindow,
                             CalphyMeltingFromTrajectory,
                             CalphySystem, BuildCalphySystem,
                             ApplyTemperatureWindow, …
    thermodynamics/landau/   LinePhase, TemperatureLinePhase, IdealSolution,
                             PhasesFromDataFrame, CalcPhaseDiagram,
                             TransitionTemperature, PlotConcPhaseDiagram, …
    ml_potentials/fitting/linearfit.py
                             RunLinearFit, SplitTrainingAndTesting,
                             PredictEnergiesAndForces, SavePotential, …
    assyst2/                 ElementInput, FilterSize, SpaceGroupSampling,
                             CombineStructureSets, SaveStructures
  experiment/
    youngs_modulus.py    ConvertLoadToStress, CalculateYoungsModulus,
                         PlotStressStrain
  thermodynamics/
    cp_models.py       fit_G, compute_entropy, compute_cp, LoadGibbsData, …
  scipy_nodes/
    interpolate/…      BSpline interpolation nodes
```

### Choosing a LAMMPS potential

- Only installed potentials can run. When `./POTENTIALS.md` is present it lists
  all of them (`name` | species | pair_style); Grep it for your elements. A
  potential that is not listed is not installed, however well known it is.
- Select with `GetPotential(structure=..., type_filter=..., name_filter=...)`,
  taking the `name_filter` words from a listed name. Never invent a name.
- `GetPotential` and `ListPotentials` offer only potentials that cover **every
  species in `structure`**. For an interstitial, solute or adsorbate, pass the
  structure that already contains it. A pure host returns host-only potentials,
  and LAMMPS then fails with `KeyError: '<element>'` when the extra atom appears.

### Domain note — calphy melting temperatures

**For a melting point, use `MeltingTemperature` (group node).**  It takes only a
`structure` and a `potential` string, then runs the complete two-pass protocol
automatically: an NPT superheating ramp to find a safe starting window, a first
calphy pass to locate the crossing, and a second pass centred on that estimate.
The search bounds are fixed internally (100–6000 K) and wide enough to cover
everything from light metals to refractory metals such as W — do not add
`t_start` or `t_stop` arguments, those parameters no longer exist on this node.
Check `bracketed` before trusting `T_melt`: `True` means the crossing was measured
inside the sampled window; `False` means the second pass still missed and `T_melt`
is an extrapolation (escalate to `CalphyMeltingTemperatureSearch` with a manual
window in that case).  `workflows/calphy_melt_auto_window.py` in the
aiflow_nodes repository shows a complete pipeline.

**Supercell size for calphy**: use at least `repeat_scalar=5` with `Repeat` (giving
500 atoms for an FCC conventional cell).  Smaller cells — 3×3×3 (108 atoms) —
can stochastically solidify or melt during the averaging phase and cause
`SolidifiedError` / `MeltedError` at temperatures where a larger cell would be
stable.  This is not a window-placement problem; it is a finite-size effect.

**Do not pick the temperature window yourself.**  The window is fragile in both
directions at once: reversible scaling holds the thermostat at `temperature` and
reaches `temperature_stop` by scaling the potential, so the solid leg is superheated
at the top and the liquid undercooled at the bottom.  A window that contains `T_m`
necessarily pushes both phases into metastability, and calphy does not degrade
gracefully when either gives up — it aborts with `MeltedError` or `SolidifiedError`,
saying nothing about where the window should have been.  `MeltingTemperature`
measures the window with `EstimateCalphyTemperatureRange` (an NPT ramp with the
*same* potential calphy will use) so you never guess.

**`CalphyMeltingTemperatureSearch` is the underlying search loop**, not the entry
point.  It takes a pre-built `InputClass` (with a manually chosen `temperature` and
`temperature_stop`), runs the solid and liquid legs, and turns every outcome into a
bound on `T_m`, then places the next window inside the resulting bracket — but it
cannot find a safe starting window on its own.  Use it only when `MeltingTemperature`
is not flexible enough: e.g., you already have a tight window from
`EstimateCalphyTemperatureRange` and need full control over sampling settings.
Check `converged` before using `T_melt`: `False` means the number is an
extrapolation, and `temperature_min` / `temperature_max` are then the window to
re-run in.

**A window can miss the melting point without anything failing at all.**  Above
`T_m` the liquid is the stable phase and sits at `temperature_min` quite happily,
while a defect-free periodic solid superheats past `temperature_max`; both legs
finish clean and produce two `G(T)` curves that never cross.  This is not rare —
`EstimateCalphyTemperatureRange` reports a *superheating* limit and leaves a 15 %
margin below it, which for `1995--Angelo-J-E--Ni-Al-H` is not enough and puts the
whole window above `T_m`.  `FindMeltingTemperature` returns the curves' closest
approach for this, i.e. a window endpoint, with a warning and no way to tell it
from a result.

The fix is a second pass, not a wider window (widening pushes both phases further
into metastability until one aborts).  `SuggestTemperatureWindow(temp_solid,
fe_solid, temp_liquid, fe_liquid)` takes any pair of legs and returns `T_melt`
plus `temperature_min` / `temperature_max` — named to match
`EstimateCalphyTemperatureRange`, so they drop straight into a fresh
`InputClass(temperature=…, temperature_stop=…)` — and a `bracketed` flag that is
`False` when `T_melt` is an extrapolation.  It refuses rather than guessing when
the curves are parallel or when the crossing lands more than
`max_extrapolation` (default 0.5) window-spans outside the sampled range.
`CalphyMeltingTemperatureSearch` uses it internally; wire it in explicitly when
running the manual `SolidFreeEnergyWithTemp` / `LiquidFreeEnergyWithTemp` route.

The manual route — `SolidFreeEnergyWithTemp` + `LiquidFreeEnergyWithTemp` +
`FindMeltingTemperature` — is still right when the window is already known, e.g.
for a phase diagram at fixed conditions.  `FindMeltingTemperature` intersects the
fitted `G_solid(T)` and `G_liquid(T)` curves and cannot tell a physical crossing
from an artefact, so two `InputClass` settings decide whether the answer means
anything:

- **The temperature window must bracket the expected melting point.**
  `temperature` / `temperature_stop` define the reversible-scaling sweep.  A
  window far below `T_m` still produces a crossing — of a solid against a
  "liquid" that is not a physical phase — and it is returned without a warning.
- **Keep the window narrow, too.**  Reversible scaling holds the thermostat at
  `temperature` and scales the potential by λ = `temperature`/T, so a low
  `temperature` means a small λ and a badly converged sweep.  Measured for fcc
  Al with `1999--Mishin-Y--Al`: 700→1100 K (λ = 0.64) gives a switching
  hysteresis of 2.2e-03 eV/atom, 100→1100 K (λ = 0.09) gives 5.6e-02 — 550×
  calphy's 1e-4 threshold, biasing `G` by +18 meV/atom.  Keep
  `temperature/temperature_stop ≳ 0.5`.  Do not go as wide as that bound
  allows, though: the liquid leg is anchored at `temperature`, so a window
  spanning λ = 0.5 asks for a liquid at half the melting point, which freezes
  and aborts with `SolidifiedError` (measured for Cu with
  `1985--Foiles-S-M--Ni-Cu`).  `EstimateCalphyTemperatureRange` keeps
  `temperature/temperature_stop` at `window_fraction` = 0.85 for that reason.
  It must be given an ASE-style `engine`, not a LAMMPS potential name; use
  `LammpsEngine(potential=...)` so the window is measured with the *same*
  potential calphy then runs — a window is only valid for the potential it came
  from.
- **Keep `melting_cycle=True` for the liquid leg** (the default).  It is what
  makes calphy melt the input structure at `temperature_high` and check that it
  actually melted.  With it off, the liquid free energy is computed for whatever
  configuration was handed in.
- **calphy can only issue one `pair_coeff` command**, because its
  reversible-scaling routine rewrites that single line into a
  `pair_style hybrid/scaled` pair to apply λ.  A `pair_style eam` (funcfl)
  catalogue entry instead carries one diagonal `pair_coeff i i <file>` line per
  element.  Such an entry works for a *single-element* structure — the node keeps
  the one file that structure needs and renumbers it onto type 1 — but a
  multi-element structure on a funcfl potential raises with a list of
  alternatives.  Prefer `eam/alloy`, `eam/fs` or `meam`, which map every type in
  one line.  `GetPotential(structure=...)` without filters returns the *first* catalogue
  entry, which is often an alloy potential (for Cu: `1985--Foiles-S-M--Ni-Cu`;
  for Al: a Ni-Al-H potential whose Al melts near 400 K).  Use `name_filter`
  to select a specific family:
  `GetPotential(structure=wf.bulk, type_filter="eam", name_filter="Mishin-Y--Cu", index=0)`
  If the task names a specific potential (`1999--Mishin-Y--Al--LAMMPS--ipr1`),
  pass it as a literal string to `LammpsEngine(potential=...)`.  For a variant
  that changes the element, prefer `GetPotential` with `name_filter` over
  guessing an analogous potential name string — name strings are not
  predictable from one element to another (e.g. Cu adds a `-1` suffix:
  `2001--Mishin-Y--Cu-1--LAMMPS--ipr1`).  Never guess a potential name; use
  `ListPotentials(structure=...)` or `GetPotential` with `name_filter`.

Both free-energy nodes emit a third `diagnostics` port carrying a
`CalphyDiagnostics` bundle.  Wire it into `CalphyDiagnosticsTable` (scalars and
calphy's own WARNING lines) and `PlotCalphyHysteresis` (forward vs. backward
`G(T)`) to check a run.  `rs_max_dissipation` above calphy's 1e-4 eV/atom
threshold means the switching was too fast or the structure changed — raise
`n_switching_steps` / `n_equilibration_steps`.

**When a solid leg fails with `MeltedError`, do not re-run MD to find out why.**
calphy dumps the sweep trajectory every `n_print_steps` and only checks for
melting *after* the sweep, so the frames are all on disk.  The exception carries
the run directory both as a note and as an `exc.simfolder` attribute (which is
how `CalphyMeltingTemperatureSearch` reads it from inside its own `except` block); pass
it to `CalphyMeltingFromTrajectory(simfolder=...)` and it maps each frame onto
`T = temperature/λ`, classifies it with the same pyscal q6 test calphy uses, and
returns the temperature at which the solid fraction crossed 0.7 — at no
simulation cost.  Cap `temperature_stop` below that.  Run it on *successful*
solid legs too: partial melting biases `G` without raising anything, and a large
`rs_max_dissipation` cannot say where along the sweep it happened.  It needs
`n_print_steps > 0` (the default 1000 is fine) and a two-temperature (`ts`-mode)
run — an `fe`-mode failure has no temperature axis to place the event on.

### How to search for a node

```bash
# 1. Search by function name or keyword in the source
grep -rn "def MyNode\|def.*keyword" /path/to/aiflow_nodes/ --include="*.py"

# 2. List all node definitions in a module
grep -n "^def \|^@as_function_node\|^@as_inp_dataclass\|^@as_out_dataclass" \
     aiflow_nodes/controls.py

# 3. Search by input/output type
grep -rn "Atoms\|pd.DataFrame\|np.ndarray" aiflow_nodes/ --include="*.py" -l
```

### How to import and use an existing node

```python
# Import by full module path — the node name is the class name (PascalCase)
from aiflow.core.stdlib.controls       import IterToDataFrame, loop_until
from aiflow.core.stdlib.executors      import ThreadPoolExecutor
from aiflow.core.stdlib.utilities      import Linspace
from aiflow_nodes.math_utils    import Add, Mean
from aiflow_nodes.plotting      import Plot, PlotDataFrame
from aiflow_nodes.dataframe     import ReadDataFrame, GetColumnFromDataFrame
from aiflow_nodes.atomistic.structure.build     import Bulk
from aiflow_nodes.atomistic.engine.ase          import EMT
from aiflow_nodes.atomistic.calculator.data     import OutputSEFS
from aiflow_nodes.atomistic.calculator.generic  import ApplyEngine, CreateSEFSContainer

from aiflow.core import Workflow

# Energy vs. lattice constant for fcc Al with the EMT potential.
wf = Workflow("lattice_scan")
wf.engine            = EMT()
wf.lattice_constants = Linspace(x_min=3.9, x_max=4.2, num_points=10)
# `Bulk` acts as a *template*: `a` is left unset and IterToDataFrame supplies
# one value per iteration, collecting the resulting structures in a DataFrame.
wf.bulk       = Bulk(name="Al", crystalstructure="fcc", cubic=True)
wf.scan       = IterToDataFrame(
    node=wf.bulk, input_label="a", values=wf.lattice_constants
)
wf.structures = GetColumnFromDataFrame(df=wf.scan, column_name="structure")
wf.container  = CreateSEFSContainer(structures=wf.structures)
wf.calculated = ApplyEngine(sefs_container=wf.container, engine=wf.engine)
wf.results    = OutputSEFS(input=wf.calculated)
wf.plot       = Plot(x=wf.lattice_constants, y=wf.results.outputs.energies)
```
