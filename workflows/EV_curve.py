from aiflow.core import Workflow
from aiflow_nodes.atomistic.calculator.data import OutputSEFS
from aiflow_nodes.atomistic.calculator.generic import ApplyEngine, CreateSEFSContainer
from aiflow_nodes.atomistic.engine.ase import GRACE
from aiflow_nodes.atomistic.structure.build import Bulk
from aiflow_nodes.controls import IterToDataFrame
from aiflow_nodes.dataframe import GetColumnFromDataFrame
from aiflow_nodes.math_utils import Linspace
from aiflow_nodes.plotting import Plot

wf = Workflow("EV_curve")

wf.GRACE = GRACE(model="GRACE-1L-MP-r6")

wf.Linspace_1 = Linspace(x_min=3, x_max=5, num_points=5)

wf.bulk = Bulk(name="Al", crystalstructure="fcc", cubic=True)

wf.IterToDataFrame = IterToDataFrame(
    node=wf.bulk,
    input_label="a",
    values=wf.Linspace_1,
    debug=False,
    executor=None,
    store=False,
)

wf.GetColumnFromDataFrame = GetColumnFromDataFrame(
    df=wf.IterToDataFrame, column_name="structure"
)

wf.CreateSEFSContainer = CreateSEFSContainer(structures=wf.GetColumnFromDataFrame)

wf.ApplyEngine = ApplyEngine(
    sefs_container=wf.CreateSEFSContainer, engine=wf.GRACE, store=False
)

wf.OutputSEFS = OutputSEFS(input=wf.ApplyEngine)

wf.Plot = Plot(y=wf.OutputSEFS.outputs.energies, x=wf.Linspace_1)
