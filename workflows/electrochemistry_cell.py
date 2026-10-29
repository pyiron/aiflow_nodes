from aiflow.core import Workflow
from aiflow_nodes.atomistic.structure.build import Surface
from aiflow_nodes.atomistic.structure.view import Plot3d
from aiflow_nodes.electrochemistry.structure.build import AddNeonLayer, AddWaterFilm

wf = Workflow("electrochemistry_cell")

wf.Surface = Surface(element="Al", size="3 4 4", vacuum=20, orthogonal=True)

wf.AddWaterFilm = AddWaterFilm(electrode=wf.Surface)

wf.AddNeonLayer = AddNeonLayer(structure=wf.AddWaterFilm)

wf.Plot3d = Plot3d(structure=wf.AddNeonLayer)
