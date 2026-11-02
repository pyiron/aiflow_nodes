"""``aiflow_nodes`` as a node library for the aiflow.ai assistant."""

from importlib.metadata import entry_points

from aiflow.ai.node_library import ENTRY_POINT_GROUP, NodeLibrary, registered_libraries

from aiflow_nodes import assistant_plugin as plugin


def test_the_plugin_describes_aiflow_nodes(tmp_path):
    root = tmp_path / "aiflow_nodes"
    root.mkdir()

    library = plugin.library(root)

    assert isinstance(library, NodeLibrary)
    assert library.name == "aiflow_nodes"
    assert library.root == root
    assert library.guide == plugin.NODE_GUIDE and plugin.NODE_GUIDE.is_file()


def test_without_aiflow_nodes_the_plugin_offers_nothing(monkeypatch):
    monkeypatch.setattr(plugin, "package_root", lambda: None)

    assert plugin.library() is None


def test_the_plugin_is_registered_with_the_assistant():
    names = [ep.name for ep in entry_points(group=ENTRY_POINT_GROUP)]
    assert "aiflow_nodes" in names

    (library,) = [lib for lib in registered_libraries() if lib.name == "aiflow_nodes"]
    assert library.root == plugin.package_root()
    assert "from aiflow_nodes.atomistic.structure.build import Bulk" in library.index()


def test_the_guide_sits_alone_so_it_can_be_shared_without_the_package():
    """The benchmark hands the guide directory to the agent with ``--add-dir``."""
    assert sorted(p.name for p in plugin.GUIDES_DIR.iterdir()) == ["node_guide.md"]


def test_the_guide_carries_what_the_aiflow_guide_no_longer_does():
    text = plugin.NODE_GUIDE.read_text(encoding="utf-8")

    assert "### Package layout" in text
    assert "from aiflow_nodes.atomistic.structure.build     import Bulk" in text
    assert "pyiron_nodes" not in text
