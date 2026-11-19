from aiflow.core import as_function_node


@as_function_node("structure")
def BinaryB2Bulk(species_a: str = "Ni", species_b: str = "Ti", a: float = 3.015):
    """Build the 2-atom cubic B2 (CsCl-type, Pm-3m) cell of an equiatomic AB compound.

    ``species_a`` sits at (0, 0, 0), ``species_b`` at (1/2, 1/2, 1/2).
    ``a`` is the cubic lattice parameter in Angstrom (NiTi austenite ~3.015).
    """
    from ase.spacegroup import crystal

    structure = crystal(
        symbols=[species_a, species_b],
        basis=[(0.0, 0.0, 0.0), (0.5, 0.5, 0.5)],
        spacegroup=221,
        cellpar=[a, a, a, 90.0, 90.0, 90.0],
        primitive_cell=False,
    )
    return structure


@as_function_node("structure")
def BinaryB19PrimeBulk(
    species_a: str = "Ni",
    species_b: str = "Ti",
    a: float = 2.898,
    b: float = 4.108,
    c: float = 4.646,
    beta: float = 97.78,
    x_a: float = 0.0372,
    z_a: float = 0.6752,
    x_b: float = 0.4176,
    z_b: float = 0.2164,
):
    """Build the 4-atom monoclinic B19' (P2_1/m, unique axis b) cell of an AB compound.

    Both species occupy Wyckoff 2e sites (x, 1/4, z).  The defaults are the
    experimental NiTi martensite parameters of Kudoh et al., Acta Metall. 33,
    2049 (1985); lengths in Angstrom, the monoclinic angle ``beta`` in degrees.
    """
    from ase.spacegroup import crystal

    structure = crystal(
        symbols=[species_a, species_b],
        basis=[(x_a, 0.25, z_a), (x_b, 0.25, z_b)],
        spacegroup=11,
        cellpar=[a, b, c, 90.0, beta, 90.0],
        primitive_cell=False,
    )
    return structure
