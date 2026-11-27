import unittest

from aiflow_nodes.controls import FloatToPath


class TestFloatToPath(unittest.TestCase):
    def test_with_subdir(self):
        self.assertEqual(
            FloatToPath._original_func("runs/Ag", 0.01, "strain", "bulk"),
            "runs/Ag/strain_+0.010000/bulk",
        )

    def test_negative_value(self):
        self.assertEqual(
            FloatToPath._original_func("runs/Ag", -0.05, "strain", "bulk"),
            "runs/Ag/strain_-0.050000/bulk",
        )

    def test_other_prefix(self):
        self.assertEqual(
            FloatToPath._original_func("runs/Ag", 0.5, "fraction", "slip"),
            "runs/Ag/fraction_+0.500000/slip",
        )

    def test_no_subdir_and_custom_format(self):
        self.assertEqual(
            FloatToPath._original_func("runs/Ag", 300, "T", fmt=".0f"),
            "runs/Ag/T_300",
        )

    def test_matches_strain_working_directory_layout(self):
        """Defaults reproduce SurfaceStressNodes.StrainWorkingDirectory's paths."""
        from aiflow_nodes.SurfaceStressNodes import StrainWorkingDirectory

        for strain in (-0.05, 0.0, 0.0125):
            self.assertEqual(
                FloatToPath._original_func("base", strain, "strain", "surface"),
                StrainWorkingDirectory._original_func("base", strain, "surface"),
            )


if __name__ == "__main__":
    unittest.main()
