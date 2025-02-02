import unittest  # testのためのライブラリ

import numpy as np

from src.modules.protein.exceptions import ProteinRepresentationsUnavailableException
from src.modules.protein.protein import Protein, ProteinProps, ProteinRaw


class TestProtein(unittest.TestCase):
    def setUp(self) -> None:
        self.raw: ProteinRaw = {"seq": "LLLYAASR", "representations": None}
        self.props: ProteinProps = {
            "ccs": None,
            "rt": 2485.3635,
            "mass": None,
            "length": len(self.raw["seq"]),
        }
        self.key = "key"

    def test_init(self) -> None:
        protein = Protein(raw=self.raw, props=self.props, key=self.key)

        self.assertDictEqual(protein.raw, self.raw)
        self.assertDictEqual(protein.props, self.props)
        self.assertEqual(protein.seq, self.raw["seq"])
        self.assertEqual(protein.length, self.props["length"])
        self.assertEqual(protein.key, self.key)

        with self.assertRaises(ProteinRepresentationsUnavailableException):
            protein.representations

    def test_set_representations(self) -> None:
        protein = Protein(raw=self.raw, props=self.props, key=self.key)

        with self.assertRaises(ProteinRepresentationsUnavailableException):
            protein.representations

        representations = np.array([0, 0, 0])
        protein.set_representations(representations=representations)

        np.testing.assert_array_equal(protein.representations, representations)
