import unittest
from types import SimpleNamespace

import numpy as np

from emsoft_di.monte_carlo import MCSettings, material_properties, simulate_cpu
from emsoft_di.crystal_file import (
    interactive_spec, materials_project_spec, validate_spec,
)


class MonteCarloTests(unittest.TestCase):
    def setUp(self):
        # A conventional four-atom fcc Ni cell for material-property checks.
        self.crystal = SimpleNamespace(
            positions=[np.zeros((4, 3))],
            atomic_numbers=np.asarray([28]),
            occupancy=np.asarray([1.0]),
            volume=0.356**3,
        )

    def test_material_properties_from_crystal(self):
        z, atomic_weight, density = material_properties(self.crystal)
        self.assertEqual(z, 28.0)
        self.assertAlmostEqual(atomic_weight, 58.6934, places=4)
        self.assertAlmostEqual(density, 8.6407, places=3)

    def test_seeded_cpu_smoke_has_expected_histogram_shapes(self):
        settings = MCSettings(
            electrons=200, energy_kev=30, energy_min_kev=1,
            energy_bin_kev=1, detector_size=11, steps=30, seed=10,
        )
        accum_e, accum_z = simulate_cpu(self.crystal, settings)
        accum_e_again, accum_z_again = simulate_cpu(self.crystal, settings)
        self.assertEqual(accum_e.shape, (11, 11, 30))
        self.assertEqual(accum_z.shape, (1, 1, 101, 30))
        self.assertLessEqual(int(accum_e.sum()), settings.electrons)
        self.assertTrue(np.array_equal(accum_e, accum_e_again))
        self.assertTrue(np.array_equal(accum_z, accum_z_again))
        self.assertTrue(np.all(accum_z.sum(axis=(0, 1, 2)) <= accum_e.sum(axis=(0, 1))))

    def test_emmkxtal_json_spec_maps_to_emsoft_crystal_datasets(self):
        spec = {
            'crystal_system': 1,
            'lattice_parameters_nm_deg': [0.35236, 0.35236, 0.35236, 90, 90, 90],
            'space_group_number': 225,
            'space_group_setting': 1,
            'sites': [{
                'atomic_number': 28,
                'fractional_coordinates': [0, 0, 0],
                'occupancy': 1,
                'b_iso_nm2': 0.0035,
            }],
            'source': 'test',
        }
        normalized = validate_spec(spec)
        self.assertEqual(normalized['atom_types'].tolist(), [28])
        self.assertEqual(normalized['atom_data'].shape, (5, 1))
        np.testing.assert_allclose(normalized['atom_data'][:, 0], [0, 0, 0, 1, 0.0035])

    def test_crystal_spec_rejects_nonstandard_setting(self):
        with self.assertRaisesRegex(ValueError, 'setting 1'):
            validate_spec({
                'crystal_system': 1,
                'lattice_parameters_nm_deg': [0.35, 0.35, 0.35, 90, 90, 90],
                'space_group_number': 225,
                'space_group_setting': 2,
                'sites': [{'atomic_number': 28, 'fractional_coordinates': [0, 0, 0]}],
            })

    def test_interactive_crystal_entry_matches_emmkxtal_field_order(self):
        answers = iter([
            '1', '0.35236', '225', '28', '0', '0', '0', '1', '0.0035', 'n',
            'EMsoft Crystal Data Entry Example: fcc Ni',
        ])
        spec = interactive_spec(input_fn=lambda _prompt: next(answers))
        normalized = validate_spec(spec)
        self.assertEqual(spec['crystal_system'], 1)
        self.assertEqual(spec['lattice_parameters_nm_deg'], [0.35236, 0.35236, 0.35236, 90, 90, 90])
        self.assertEqual(normalized['atom_types'].tolist(), [28])
        np.testing.assert_allclose(normalized['atom_data'][:, 0], [0, 0, 0, 1, 0.0035])

    def test_materials_project_conversion_uses_requested_missing_data_defaults(self):
        class Element:
            Z = 28

        class Species:
            element = Element()

        species = Species()
        site = SimpleNamespace(
            frac_coords=np.asarray([0.0, 0.0, 0.0]),
            species={species: 1.0}, properties={},
        )
        structure = SimpleNamespace(
            lattice=SimpleNamespace(abc=[3.52, 3.52, 3.52], angles=[90, 90, 90])
        )
        symmetrized = SimpleNamespace(equivalent_sites=[[site]])

        class FakeAnalyzer:
            def __init__(self, _structure, symprec):
                self.symprec = symprec

            def get_refined_structure(self):
                return structure

            def get_symmetrized_structure(self):
                return symmetrized

            def get_crystal_system(self):
                return 'cubic'

            def get_space_group_number(self):
                return 225

        document = SimpleNamespace(
            structure=structure, material_id='mp-example', formula_pretty='Ni'
        )
        spec = materials_project_spec(document, analyzer_class=FakeAnalyzer)
        np.testing.assert_allclose(
            spec['lattice_parameters_nm_deg'], [0.352, 0.352, 0.352, 90, 90, 90]
        )
        self.assertEqual(spec['sites'][0]['atomic_number'], 28)
        self.assertEqual(spec['sites'][0]['occupancy'], 1.0)
        self.assertEqual(spec['sites'][0]['b_iso_nm2'], 0.005)


if __name__ == '__main__':
    unittest.main()
