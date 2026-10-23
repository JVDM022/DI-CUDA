"""Create EMsoft-compatible crystal-structure HDF5 (.xtal) files in Python.

The file fields and layout follow EMsoft's EMmkxtal / SaveDataHDF routines.
Input is an explicit JSON description of the unit cell and asymmetric sites;
the crystallographic values must come from a trusted source such as a paper,
materials database, or CIF that has been checked for the correct phase.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path

import numpy as np


_TEMPLATE = {
    "crystal_system": 1,
    "lattice_parameters_nm_deg": [0.35236, 0.35236, 0.35236, 90.0, 90.0, 90.0],
    "space_group_number": 225,
    "space_group_setting": 1,
    "sites": [
        {
            "atomic_number": 28,
            "fractional_coordinates": [0.0, 0.0, 0.0],
            "occupancy": 1.0,
            "b_iso_nm2": 0.0035,
        }
    ],
    "source": "EMsoft Crystal Data Entry Example; fcc Ni; Peng et al., Acta Cryst. A 52 (1996), 456-470",
}

_CRYSTAL_SYSTEMS = {
    1: 'Cubic', 2: 'Tetragonal', 3: 'Orthorhombic', 4: 'Hexagonal',
    5: 'Trigonal (standard hexagonal setting)', 6: 'Monoclinic', 7: 'Triclinic',
}


def _ask_number(prompt, input_fn=input, *, integer=False):
    while True:
        raw = input_fn(prompt).strip()
        try:
            value = int(raw) if integer else float(raw)
            if np.isfinite(value):
                return value
        except ValueError:
            pass
        print('Enter a valid ' + ('integer.' if integer else 'number.'))


def interactive_spec(input_fn=input):
    """Prompt for the same crystal fields requested by EMsoft's EMmkxtal."""
    print('Select crystal system:')
    for number, label in _CRYSTAL_SYSTEMS.items():
        print(f'  {number}. {label}')
    system = int(_ask_number('Crystal system [1-7]: ', input_fn, integer=True))
    if system not in _CRYSTAL_SYSTEMS:
        raise ValueError('Choose a crystal system from 1 through 7')

    def ask_length(label):
        return _ask_number(f'{label} [nm]: ', input_fn)

    def ask_angle(label):
        return _ask_number(f'{label} [degrees]: ', input_fn)

    if system == 1:
        a = ask_length('a')
        lattice = [a, a, a, 90, 90, 90]
    elif system == 2:
        a, c = ask_length('a'), ask_length('c')
        lattice = [a, a, c, 90, 90, 90]
    elif system == 3:
        a, b, c = ask_length('a'), ask_length('b'), ask_length('c')
        lattice = [a, b, c, 90, 90, 90]
    elif system in (4, 5):
        a, c = ask_length('a'), ask_length('c')
        lattice = [a, a, c, 90, 90, 120]
    elif system == 6:
        a, b, c = ask_length('a'), ask_length('b'), ask_length('c')
        beta = ask_angle('beta')
        lattice = [a, b, c, 90, beta, 90]
    else:
        a, b, c = ask_length('a'), ask_length('b'), ask_length('c')
        alpha, beta, gamma = (ask_angle('alpha'), ask_angle('beta'), ask_angle('gamma'))
        lattice = [a, b, c, alpha, beta, gamma]

    sg = int(_ask_number('Space-group number [1-230]: ', input_fn, integer=True))
    sites = []
    while True:
        z = int(_ask_number('Atomic number: ', input_fn, integer=True))
        xyz = [_ask_number(f'{axis} fractional coordinate [0-1): ', input_fn)
               for axis in ('x', 'y', 'z')]
        occupancy = _ask_number('Site occupancy [0-1]: ', input_fn)
        b_iso = _ask_number('Isotropic Debye-Waller factor [nm^2]: ', input_fn)
        sites.append({
            'atomic_number': z, 'fractional_coordinates': xyz,
            'occupancy': occupancy, 'b_iso_nm2': b_iso,
        })
        if input_fn('Add another asymmetric-unit atom? [y/N]: ').strip().lower() not in ('y', 'yes'):
            break
    citation = input_fn('Structure source/citation: ').strip()
    return {
        'crystal_system': system,
        'lattice_parameters_nm_deg': lattice,
        'space_group_number': sg,
        'space_group_setting': 1,
        'sites': sites,
        'source': citation,
    }


def validate_spec(spec):
    """Validate EMmkxtal inputs and return normalized numeric arrays."""
    if not isinstance(spec, dict):
        raise ValueError('Crystal specification must be a JSON object')
    required = ('crystal_system', 'lattice_parameters_nm_deg',
                'space_group_number', 'sites')
    missing = [key for key in required if key not in spec]
    if missing:
        raise ValueError(f'Missing crystal fields: {", ".join(missing)}')
    system = int(spec['crystal_system'])
    sg = int(spec['space_group_number'])
    setting = int(spec.get('space_group_setting', 1))
    lattice = np.asarray(spec['lattice_parameters_nm_deg'], dtype=np.float64)
    sites = spec['sites']
    if system not in range(1, 8):
        raise ValueError('crystal_system must be 1 (cubic) through 7 (triclinic)')
    if not 1 <= sg <= 230:
        raise ValueError('space_group_number must be between 1 and 230')
    # The downstream Python crystal reader currently handles standard setting 1.
    if setting != 1:
        raise ValueError('Only standard space-group setting 1 is supported')
    if lattice.shape != (6,) or not np.isfinite(lattice).all():
        raise ValueError('lattice_parameters_nm_deg must contain six finite values')
    if np.any(lattice[:3] <= 0) or np.any(lattice[3:] <= 0) or np.any(lattice[3:] >= 180):
        raise ValueError('Cell lengths must be positive and angles must be between 0 and 180 degrees')
    if not isinstance(sites, list) or not sites:
        raise ValueError('sites must be a non-empty list of asymmetric-unit atoms')
    atom_data = []
    atom_types = []
    for index, site in enumerate(sites, start=1):
        try:
            z = int(site['atomic_number'])
            xyz = np.asarray(site['fractional_coordinates'], dtype=np.float64)
            occupancy = float(site.get('occupancy', 1.0))
            b_iso = float(site.get('b_iso_nm2', 0.004))
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f'Invalid site {index}: {exc}') from exc
        if not 1 <= z <= 98:
            raise ValueError(f'Site {index}: atomic_number must be between 1 and 98')
        if xyz.shape != (3,) or not np.isfinite(xyz).all() or np.any(xyz < 0) or np.any(xyz >= 1):
            raise ValueError(f'Site {index}: fractional coordinates must be three values in [0, 1)')
        if not np.isfinite(occupancy) or not 0 <= occupancy <= 1:
            raise ValueError(f'Site {index}: occupancy must be in [0, 1]')
        if not np.isfinite(b_iso) or b_iso < 0:
            raise ValueError(f'Site {index}: b_iso_nm2 must be finite and non-negative')
        atom_types.append(z)
        atom_data.append([*xyz, occupancy, b_iso])
    if not any(site[3] > 0 for site in atom_data):
        raise ValueError('At least one site must have positive occupancy')
    source = str(spec.get('source', 'User supplied crystal structure')).strip()
    if not source:
        raise ValueError('source must identify where the structure values came from')
    if len(source) > 512:
        raise ValueError('source must be 512 characters or fewer')
    return {
        'crystal_system': system,
        'lattice': lattice,
        'space_group_number': sg,
        'space_group_setting': setting,
        'atom_types': np.asarray(atom_types, dtype=np.int32),
        # EMsoft's HDF5 helper reverses Fortran dimensions. This is the layout
        # read by h5py and expected by emsoft_di.master.physics.read_crystal.
        'atom_data': np.asarray(atom_data, dtype=np.float32).T,
        'source': source,
    }


def write_xtal(spec, output):
    """Write a new EMsoft-style `.xtal` HDF5 file; existing paths are preserved."""
    import h5py

    data = validate_spec(spec)
    target = Path(output).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    now = datetime.now(timezone.utc)
    strings = h5py.string_dtype(encoding='utf-8')
    with h5py.File(target, 'x') as file:
        crystal = file.create_group('CrystalData')
        crystal.create_dataset('ProgramName', data=np.asarray(['KikuchiCUDA Python crystal writer'], dtype=strings))
        crystal.create_dataset('CreationDate', data=np.asarray([now.strftime('%Y-%m-%d')], dtype=strings))
        crystal.create_dataset('CreationTime', data=np.asarray([now.strftime('%H:%M:%S UTC')], dtype=strings))
        crystal.create_dataset('Creator', data=np.asarray(['KikuchiCUDA Python'], dtype=strings))
        crystal['CrystalSystem'] = np.asarray([data['crystal_system']], dtype=np.int32)
        crystal['LatticeParameters'] = data['lattice']
        crystal['SpaceGroupNumber'] = np.asarray([data['space_group_number']], dtype=np.int32)
        crystal['SpaceGroupSetting'] = np.asarray([data['space_group_setting']], dtype=np.int32)
        crystal['Natomtypes'] = np.asarray([len(data['atom_types'])], dtype=np.int32)
        crystal['Atomtypes'] = data['atom_types']
        crystal['AtomData'] = data['atom_data']
        crystal.create_dataset('Source', data=np.asarray([data['source']], dtype=strings))
    print(f'Created EMsoft crystal HDF5: {target}')
    return target


def extract_xtal(source, output):
    """Extract CrystalData from an EMsoft result HDF5 into a standalone .xtal."""
    import h5py

    source = Path(source).expanduser().resolve(strict=True)
    target = Path(output).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(source, 'r') as src:
        if 'CrystalData' not in src:
            raise ValueError(f'{source} has no top-level CrystalData group')
        with h5py.File(target, 'x') as dst:
            src.copy('CrystalData', dst)
    print(f'Extracted EMsoft crystal HDF5: {target}')
    return target


_MP_CRYSTAL_SYSTEM = {
    'cubic': 1, 'tetragonal': 2, 'orthorhombic': 3, 'hexagonal': 4,
    'trigonal': 5, 'rhombohedral': 5, 'monoclinic': 6, 'triclinic': 7,
}


def materials_project_spec(document, *, symprec=0.1, analyzer_class=None):
    """Convert an MP structure document to the EMmkxtal asymmetric-site fields."""
    if analyzer_class is None:
        try:
            from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
        except ImportError as exc:
            raise RuntimeError('Install pymatgen to convert Materials Project structures') from exc
        analyzer_class = SpacegroupAnalyzer
    conventional = analyzer_class(document.structure, symprec=symprec).get_refined_structure()
    symmetry = analyzer_class(conventional, symprec=symprec)
    symmetrized = symmetry.get_symmetrized_structure()
    crystal_system = symmetry.get_crystal_system().lower()
    if crystal_system not in _MP_CRYSTAL_SYSTEM:
        raise ValueError(f'Unsupported Materials Project crystal system: {crystal_system}')

    sites = []
    for equivalent_sites in symmetrized.equivalent_sites:
        representative = equivalent_sites[0]
        coordinates = np.mod(np.asarray(representative.frac_coords, dtype=float), 1.0)
        components = list(representative.species.items())
        if not components:
            components = [(representative.specie, 1.0)]
        properties = getattr(representative, 'properties', {}) or {}
        dwf = next((float(properties[key]) for key in
                    ('b_iso_nm2', 'B_iso_nm2', 'debye_waller_nm2', 'dwf_nm2')
                    if key in properties), 0.005)
        for species, amount in components:
            element = getattr(species, 'element', species)
            atomic_number = getattr(element, 'Z', None)
            if atomic_number is None:
                raise ValueError(f'Could not determine atomic number for MP site species {species!r}')
            sites.append({
                'atomic_number': int(atomic_number),
                'fractional_coordinates': coordinates.tolist(),
                'occupancy': 1.0 if amount is None else float(amount),
                'b_iso_nm2': dwf,
            })

    material_id = str(getattr(document, 'material_id', 'unknown'))
    formula = str(getattr(document, 'formula_pretty', material_id))
    lattice = conventional.lattice
    lengths_nm = np.asarray(lattice.abc, dtype=float) * 0.1  # Angstrom to nm
    angles_deg = np.asarray(lattice.angles, dtype=float)
    return {
        'crystal_system': _MP_CRYSTAL_SYSTEM[crystal_system],
        'lattice_parameters_nm_deg': np.concatenate((lengths_nm, angles_deg)).tolist(),
        'space_group_number': int(symmetry.get_space_group_number()),
        'space_group_setting': 1,
        'sites': sites,
        'source': (f'Materials Project {material_id} ({formula}); retrieved '
                   f'{datetime.now(timezone.utc).date().isoformat()}; DWF default 0.005 nm^2 when absent'),
    }


def materials_project_xtal(*, formula=None, material_id=None, output, api_key=None,
                           input_fn=input, symprec=0.1):
    """Search/select an MP structure, then write an EMsoft-compatible .xtal."""
    try:
        from mp_api.client import MPRester
    except ImportError as exc:
        raise RuntimeError("Install Materials Project support with `python -m pip install -e '.[materials-project]'`") from exc
    api_key = api_key or os.environ.get('MP_API_KEY')
    if not api_key:
        raise RuntimeError('Set MP_API_KEY in the environment; do not put the key in a command or repository file')
    if bool(formula) == bool(material_id):
        raise ValueError('Provide exactly one of formula or material_id')
    fields = ['material_id', 'formula_pretty', 'structure']
    with MPRester(api_key) as mpr:
        if material_id:
            documents = mpr.materials.summary.search(material_ids=[material_id], fields=fields)
        else:
            documents = mpr.materials.summary.search(formula=formula, fields=fields, deprecated=False)
    if not documents:
        raise ValueError('Materials Project returned no matching structures')
    if len(documents) == 1:
        selected = documents[0]
    else:
        print(f'Materials Project matches for {formula or material_id}:')
        for index, document in enumerate(documents, start=1):
            print(f'  {index}. {document.material_id}  {document.formula_pretty}')
        while True:
            try:
                choice = int(input_fn(f'Select structure [1-{len(documents)}]: '))
                if 1 <= choice <= len(documents):
                    selected = documents[choice - 1]
                    break
            except ValueError:
                pass
            print('Enter one of the listed numbers.')
    return write_xtal(materials_project_spec(selected, symprec=symprec), output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    template = commands.add_parser('template', help='write a starter JSON spec for fcc nickel')
    template.add_argument('--output', type=Path, required=True)
    create = commands.add_parser('create', help='create a .xtal HDF5 from a JSON crystal specification')
    create.add_argument('--input', type=Path, required=True)
    create.add_argument('--output', type=Path, required=True)
    extract = commands.add_parser('extract', help='copy CrystalData from an existing EMsoft HDF5')
    extract.add_argument('--input', type=Path, required=True)
    extract.add_argument('--output', type=Path, required=True)
    interactive = commands.add_parser('interactive', help='enter crystal data through EMmkxtal-style prompts')
    interactive.add_argument('--output', type=Path, required=True)
    mp = commands.add_parser('materials-project', help='search Materials Project and write the selected structure as .xtal')
    selection = mp.add_mutually_exclusive_group(required=True)
    selection.add_argument('--formula', help='formula to search, e.g. Ni or Ni3Al')
    selection.add_argument('--material-id', help='exact Materials Project ID, e.g. mp-149')
    mp.add_argument('--output', type=Path, required=True)
    mp.add_argument('--symprec', type=float, default=0.1,
                    help='symmetry tolerance in angstrom (default: 0.1)')
    args = parser.parse_args()
    if args.command == 'template':
        target = args.output.expanduser().resolve()
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open('x', encoding='utf-8') as stream:
            json.dump(_TEMPLATE, stream, indent=2)
            stream.write('\n')
        print(f'Created starter crystal specification: {target}')
    elif args.command == 'create':
        spec = json.loads(args.input.expanduser().read_text(encoding='utf-8'))
        write_xtal(spec, args.output)
    elif args.command == 'interactive':
        write_xtal(interactive_spec(), args.output)
    elif args.command == 'materials-project':
        materials_project_xtal(formula=args.formula, material_id=args.material_id,
                               output=args.output, symprec=args.symprec)
    else:
        extract_xtal(args.input, args.output)


if __name__ == '__main__':
    main()
