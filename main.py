import pandas as pd
import numpy as np
import datetime
import time

import utils
import cost_calculator
from molecule import Molecule, Molecule_Hamiltonian

print('\n#####################################################################')
print('##                             TFermion                            ##')
print('##                                                                 ##') 
print('##  A non-Clifford gate cost assessment library of quantum phase   ##')
print('##            estimation algorithms for quantum chemistry          ##')
print('#####################################################################\n')

start_time = time.time()

# Read config file with the QFold configuration variables
config_path = './config/config.json'
tools = utils.Utils(config_path)

args = tools.parse_arguments()
if args.charge is None:
    args.charge = 0

# Detect type for the CLI argument (single target mode)
cli_type = tools.check_molecule_info(args.molecule_info)

# --------------------------- BATCH MODE ---------------------------
if not cli_type:
    results = {}

    for mol_name in tools.config_variables['molecules']:
        # Resuelve tipo y posible alias (p.ej. 'femoco' -> 'integrals/FeMoco.h5')
        args.molecule_info = mol_name
        molecule_info_type = tools.check_molecule_info(args.molecule_info)

        if molecule_info_type == "error":
            continue

        results[mol_name] = {}

        # Instancia de la molécula
        if molecule_info_type in ('name', 'geometry', 'h5'):
            molecule = Molecule(
                molecule_info=args.molecule_info,
                molecule_info_type=molecule_info_type,
                tools=tools,
                charge=args.charge
            )
        elif molecule_info_type == 'hamiltonian':
            molecule = Molecule_Hamiltonian(
                molecule_info=args.molecule_info,
                tools=tools
            )
        else:
            # tipo desconocido
            continue

        # Active space solo si viene de 'name' o 'geometry'
        if args.ao_labels and getattr(molecule, 'has_data', False) and molecule_info_type in ('name', 'geometry'):
            molecule.active_space(args.ao_labels[0].replace('\\', ''))

        # Calcula costes para todos los métodos declarados
        c_calculator = cost_calculator.Cost_calculator(molecule, tools, molecule_info_type)
        for method in tools.methods:
            c_calculator.calculate_cost(method)
            vals = [x for x in c_calculator.costs[method] if (not np.isnan(x) and not np.isinf(x))]
            print(method, mol_name, len(vals))
            median = np.nanmedian(vals) if len(vals) else np.nan
            results[mol_name][method] = "{:0.2e}".format(median) if not np.isnan(median) else "nan"

    pd.DataFrame(results).to_csv('./results/results_' + str(tools.config_variables['gauss2plane_overhead']) + '.csv')

# --------------------------- SINGLE MODE --------------------------
else:
    molecule_info_type = cli_type

    # Instancia según tipo
    if molecule_info_type in ('name', 'geometry', 'h5'):
        molecule = Molecule(
            molecule_info=args.molecule_info,
            molecule_info_type=molecule_info_type,
            tools=tools,
            charge=args.charge
        )
    elif molecule_info_type == 'hamiltonian':
        molecule = Molecule_Hamiltonian(
            molecule_info=args.molecule_info,
            tools=tools
        )
    else:
        raise SystemExit(1)

    # Active space solo si viene de 'name' o 'geometry'
    if args.ao_labels and getattr(molecule, 'has_data', False) and molecule_info_type in ('name', 'geometry'):
        molecule.active_space(args.ao_labels[0].replace('\\', ''))

    methods_to_execute = [args.method] if args.method != 'all' else tools.methods
    for method in methods_to_execute:
        c_calculator = cost_calculator.Cost_calculator(molecule, tools, molecule_info_type)
        c_calculator.calculate_cost(method)
        vals = [x for x in c_calculator.costs[method] if (not np.isnan(x) and not np.isinf(x))]
        median = np.nanmedian(vals) if len(vals) else np.nan
        print(
            '<i> RESULT => The cost to calculate the energy of',
            str(args.molecule_info).upper(),
            'with method', method.upper(),
            'is', "{:0.2e}".format(median) if not np.isnan(median) else "nan",
            'T gates'
        )

execution_time = time.time() - start_time


print('\n** -------------------------------------------------- **')
print('**                                                    **')
print('** Execution time     =>', str(datetime.timedelta(seconds=execution_time)) ,' in hh:mm:ss  **')
print('********************************************************\n\n')

#### TESTS ###
# Single (nombre → alias .h5):
#   python3 main.py femoco double_factorization
# Single (HDF5 directo):
#   python3 main.py integrals/FeMoco.h5 double_factorization
# Otros:
#   python3 main.py water qdrift 'C 2p'
#   python3 main.py water taylor_on_the_fly 'C 2p'

#### TESTS ###

# QDRIFT: python3 main.py water qdrift 'C 2p' OK
# RAND-HAM: python3 main.py water rand_ham 'C 2p' OK
# Taylor naive: python3 main.py water taylor_naive 'C 2p' OK
# Taylor on the fly: python3 main.py water taylor_on_the_fly 'C 2p' OK
# Configuration interaction: python3 main.py water configuration_interaction 'C 2p' OK
# Low Depth Trotter: python3 main.py water low_depth_trotter 'C 2p' OK
# Low Depth Taylor: python3 main.py water low_depth_taylor 'C 2p' OK
# Low Depth On The Fly: python3 main.py water low_depth_taylor_on_the_fly 'C 2p' OK
# Linear T: python3 main.py water linear_t 'C 2p' OK
# Sparsity Low Rank: python3 main.py water sparsity_low_rank 'C 2p' OK
# Interaction Picture: python3 main.py water interaction_picture 'C 2p' OK 
# Sublinear Scaling: python3 main.py water sublinear_scaling 'C 2p'DEPRECATED