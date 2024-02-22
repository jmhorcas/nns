import os
import argparse
from typing import Any

from flamapy.metamodels.pysat_metamodel.transformations import DimacsReader

from nns4fms.utils import utils


def get_dimacs_stats(model_path: str) -> tuple[Any, Any, Any]:
    # Read model
    sat_model = DimacsReader(model_path).transform()

    # Get stats
    clauses = sat_model.get_all_clauses().clauses
    n_clauses = len(clauses)
    max_terms_in_clause = len(max(clauses, key=len))
    n_variables = len(sat_model.variables) 

    return n_variables, n_clauses, max_terms_in_clause
 

def main_model(model_path: str) -> None:
    n_variables, n_clauses, max_terms_in_clause = get_dimacs_stats(model_path)
    print(f'#variable: {n_variables}')
    print(f'#Clauses: {n_clauses}')
    print(f'Max terms: {max_terms_in_clause}')


def main_dir(dir: str) -> None:
    n_variables_dict = {}
    n_clauses_dict = {}
    max_terms_in_clause_dict = {}
    for model_path in utils.get_filepaths(dir, ['.dimacs']):
        # Get feature model name
        path, filename = os.path.split(model_path)
        filename = '.'.join(filename.split('.')[:-1])

        n_variables, n_clauses, max_terms_in_clause = get_dimacs_stats(model_path)
        n_variables_dict[filename] = n_variables
        n_clauses_dict[filename] = n_clauses
        max_terms_in_clause_dict[filename] = max_terms_in_clause

    # Calc statistics
    max_variables_key = max(n_variables_dict, key=lambda key: n_variables_dict[key])
    max_clauses_key = max(n_clauses_dict, key=lambda key: n_clauses_dict[key])
    max_terms_key = max(max_terms_in_clause_dict, key=lambda key: max_terms_in_clause_dict[key])

    print(f'Max variable: {n_variables_dict[max_variables_key]} ({max_variables_key})')
    print(f'Max clauses: {n_clauses_dict[max_clauses_key]} ({max_clauses_key})')
    print(f'Max terms: {max_terms_in_clause_dict[max_terms_key]} ({max_terms_key})')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Get stats for a batch of .dimacs models.')
    input_model = parser.add_mutually_exclusive_group(required=True)
    input_model.add_argument('-fm', '--featuremodel', dest='feature_model', type=str, help='Input feature model in .dimacs format.')
    input_model.add_argument('-d', '--dir', dest='dir', type=str, help='Input directory with the models in the same format (.dimacs) to be analyzed.')
    args = parser.parse_args()

    if args.feature_model:
        main_model(args.feature_model)
    elif args.dir:
        main_dir(args.dir)
