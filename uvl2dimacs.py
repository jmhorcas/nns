import os
import argparse

from flamapy.metamodels.fm_metamodel.transformations import UVLReader
from flamapy.metamodels.pysat_metamodel.transformations import FmToPysat, DimacsWriter
           
from nns4fms.utils import utils


def main_model(fm_path: str) -> None:
    # Get feature model name
    path, filename = os.path.split(fm_path)
    filename = '.'.join(filename.split('.')[:-1])

    # Read feature model
    print(f'Reading feature model {filename}...')
    fm_model = UVLReader(fm_path).transform()
    print(f'#Features: {len(fm_model.get_features())}')
    print(f'#Constraints: {len(fm_model.get_constraints())}')
    
    # Convert to dimacs
    print(f'Converting to SAT...')
    sat_model = FmToPysat(fm_model).transform()
    print(f'#Variables: {len(sat_model.variables)}')
    print(f'#Clauses: {len(sat_model.get_all_clauses().clauses)}')
    
    output_path = os.path.join(path, f'{filename}.{DimacsWriter.get_destination_extension()}')
    print(f'Serializing model {output_path}...')
    DimacsWriter(output_path, sat_model).transform()


def main_dir(dir: str) -> None:
    for model_path in utils.get_filepaths(dir, ['.uvl']):
        main_model(model_path)


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