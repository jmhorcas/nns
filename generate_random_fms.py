import os
import argparse

from flamapy.metamodels.fm_metamodel.transformations import UVLWriter

from language_constructs.models import FMLanguage
from language_constructs.models.constructs import (
    FeatureModelConstruct, 
    RootFeature, 
    OptionalFeature, 
    MandatoryFeature, 
    XorGroup,
    OrGroup,
    XorChildFeature,
    OrChildFeature,
    RequiresConstraint,
    ExcludesConstraint
)


N_MODELS = 10
N_FEATURES = 1000
N_CONSTRAINTS = 100
OUTPUT_FOLDER = 'generated'


def generate_random_models(n_models: int, n_features: int, n_constraints: int) -> None:
    if not os.path.exists(OUTPUT_FOLDER):
        os.makedirs(OUTPUT_FOLDER)

    features_names = [f'F{i}' for i in range(1, n_features + 1)]
    language_constructs = [FeatureModelConstruct, RootFeature, OptionalFeature, MandatoryFeature, XorGroup, OrGroup, XorChildFeature, OrChildFeature]
    optional_language_constructs = [RequiresConstraint, ExcludesConstraint]
    language = FMLanguage(language_constructs, optional_language_constructs)
    for i in range(n_models):
        print(f'Generating model {i}...')
        fm = language.generate_random_feature_model(features_names, n_constraints)
        print(f'FM{i}: {fm}')
        output_file = os.path.join(OUTPUT_FOLDER, f'fm{i}_{len(fm.get_features())}f_{len(fm.get_constraints())}c.uvl')
        UVLWriter(fm, output_file).transform()
    print(f'#Total models generated: {n_models}')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate random feature model in .uvl format.')
    parser.add_argument('-n', '--n_models', dest='n_models', type=int, default=1, required=False, help='Number of features.')
    parser.add_argument('-f', '--features', dest='n_features', type=int, required=True, help='Number of features.')
    parser.add_argument('-c', '--constraints', dest='n_constraints', type=int, default=0, required=False, help='Number of cross-tree constraints.')
    args = parser.parse_args()

    generate_random_models(args.n_models, args.n_features, args.n_constraints)