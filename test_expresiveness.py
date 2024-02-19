import os

from flamapy.metamodels.fm_metamodel.transformations import UVLWriter

from language_constructs.utils import utils
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


MAX_FEATURES = 3
OUTPUT_FOLDER = 'generated'


if __name__ == "__main__":
    if not os.path.exists(OUTPUT_FOLDER):
        os.makedirs(OUTPUT_FOLDER)

    features_names = [f'F{i}' for i in range(1, MAX_FEATURES + 1)]
    language_constructs = [FeatureModelConstruct, RootFeature, OptionalFeature, MandatoryFeature, XorGroup, OrGroup, XorChildFeature, OrChildFeature]
    optional_language_constructs = [RequiresConstraint, ExcludesConstraint]
    #optional_language_constructs = []
    language = FMLanguage(language_constructs, optional_language_constructs)
    #fms = language.generate_feature_models(features_names)
    # for i, fm in enumerate(fms, 1):
    #     print(f'FM{i}: {fm}')
    #     output_file = os.path.join(OUTPUT_FOLDER, f'fm{i}_{len(fm.get_features())}f_{len(fm.get_constraints())}c.uvl')
    #     UVLWriter(fm, output_file).transform()
    count = language.generate_feature_models_and_serializing(features_names, OUTPUT_FOLDER)
    print(f'#Total models generated: {count}')
    
    # Measure the expressiveness of the models
    # Number of configurations
    #ps = utils.powerset(features_names)
    #print(f'#Configurations: {len(ps)}')
    # for i, p in enumerate(ps, 1):
    #     print(f'Config {i}: {p}')

    # Number of products lines
    #pps = utils.powerset(ps)
    # for i, p in enumerate(pps, 1):
    #     print(f'SPL {i}: {p}')
    #print(f'#SPLs: {len(pps)}')