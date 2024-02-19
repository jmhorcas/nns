import os
import copy
import queue

from flamapy.metamodels.fm_metamodel.models import FeatureModel
from flamapy.metamodels.fm_metamodel.transformations import UVLWriter

from language_constructs.models import LanguageConstruct


class FMLanguage():

    def __init__(self, lcs: list[LanguageConstruct]) -> None:
        self.lcs = lcs

    def generate_feature_models(self, features_names: set[str]) -> list[FeatureModel]:
        incomplete_feature_models = queue.Queue()
        incomplete_feature_models.put(None)
        completed_fms = []
        while not incomplete_feature_models.empty():
            print(f'#Completed/Incompleted fms: {len(completed_fms)} / {incomplete_feature_models.qsize()}')
            fm = incomplete_feature_models.get()
            applicable_lcs = []
            for lc in self.lcs:
                applicable_lcs.extend(lc.get_applicable_instances(fm, features_names))
            if not applicable_lcs:
                completed_fms.append(fm)
            else:
                for alc in applicable_lcs:
                    new_fm = copy.deepcopy(fm)
                    incomplete_feature_models.put(alc.apply(new_fm))
        return completed_fms
    
    def generate_feature_models_and_serializing(self, features_names: set[str],
                                                output_folder: str) -> int:
        incomplete_feature_models = queue.Queue()
        incomplete_feature_models.put(None)
        count = 1
        while not incomplete_feature_models.empty():
            print(f'#Completed/Incompleted fms: {count} / {incomplete_feature_models.qsize()}')
            fm = incomplete_feature_models.get()
            applicable_lcs = []
            for lc in self.lcs:
                applicable_lcs.extend(lc.get_applicable_instances(fm, features_names))
            if not applicable_lcs:
                print(f'FM{count}: {fm}')
                output_file = os.path.join(output_folder, f'fm{count}_{len(fm.get_features())}f_{len(fm.get_constraints())}c.uvl')
                UVLWriter(fm, output_file).transform()
                count += 1
            else:
                for alc in applicable_lcs:
                    new_fm = copy.deepcopy(fm)
                    incomplete_feature_models.put(alc.apply(new_fm))
        return count
