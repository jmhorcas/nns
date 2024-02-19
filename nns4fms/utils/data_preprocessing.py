from nns4fms.models import FMInputCodification
from nns4fms.utils import utils


def get_dataset_from_featuremodels(dir: str) -> list[FMInputCodification]:
    """Return the dataset from the feature models (in .dimacs) contained in a given directory."""
    return [FMInputCodification(path) for path in utils.get_filepaths(dir, ['.dimacs'])]

