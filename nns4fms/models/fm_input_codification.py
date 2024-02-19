import os

from flamapy.metamodels.pysat_metamodel.transformations import DimacsReader

from flamapy.metamodels.bdd_metamodel.transformations.sat_to_bdd import SATToBDD
from flamapy.metamodels.bdd_metamodel.operations import BDDProductsNumber


class FMInputCodification():
    """The feature model is codified by the values of its clauses.
    
    It preprocess the data by:
        (1) augmenting its clauses to the maximum number of clauses of the models in the directory,
        (2) augmenting the size of each clauses to the maximum number of terms in clauses,
        (3) [NOT NEEDED!!] normalizing the values of the variables to [0..1].
    """

    def __init__(self, model_path: str) -> None:
        # Get feature model name
        path, filename = os.path.split(model_path)
        filename = '.'.join(filename.split('.')[:-1])
        self._sat_model = DimacsReader(model_path).transform()
        self._bdd_model = None

        self.fm_name = filename
        self.features_variables_dict = self._sat_model.variables
        self.variables_features_dict = self._sat_model.features
        self.clauses = self._sat_model.get_all_clauses().clauses
    
    def get_codification(self, max_terms_in_clause: int, max_clauses: int) -> list[list[int]]:
        """Codify the clauses by augmenting the terms in clauses and the number of clauses 
        to the given numbers."""
        padding_clauses = [padding_clause(clause, max_terms_in_clause) for clause in self.clauses]
        padding_clauses = padding_model(padding_clauses, max_clauses)
        return padding_clauses

    def max_clauses(self) -> int:
        """Return the largest clause in the model."""
        return max(len(c) for c in self.clauses)
    
    def max_variable(self) -> int:
        """Return the maximum variable number in the model."""
        return max(self.variables_features_dict.keys())
    
    def get_configurations_number(self) -> int:
        if self._bdd_model is None:
            self._bdd_model = SATToBDD(self._sat_model).transform()
        return BDDProductsNumber().execute(self._bdd_model).get_result()


def padding_clause(clause: list[int], size: int) -> list[int]:
    """Fill a clause with zeros until the clause's length is size."""
    return clause + [0] * (size - len(clause))


def padding_model(clauses: list[list[int]], size: int) -> list[list[int]]:
    """Fill the list of clauses with new empty clauses until getting size clauses."""
    return clauses + [[0] * len(clauses[0])] * (size - len(clauses))
