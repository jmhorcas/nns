import logging
from typing import Any

#import tensorflow
#import numpy
#from matplotlib import pyplot

from flamapy.metamodels.pysat_metamodel.transformations import DimacsReader


#logger = tensorflow.get_logger()
#logger.setLevel(logging.ERROR)
           

FM_PATH = 'models/linux-26333_simple.dimacs'
MAX_CLAUSES = 100


def padding_clause(clause: list[int], size: int) -> list[int]:
    """Fill a clause with zeros until the clause's length is size."""
    return clause + [0] * (size - len(clause))


def padding_model(clauses: list[list[int]], size: int) -> list[list[int]]:
    """Fill the list of clauses with new empty clauses until getting size clauses."""
    return clauses + [[0] * len(clauses[0])] * (size - len(clauses))


def main():
    sat_model = DimacsReader(FM_PATH).transform()
    clauses = sat_model.get_all_clauses().clauses
    max_clause = max(len(c) for c in clauses)
    print(f'#Variables: {len(sat_model.variables)}')
    #print(f'Clauses: {clauses}')
    print(f'#Clauses: {len(clauses)}')
    print(f'#Max clause: {max_clause}')

    clauses.append([])
    padding_clauses = [padding_clause(clause, max_clause) for clause in clauses]
    #print(f'Padding Clauses: {padding_clauses}')
    padding_clauses = padding_model(padding_clauses, MAX_CLAUSES)
    print(padding_clauses)

    # Testing the execution time of some stuf.
    #import timeit
    #exec_time = timeit.timeit(lambda: [clause + [0] * (max_clause - len(clause)) for clause in clauses], number=10000)
    #print(exec_time)


if __name__ == '__main__':
    main()