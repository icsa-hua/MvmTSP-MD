from typing import List, Dict 

def opa_weights(ranks:List[int], criteria_names:List[str])->Dict[str,float]: 
    """
    Calculates weights using the Ordinal Priority Approach (OPA).

    Args:
        ranks (list): Ordinal ranks of each criterion (1 = most important).
        criteria_names (list): Names of the criteria.

    Returns:
        dict: Mapping of criterion name to its normalized weight.
    """
    weights = [1 / r for r in ranks]
    weights = [w / sum(weights) for w in weights]
    return dict(zip(criteria_names, weights))