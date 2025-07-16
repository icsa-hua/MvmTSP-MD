import numpy as np

from typing import List, Dict 

def opa_weights(ranks:np.ndarray, criteria_names:List[str])->Dict[str,float]: 
    """
    Calculates weights using the Ordinal Priority Approach (OPA).
    """

    weights = [1 / r for r in ranks]
    weights = [w / sum(weights) for w in weights]
    return dict(zip(criteria_names, weights))
