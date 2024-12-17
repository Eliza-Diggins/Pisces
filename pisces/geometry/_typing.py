from typing import Union, Callable
import numpy as np

AxisAlias: Union[str,int]
ScalarFieldFunction: Callable[[np.ndarray,...],np.ndarray]
VectorFieldFunction: Callable[[np.ndarray,...],np.ndarray]