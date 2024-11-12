from typing import Callable
from numpy.typing import NDArray, ArrayLike

ProfileType = Callable[[ArrayLike,...],ArrayLike]