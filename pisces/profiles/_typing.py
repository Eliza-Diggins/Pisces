from typing import Callable

from numpy.typing import ArrayLike

ProfileType = Callable[[ArrayLike, ...], ArrayLike]
