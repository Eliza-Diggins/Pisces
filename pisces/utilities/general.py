from typing import Type, TypeVar

_T = TypeVar('_T')

def find_in_subclasses(base_class: Type[_T], class_name: str) -> Type[_T]:
    for subclass in base_class.__subclasses__():
        if subclass.__name__ == class_name:
            return subclass
        result = find_in_subclasses(subclass, class_name)
        if result:
            return result

    raise ValueError(f"Failed to find subclass of {base_class.__name__} named {class_name}.")