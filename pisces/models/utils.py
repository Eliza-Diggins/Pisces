def node(name=None):
    """
    Decorator to mark a method as a pipeline.

    Parameters
    ----------
    name :
        The name to associate with this node. Depending on the design of the
        solver, this may be left as None, in which case it is named as the function name.
    """
    def decorator(func):
        func._is_node = True
        func.name = name or func.__name__
        return func
    return decorator

def condition(name: str, typ: str = 'model'):
    """
    Decorator to mark a method as a state checker.

    Parameters
    ----------
    name: str
        The kwarg name to register this checker under.
    typ : str
        Type of state checker ('model' or 'grid').
    """
    def decorator(func):
        func._is_condition = True
        func.type = typ
        func.name = name or func.__name__
        return func
    return decorator

