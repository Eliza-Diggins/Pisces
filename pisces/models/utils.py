def pipeline(**kwargs):
    """
    Decorator to mark a method as a pipeline.

    Parameters
    ----------
    **kwargs : dict
        Metadata to associate with the pipeline.
    """
    def decorator(func):
        func._is_pipeline = True
        func._kwargs = kwargs
        return func
    return decorator

def state_checker(name: str = None, typ: str = 'model'):
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
        func._is_state_checker = True
        func.type = typ
        func.name = name or func.__name__
        return func
    return decorator

