"""Logging management."""
import logging
import sys
from typing import Type

from pisces.utilities.config import pisces_params
from pisces.utilities._typing import Instance

# Setting up the logging system
streams = dict(
    mylog=getattr(sys, pisces_params['logging.mylog.stream']),
    devlog=getattr(sys, pisces_params['logging.devlog.stream']),
)
_loggers = dict(
    mylog=logging.Logger("cluster_generator"), devlog=logging.Logger("cg-development")
)

_handlers = {}

for k, v in _loggers.items():
    # Construct the formatter string.
    _handlers[k] = logging.StreamHandler(streams[k])
    _handlers[k].setFormatter(pisces_params[f'logging.{k}.format'])
    v.addHandler(_handlers[k])
    v.setLevel(pisces_params[f'logging.{k}.level'])
    v.propagate = False

    if k != "mylog":
        v.disabled = not pisces_params[f'logging.{k}.enabled']

mylog: logging.Logger = _loggers["mylog"]
""":py:class:`logging.Logger`: The main logger for ``pyXMIP``."""
devlog: logging.Logger = _loggers["devlog"]
""":py:class:`logging.Logger`: The development logger for ``pyXMIP``."""


class LogDescriptor:
    def __get__(self, instance: Instance, owner: Type[Instance]) -> logging.Logger:
        # check owner for an existing logger:
        logger = logging.getLogger(owner.__name__)

        if len(logger.handlers) > 0:
            pass
        else:
            # the logger needs to be created.
            _handler = logging.StreamHandler(
                getattr(sys, pisces_params['logging.mylog.stream'])
            )
            _handler.setFormatter(logging.Formatter(pisces_params['logging.code.format']))
            logger.addHandler(_handler)
            logger.setLevel(pisces_params['logging.code.level'])
            logger.propagate = False
            logger.disabled = not pisces_params['logging.code.enabled']

        return logger


# Defining special exceptions and other error raising entities
class ErrorGroup(Exception):
    """Special error class containing a group of exceptions."""

    def __init__(self, message: str, error_list: list[Exception]):
        self.errors: list[Exception] = error_list

        # Determining the message
        self.message = f"{len(self.errors)} ERRORS: {message}\n\n"

        for err_id, err in enumerate(self.errors):
            self.message += "%(err_id)-5s[%(err_type)10s]: %(msg)s\n" % dict(
                err_id=err_id + 1, err_type=err.__class__.__name__, msg=str(err)
            )

    def __len__(self):
        return len(self.errors)

    def __str__(self):
        return self.message