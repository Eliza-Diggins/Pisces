"""
Logging and Error Management Module for Pisces

This module sets up logging utilities and defines a special exception class to handle grouped errors.

Loggers:
--------

- :py:attr:`mylog`: The main logger for the application.
- :py:attr:`devlog`: The development logger for debugging and development purposes.

The base levels of each of these loggers may be set in the configuration.
"""
import logging
import sys
from typing import Type

from pisces.utilities._typing import Instance
from pisces.utilities.config import pisces_params

# @@ SETTING UP LOGGERS @@ #
# We load streams, formatters, and handlers from the configuration
# and load them dynamically.
streams = dict(
    mylog=getattr(sys, pisces_params["logging.mylog.stream"]),
    devlog=getattr(sys, pisces_params["logging.devlog.stream"]),
)
_loggers = dict(mylog=logging.Logger("Pisces"), devlog=logging.Logger("PISCES-DEV"))

_handlers = {}

for k, v in _loggers.items():
    # Construct the formatter string.
    _handlers[k] = logging.StreamHandler(streams[k])
    _handlers[k].setFormatter(logging.Formatter(pisces_params[f"logging.{k}.format"]))

    v.addHandler(_handlers[k])
    v.setLevel(pisces_params[f"logging.{k}.level"])
    v.propagate = False

    if k != "mylog":
        v.disabled = not pisces_params[f"logging.{k}.enabled"]

# Core logger instances.
mylog: logging.Logger = _loggers["mylog"]
""":py:class:`logging.Logger`: The main logger for ``Pisces``.

:py:attr:`mylog`: provides the core stdout and stderr passing for Pisces and should generally be turned on. The
convention followed here is that the levels correspond to the following:

- ``DEBUG``: Effectively verbose mode for users who need to do their own debugging / get a finer
  idea of what's going on. This should not be overloaded with super fine logging.
- ``INFO``: Standard output for the user to see, including messages and print-like statements.
- ``WARNING``: Any relevant warnings for the user to be concerned with.
- ``ERROR``: Any relevant errors for the user to be concerned with.
- ``CRITICAL``: Any relevant errors for the user to be concerned with.

The default level of this logger can be configured in the configuration file.
"""
devlog: logging.Logger = _loggers["devlog"]
""":py:class:`logging.Logger`: The development logger for ``Pisces``.

:py:attr:`devlog`: provides development tracking and should generally be turned off. The convention
followed here is that the levels correspond to the following:

- ``DEBUG``: Extremely fine grained information for complex areas of the code base. There should be no restrictions
  on how frequently this is called to.
- ``INFO``: Developer notes / updates on runtimes and efficiency for specific processes.
- ``WARNING``: Any relevant warnings for the developer to be concerned with.
- ``ERROR``: Any relevant errors for the developer to be concerned with.
- ``CRITICAL``: Any relevant errors for the developer to be concerned with.

The default level of this logger can be configured in the configuration file.
"""


class LogDescriptor:
    """
    A descriptor for dynamically creating and managing loggers for a class.

    At its core, the :py:class:`LogDescriptor` is used for classes like :py:class:`~pisces.models.base.Model` to
    create a class-specific logger.
    """

    def __get__(self, instance: Instance, owner: Type[Instance]) -> logging.Logger:
        # Retrieve or create a logger for the class
        logger = logging.getLogger(owner.__name__)

        if not logger.handlers:
            # Create and configure the logger if not already set up
            handler = logging.StreamHandler(
                getattr(sys, pisces_params["logging.mylog.stream"])
            )
            handler.setFormatter(
                logging.Formatter(pisces_params["logging.code.format"])
            )
            logger.addHandler(handler)
            logger.setLevel(pisces_params["logging.code.level"])
            logger.propagate = False
            logger.disabled = not pisces_params["logging.code.enabled"]

        return logger


# Defining special exceptions and other error raising entities
class ErrorGroup(Exception):
    """
    Exception class for aggregating multiple errors.

    Attributes:
    -----------
    - `errors`: List of exceptions contained in the group.
    - `message`: String representation of the aggregated error messages.

    Methods:
    --------
    - `__len__`: Returns the number of errors in the group.
    - `__str__`: Returns a formatted string representation of the errors.

    Parameters:
    -----------
    - `message` (str): A general description of the error group.
    - `error_list` (list[Exception]): A list of exceptions to include in the group.
    """

    def __init__(self, message: str, error_list: list[Exception]):
        self.errors: list[Exception] = error_list

        # Create the formatted message
        self.message = f"{len(self.errors)} ERRORS: {message}\n\n"

        for err_id, err in enumerate(self.errors):
            self.message += "%(err_id)-5s[%(err_type)10s]: %(msg)s\n" % {
                "err_id": err_id + 1,
                "err_type": err.__class__.__name__,
                "msg": str(err),
            }

    def __len__(self):
        """Returns the number of errors in the group."""
        return len(self.errors)

    def __str__(self):
        """Returns the formatted string representation of the error group."""
        return self.message
