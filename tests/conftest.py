"""
Configuration for Pisces unit testing structure.
"""
from pisces.utilities.config import pisces_params
import pytest
import os
from _pytest.config.argparsing import Parser
from _pytest.main import Session
from _pytest.nodes import Item

# @@ CONFIGURING SETTINGS @@ #
# The progress bars need to be disabled during unit testing
# because github actions console will not emulate them correctly.

# @@ PYTEST OPTIONS CONFIG @@ #
def pytest_addoption(parser: Parser) -> None:
    """
    Add custom command-line options to pytest for controlling test behavior.

    Args:
        parser (Parser): The pytest parser object.

    Returns:
        None
    """
    parser.addoption("--answer_dir", help="Directory where answers are stored.")
    parser.addoption(
        "--answer_store",
        action="store_true",
        help="Generate new answers, but don't test.",
    )
    parser.addoption("--tmp", help="The temporary directory to use.", default=None)
    parser.addoption(
        "--package",
        help="Package the results of this test for use externally.",
        action="store_true",
    )

# @@ SESSION FIXTURES @@ #
# These are the core fixtures which are present in every session of
# pytest.
@pytest.fixture()
def answer_store(request) -> bool:
    """fetches the ``--answer_store`` option."""
    return request.config.getoption("--answer_store")


@pytest.fixture()
def answer_dir(request) -> str:
    """fetches the ``--answer_dir`` option."""
    ad = os.path.abspath(request.config.getoption("--answer_dir"))
    if not os.path.exists(ad):
        os.makedirs(ad)
    return ad

@pytest.fixture()
def temp_dir(request) -> str:
    """Pull the temporary directory.

    If this is specified by the user, then it may be a non-temp directory which is not
    wiped after runtime. If not specified, then a temp directory is generated and wiped
    after runtime.
    """
    td = request.config.getoption("--tmp")

    if td is None:
        from tempfile import TemporaryDirectory

        td = TemporaryDirectory()

        yield td.name

        td.cleanup()
    else:
        yield os.path.abspath(td)