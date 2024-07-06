import pprint

from .uncertainty_array import UncertaintyDtype, UncertaintyArray

try:
    from importlib.metadata import version
except ImportError:
    # Backport for Python < 3.8
    from importlib_metadata import version  # type: ignore

try:  # pragma: no cover
    __version__ = version("uncertainties_pandas")
except Exception:  # pragma: no cover
    # we seem to have a local copy not installed without setuptools
    # so the reported version will be unknown
    __version__ = "unknown"

__all__ = ["UncertaintyArray", "UncertaintyDtype", "__version__"]


def show_versions():
    deps = [
        "uncertainties_pandas",
        "uncertainties",
        "pandas",
        "numpy",
    ]

    versions = {dep: version(dep) for dep in deps}
    pprint.pprint(versions)
