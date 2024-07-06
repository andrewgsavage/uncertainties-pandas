from __future__ import annotations

import numpy
import pandas as pd
import pandas.api.extensions
from pandas.api.types import is_dtype_equal, is_list_like, is_scalar, pandas_dtype

from uncertainties import UFloat, ufloat, ufloat_fromstr, umath, unumpy

import numpy as np
from pandas.core import (
    ops,
)

from pandas.core.construction import (
    extract_array,
)
from pandas.core.arrays._mixins import NDArrayBackedExtensionArray
from pandas._libs import lib


@pandas.api.extensions.register_extension_dtype
class UncertaintyDtype(pandas.api.extensions.ExtensionDtype):
    """
    Extension dtype for uncertainty objects.
    """

    na_value = pd.NA
    kind = "f"
    names = None
    name = "UncertaintyDtype"
    type = UFloat
    itemsize = 64

    @property
    def _is_numeric(self) -> bool:
        return True

    @property
    def _is_boolean(self) -> bool:
        return False

    @classmethod
    def construct_from_string(cls, name: str):
        if not isinstance(name, str):
            raise TypeError(
                f"'construct_from_string' expects a string, got {type(name)}"
            )

        if name != cls.name:
            raise TypeError(f"Cannot construct a '{cls.__name__}' from 'another_type'")

        return cls()

    def construct_array_type(self):
        return UncertaintyArray

    def __repr__(self):
        """
        Return a string representation for this object.

        Invoked by unicode(df) in py2 only. Yields a Unicode String in both
        py2/py3.
        """

        return self.name


def _isna(obj):
    if isinstance(obj, UFloat):
        return umath.isnan(obj.nominal_value)
    return pd.isna(obj)


def _convert_unan_to_pdna(arr):
    """
    Convert ufloat nan to pd.NA
    """
    return np.array([pd.NA if _isna(x) else x for x in arr], dtype=object)


def _convert_pdna_to_nan(arr):
    """
    Convert pd.NA to ufloat nan
    """
    return np.array(
        [ufloat(np.nan, np.nan) if pd.isna(x) else x for x in arr], dtype=object
    )


def _ufloat(value):
    if _isna(value):
        return pd.NA
    if not isinstance(value, UFloat):
        if isinstance(value, tuple):
            return ufloat(*value)
        if isinstance(value, str):
            return ufloat_fromstr(value)
        return ufloat(value)
    return value


class UncertaintyArray(
    pandas.core.arraylike.OpsMixin,
    NDArrayBackedExtensionArray,
):
    dtype = UncertaintyDtype()
    name = "UncertaintyArray"
    # scalar used to denote NA value inside our self._ndarray, e.g. -1 for
    # Categorical, iNaT for Period. Outside of object dtype, self.isna() should
    # be exactly locations in self._ndarray with _internal_fill_value. See:
    # https://github.com/pandas-dev/pandas/blob/main/pandas/core/arrays/_mixins.py
    _internal_fill_value = pd.NA

    def __init__(self, values, dtype=None, copy: bool = False):
        if not (
            isinstance(values, numpy.ndarray)
            and values.dtype == numpy.dtype("object")
            and all(isinstance(v, UFloat) for v in values)
        ):
            values = self.__ndarray(values)
        elif copy:
            values = values.copy()

        values = _convert_unan_to_pdna(values)
        super().__init__(values=values, dtype=values.dtype)

    @classmethod
    def __ndarray(cls, scalars):
        return numpy.array(
            [
                _ufloat(scalar) if not isinstance(scalar, UFloat) else scalar
                for scalar in scalars
            ],
            "object",
        )

    @classmethod
    def _from_sequence(cls, scalars, *, dtype=None, copy=False):
        if dtype is not None:
            assert dtype.__class__ is cls.dtype.__class__
        return cls(cls.__ndarray(scalars))

    _from_sequence_of_strings = _from_sequence

    def astype(self, dtype, copy=True):
        dtype = pandas_dtype(dtype)
        if is_dtype_equal(dtype, self.dtype):
            if not copy:
                return self
            else:
                return self.copy()

        return super().astype(dtype, copy=copy)

    def _from_factorized(self, unique, original):
        return self.__class__(unique)

    def isna(self):
        return np.array([pd.isna(x) for x in self._ndarray], dtype=bool)

    def _validate_scalar(self, value):
        """
        Validate and convert a scalar value to datetime64[ns] for storage in
        backing NumPy array.
        """
        return _ufloat(value)

    def _validate_searchsorted_value(self, value):
        """
        Convert a value for use in searching for a value in the backing numpy array.

        TODO: With pandas 2.0, this may be unnecessary. https://github.com/pandas-dev/pandas/pull/45544#issuecomment-1052809232
        """
        return self._validate_setitem_value(value)

    def __setitem__(self, index, value):
        if is_list_like(value) and is_scalar(index):
            raise ValueError("Length of index must match length of values")
        super().__setitem__(index, value)

    def _validate_setitem_value(self, value):
        """
        Convert a value for use in setting a value in the backing numpy array.
        """
        if is_list_like(value):
            return [_ufloat(v) for v in value]

        return _ufloat(value)

    def _arith_method(self, other, op):
        ops.get_op_result_name(self, other)

        lvalues = self._ndarray
        rvalues = extract_array(other, extract_numpy=True, extract_range=True)
        rvalues = ops.maybe_prepare_scalar_for_op(rvalues, lvalues.shape)
        if isinstance(rvalues, range):
            rvalues = np.arange(rvalues.start, rvalues.stop, rvalues.step)

        # pandas thinks a scalar uncertainty is already an array as it has a dtype
        if op.__name__ == "floordiv" and is_scalar(rvalues):
            rvalues = np.array(rvalues)

        with np.errstate(all="ignore"):
            result = ops.arithmetic_op(lvalues, rvalues, op)

        return self._from_sequence(result, dtype=self.dtype, copy=False)

    def _cmp_method(self, other, op):
        """Compare array values, for use in OpsMixin."""
        if is_scalar(other):
            if not isinstance(other, UFloat):
                other = _ufloat((other, 0))
            other = type(self)([other])

        if type(other) is not type(self):
            return NotImplemented

        oshape = getattr(other, "shape", None)
        if oshape != self.shape and oshape != (1,) and self.shape != (1,):
            raise TypeError(
                "Can't compare arrays with different shapes", self.shape, oshape
            )
        return op(self._ndarray, other._ndarray)

    def __invert__(self) -> UncertaintyArray:
        return type(self)(~self._ndarray)

    def __neg__(self) -> UncertaintyArray:
        return type(self)(-self._ndarray)

    def __pos__(self) -> UncertaintyArray:
        return type(self)(+self._ndarray)

    def __abs__(self) -> UncertaintyArray:
        return type(self)(abs(self._ndarray))

    def _reduce(self, name, *, skipna: bool = True, keepdims: bool = False, **kwds):
        """
        Return a scalar result of performing the reduction operation.

        Parameters
        ----------
        name : str
            Name of the function, supported values are:
            { any, all, min, max, sum, mean, median, prod,
            std, var, sem, kurt, skew }.
        skipna : bool, default True
            If True, skip NaN values.
        **kwargs
            Additional keyword arguments passed to the reduction function.
            Currently, `ddof` is the only supported kwarg.

        Returns
        -------
        scalar

        Raises
        ------
        TypeError : subclass does not define reductions
        """

        functions = {
            "any": any,
            "all": all,
            "min": np.min,
            "max": np.max,
            "sum": np.sum,
            "mean": np.mean,
            "median": np.median,
            "prod": np.prod,
            # "std": np.std,
            # "var": np.var,
            # "sem": np.sem,
            # "kurt": np.kurt,
            # "skew": np.skew,
        }
        if name not in functions:
            raise TypeError(f"cannot perform {name} with type {self.dtype}")

        if "min_count" in kwds and sum(~self.isna()) < kwds["min_count"]:
            result = pd.NA
        else:
            kwds.pop("min_count", None)
            if skipna:
                data = self[~self.isna()]._ndarray
            else:
                data = _convert_pdna_to_nan(self._ndarray)
            result = functions[name](data, **kwds)

        if keepdims:
            return UncertaintyArray([result])
        return result

    # Overide to_numpy so UA.to_numpy(dtype=float) doesn't raise error.
    # required for plotting
    def to_numpy(
        self,
        dtype=None,
        copy: bool = False,
        na_value: object = lib.no_default,
    ) -> np.ndarray:
        """
        Convert to a NumPy ndarray.

        This is similar to :meth:`numpy.asarray`, but may provide additional control
        over how the conversion is done.

        Parameters
        ----------
        dtype : str or numpy.dtype, optional
            The dtype to pass to :meth:`numpy.asarray`.
        copy : bool, default False
            Whether to ensure that the returned value is a not a view on
            another array. Note that ``copy=False`` does not *ensure* that
            ``to_numpy()`` is no-copy. Rather, ``copy=True`` ensure that
            a copy is made, even if not strictly necessary.
        na_value : Any, optional
            The value to use for missing values. The default value depends
            on `dtype` and the type of the array.

        Returns
        -------
        numpy.ndarray
        """
        if dtype == "float":
            data = _convert_pdna_to_nan(self._ndarray)
            result = unumpy.nominal_values(data)
        else:
            result = np.asarray(self, dtype=dtype)
        if copy or na_value is not lib.no_default:
            result = result.copy()
        if na_value is not lib.no_default:
            result[self.isna()] = na_value
        return result
