from collections.abc import Callable, Sequence
from types import EllipsisType
from typing import Any

from array_api._2024_12 import Array
from array_api_compat import array_namespace


def create_slice(
    ndim: int,
    axis_and_key: Sequence[tuple[int, int | slice | EllipsisType | None]],
    *,
    default: int
    | slice
    | EllipsisType
    | None
    | Callable[[], int | slice | EllipsisType | None] = lambda: slice(None),
) -> tuple[int | slice | EllipsisType | None, ...]:
    """
    Create a slice tuple with default values.

    Parameters
    ----------
    ndim : int
        The number of dimensions.
    axis_and_key : Sequence[tuple[int, int  |  slice  |  EllipsisType  |  None]]
        The axis and key pair.
    default : int | slice | EllipsisType | None, optional
        The default value, by default slice(None,)

    Returns
    -------
    tuple[int | slice | EllipsisType | None, ...]
        The slice tuple.

    """
    if isinstance(default, Callable):  # type: ignore
        default_ = default()  # type: ignore
    else:
        default_ = default
    result = [default_] * ndim
    for axis, key in axis_and_key:
        result[axis] = key
    return tuple(result)


def take_slice(a: Array, start: int, end: int, *, axis: int) -> Array:
    """
    numpy.take() alternative using slices. (faster) similar to torch.narrow().

    Parameters
    ----------
    a : Array
        The source array.
    start : int
        The index of the element to start from.
    end : int
        The index of the element to end at.
    axis : int
        The axis to take the slice from.

    Returns
    -------
    Array
        The sliced array.

    """
    ndim = a.ndim
    axis = axis % ndim
    return a[create_slice(ndim, [(axis, slice(start, end))])]


def narrow(a: Array, start: int, length: int, *, axis: int) -> Array:
    """
    torch.narrow() in xp.

    Parameters
    ----------
    a : Array
        The source array.
    start : int
        The index of the element to start from.
    length : int
        The length of the slice.
    axis : int
        The axis to narrow.

    Returns
    -------
    Array
        The narrowed array.

    """
    return take_slice(a, start, start + length, axis=axis)


def select(a: Array, index: int, *, axis: int) -> Array:
    """
    torch.select() (!= numpy.select()) in xp.

    Parameters
    ----------
    a : Array
        The source array.
    index : int
        The index of the element to select.
    axis : int
        The axis to select from.

    Returns
    -------
    Array
        The selected array.

    """
    ndim = a.ndim
    axis = axis % ndim
    return a[create_slice(ndim, [(axis, index)])]


def advanced_indexing_nan(a: Array, index: Any) -> Array:
    """
    Advanced indexing with NaN.

    Parameters
    ----------
    a : Array
        The source array.
    index : int
        The index of the element to select.

    Returns
    -------
    Array
        The selected array.

    """
    xp = array_namespace(a)
    a = a[xp.nan_to_num(index, 0)]
    a[xp.isnan(index)] = xp.nan
    return a
