import warnings
from typing import Literal

import ivy
from ivy import Array, NativeArray

from ._torch_like import take_slice


def shift_nth_row_n_steps_for_loop(
    a: Array | NativeArray,
    *,
    axis_row: int = -2,
    axis_shift: int = -1,
    cut_padding: bool = False,
) -> Array:
    """
    Shifts the nth row n steps to the right.

    Parameters
    ----------
    a : Array
        The source array.
    axis_row : int, optional
        The axis of the row to shift, by default -2
    axis_shift : int, optional
        The axis of the shift, by default -1
    cut_padding : bool, optional
        Whether to cut additional columns, by default False

    Returns
    -------
    Array
        The shifted array. If the input is (..., row, ..., shift, ...),
        the output will be (..., row, ..., shift + row - 1, ...).
        [...,i,...,j,...] -> [...,i,...,j+i,...]

    """
    outputs = ivy.zeros(
        (ivy.shape(a)[axis_row], ivy.shape(a)[axis_row]), dtype=a.dtype, device=a.device
    )
    row_len = ivy.shape(a)[axis_row]
    for i in range(row_len):
        row = take_slice(a, i, i + 1, axis=axis_row)
        row_cut = take_slice(row, 0, row_len - i, axis=axis_shift)
        outputs[i, i:] = row_cut.squeeze(axis=axis_row)
    return outputs


def shift_nth_row_n_steps(
    a: Array | NativeArray,
    *,
    axis_row: int = -2,
    axis_shift: int = -1,
    cut_padding: bool = False,
    padding_mode: Literal["constant", "wrap"] = "constant",
    padding_constant_values: float = 0,
) -> Array:
    """
    Shifts the nth row n steps to the right.

    Parameters
    ----------
    a : Array
        The source array.
    axis_row : int, optional
        The axis of the row to shift, by default -2
    axis_shift : int, optional
        The axis of the shift, by default -1
    cut_padding : bool, optional
        Whether to cut additional columns, by default False
    padding_mode : Literal["constant", "wrap"], optional
        The padding mode, by default "constant"
    padding_constant_values : float, optional
        The constant value to fill, by default 0
        Only used when padding_mode = "constant"

    Returns
    -------
    Array
        The shifted array. If the input is (..., row, ..., shift, ...),
        the output will be (..., row, ..., shift + row - 1, ...).
        [...,i,...,j,...] -> [...,i,...,j+i,...]

    """
    axis_row_ = -2
    axis_shift_ = -1
    a = ivy.moveaxis(a, (axis_row, axis_shift), (axis_row_, axis_shift_))

    shape = ivy.shape(a)
    l_row = shape[axis_row_]
    l_shift = shape[axis_shift_]
    if cut_padding and l_shift < l_row:
        warnings.warn(
            "cut_padding is True, but s < r, which results in redundant computation.",
            stacklevel=2,
        )

    output = ivy.pad(
        a,
        [(0, 0)] * (len(shape) - 1) + [(0, l_row)],
        mode=padding_mode,
        constant_values=padding_constant_values,
    )

    flatten_shape = list(ivy.shape(output))
    flatten_shape[axis_shift_] = 1
    flatten_shape[axis_row_] = -1
    output = output.reshape(flatten_shape).squeeze(axis=axis_shift_)

    output = take_slice(output, 0, (l_shift + l_row - 1) * l_row, axis=axis_shift_)

    result_shape = list(shape)
    result_shape[axis_shift_] = l_shift + l_row - 1
    output = output.reshape(result_shape)

    if cut_padding:
        output = take_slice(output, 0, l_shift, axis=axis_shift_)

    return ivy.moveaxis(output, (axis_row_, axis_shift_), (axis_row, axis_shift))
