import warnings
from typing import Literal

import ivy
from ivy import Array, NativeArray

from ._torch_like import create_slice, select, take_slice


def shift_nth_row_n_steps_for_loop_assign(
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
    input_shape = list(ivy.shape(a))
    ndim = len(input_shape)
    row_len = input_shape[axis_row]
    shift_len = input_shape[axis_shift]

    if cut_padding:
        output = ivy.zeros_like(a)
    else:
        output_shape = list(input_shape)
        output_shape[axis_shift] = row_len + shift_len - 1
        output = ivy.zeros(output_shape, dtype=a.dtype, device=a.device)

    for i in range(row_len):
        row = take_slice(a, i, i + 1, axis=axis_row)
        if cut_padding:
            output[
                create_slice(ndim, [(axis_row, i), (axis_shift, slice(i, None))])
            ] = take_slice(row, 0, shift_len - i, axis=axis_shift)
        else:
            output[
                create_slice(
                    ndim, [(axis_row, i), (axis_shift, slice(i, i + shift_len))]
                )
            ] = row
    return output


def shift_nth_row_n_steps_for_loop_concat(
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
    outputs = []
    input_shape = list(ivy.shape(a))
    row_len = input_shape[axis_row]
    shift_len = input_shape[axis_shift]
    for i in range(row_len):
        row = take_slice(a, i, i + 1, axis=axis_row)
        row_shape = ivy.shape(row)
        if cut_padding:
            row_cut = take_slice(row, 0, shift_len - i, axis=axis_shift)
            zero_shape = list(row_shape)
            zero_shape[axis_shift] = i
            output = ivy.concat(
                [ivy.zeros(zero_shape, dtype=a.dtype, device=a.device), row_cut],
                axis=axis_shift,
            ).squeeze(axis=axis_row)
        else:
            zero_shape_left = list(row_shape)
            zero_shape_left[axis_shift] = i
            zero_shape_right = list(row_shape)
            zero_shape_right[axis_shift] = row_len - 1 - i
            output = ivy.concat(
                [
                    ivy.zeros(zero_shape_left, dtype=a.dtype, device=a.device),
                    row,
                    ivy.zeros(zero_shape_right, dtype=a.dtype, device=a.device),
                ],
                axis=axis_shift,
            ).squeeze(axis=axis_row)
        outputs.append(output)
    output = ivy.stack(outputs, axis=axis_row)
    return output


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
        - constant -> shift + fill
            (result[i,j] = a[i,j-i] if j >= i else padding_constant_values)
        - wrap -> shift + roll
            (a[i,j] = b[i] then result[i,j] = b[(j-i)%len(b)])
        - reflect -> shift + symmetric
            (a[i,j] = b[i] then result[i,j] = b[abs(j-i)]
            not implemented,
            do `result + result.T - result * ivy.eye(result.shape[-1])` instead
            (current behavior aims to support cut_padding = False)
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
    # swap axis_row and -2, axis_shift and -1
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

    # first pad to [s, r] -> [s+r, r]
    # if cut_padding, could be [s, r] -> [s+r-1, r]
    # and therefore by mode="reflect", we get symmetric output
    output = ivy.pad(
        a,
        [(0, 0)] * (len(shape) - 1) + [(0, l_row)],
        mode=padding_mode,
        constant_values=padding_constant_values,
    )

    # flatten axis_shift_ to axis_row_
    flatten_shape = list(ivy.shape(output))
    flatten_shape[axis_shift_] = 1
    flatten_shape[axis_row_] = -1
    output = output.reshape(flatten_shape)
    output = select(output, 0, axis=axis_shift_)

    # remove last padding, [(s+r)*r] -> [(s+r-1)*r]
    output = take_slice(output, 0, (l_shift + l_row - 1) * l_row, axis=axis_shift_)

    # new shape is [s+r-1,r]
    result_shape = list(shape)
    result_shape[axis_shift_] = l_shift + l_row - 1
    output = ivy.reshape(output, result_shape)

    # cut padding
    if cut_padding:
        output = take_slice(output, 0, l_shift, axis=axis_shift_)

    # return the result
    return ivy.moveaxis(output, (axis_row_, axis_shift_), (axis_row, axis_shift))
