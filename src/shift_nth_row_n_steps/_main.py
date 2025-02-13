import warnings
from typing import Literal

import ivy
from ivy import Array, NativeArray
from typing_extensions import deprecated

from ._torch_like import create_slice, select, take_slice


@deprecated(
    "This function is too slow thus no longer supported."
    "For debugging purposes, please use "
    "shift_nth_row_n_steps_advanced_indexing instead.",
    category=DeprecationWarning,
)
def shift_nth_row_n_steps_for_loop_assign(
    a: Array | NativeArray,
    *,
    axis_row: int = -2,
    axis_shift: int = -1,
    cut_padding: bool = False,
    mode: Literal["fill"] = "fill",
    fill_values: Literal[0] = 0,
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
    mode : Literal["fill", "roll", "abs"], optional
        The padding mode, by default "constant"
        - fill(padding_mode=constant) -> shift + fill
            (result[i,j] = a[i,j+n_shift*i] if j >= i else fill_values)
        - roll(padding_mode=wrap) -> shift + roll
            (a[i,j] = b[i] then result[i,j] = b[(j+n_shift*i)%len(b)])
        - abs(padding_mode=reflect) -> shift + symmetric
            (a[i,j] = b[i] then result[i,j] = b[abs(j+n_shift*i)]
            not implemented,
            do `result + result.T - result * ivy.eye(result.shape[-1])` instead
            (current behavior aims to support cut_padding = False)
    fill_values : Literal[0], optional
        The constant value to fill, by default 0
        Only used when padding_mode = "constant"

    Returns
    -------
    Array
        The shifted array. If the input is (..., row, ..., shift, ...),
        the output will be (..., row, ..., shift + row - 1, ...).
        [...,i,...,j,...] -> [...,i,...,j+i,...]

    """
    input_shape = list(ivy.shape(a))
    ndim = len(input_shape)
    axis_row = axis_row % ndim
    axis_shift = axis_shift % ndim
    if axis_row == axis_shift:
        raise ValueError("axis_row and axis_shift should not be the same.")
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
            if i >= shift_len:
                break
            output[
                create_slice(
                    ndim, [(axis_row, slice(i, i + 1)), (axis_shift, slice(i, None))]
                )
            ] = take_slice(row, 0, shift_len - i, axis=axis_shift)
        else:
            output[
                create_slice(
                    ndim,
                    [
                        (axis_row, slice(i, i + 1)),
                        (axis_shift, slice(i, i + shift_len)),
                    ],
                )
            ] = row
    return output


@deprecated(
    "This function is too slow thus no longer supported."
    "For debugging purposes, please use "
    "shift_nth_row_n_steps_advanced_indexing instead.",
    category=DeprecationWarning,
)
def shift_nth_row_n_steps_for_loop_concat(
    a: Array | NativeArray,
    *,
    axis_row: int = -2,
    axis_shift: int = -1,
    cut_padding: bool = False,
    mode: Literal["fill"] = "fill",
    fill_values: Literal[0] = 0,
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
    mode : Literal["fill", "roll", "abs"], optional
        The padding mode, by default "constant"
        - fill(padding_mode=constant) -> shift + fill
            (result[i,j] = a[i,j+n_shift*i] if j >= i else fill_values)
        - roll(padding_mode=wrap) -> shift + roll
            (a[i,j] = b[i] then result[i,j] = b[(j+n_shift*i)%len(b)])
        - abs(padding_mode=reflect) -> shift + symmetric
            (a[i,j] = b[i] then result[i,j] = b[abs(j+n_shift*i)]
            not implemented,
            do `result + result.T - result * ivy.eye(result.shape[-1])` instead
            (current behavior aims to support cut_padding = False)
    fill_values : Literal[0], optional
        The constant value to fill, by default 0
        Only used when padding_mode = "constant"

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
            row_cut = take_slice(row, 0, max(0, shift_len - i), axis=axis_shift)
            zero_shape = list(row_shape)
            zero_shape[axis_shift] = min(i, shift_len)
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


def shift_nth_row_n_steps_advanced_indexing(
    a: Array | NativeArray,
    *,
    axis_row: int = -2,
    axis_shift: int = -1,
    cut_padding: bool = False,
    mode: Literal["fill", "roll", "abs"] = "fill",
    fill_values: float = 0,
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
    mode : Literal["fill", "roll", "abs"], optional
        The padding mode, by default "constant"
        - fill(padding_mode=constant) -> shift + fill
            (result[i,j] = a[i,j+n_shift*i] if j >= i else fill_values)
        - roll(padding_mode=wrap) -> shift + roll
            (a[i,j] = b[i] then result[i,j] = b[(j+n_shift*i)%len(b)])
        - abs(padding_mode=reflect) -> shift + symmetric
            (a[i,j] = b[i] then result[i,j] = b[abs(j+n_shift*i)]
            not implemented,
            do `result + result.T - result * ivy.eye(result.shape[-1])` instead
            (current behavior aims to support cut_padding = False)
    fill_values : float, optional
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
    i_row = ivy.arange(shape[axis_row_])[:, None]
    i_shift = (
        ivy.arange(shape[axis_shift_] + (0 if cut_padding else shape[axis_row_] - 1))[
            None, :
        ]
        - i_row
    )
    i_shift = ivy.clip(i_shift, -1, shape[axis_shift_])
    if not cut_padding:
        i_shift = ivy.where(i_shift == shape[axis_shift_], -1, i_shift)
    a = a[
        create_slice(
            len(shape),
            [(axis_row_, i_row), (axis_shift_, i_shift)],
            default=slice(None),
        )
    ]
    a[create_slice(len(shape) - 1, [(-1, i_shift == -1)], default=slice(None))] = 0
    return ivy.moveaxis(a, (axis_row_, axis_shift_), (axis_row, axis_shift))


def shift_nth_row_n_steps(
    a: Array | NativeArray,
    *,
    axis_row: int = -2,
    axis_shift: int = -1,
    cut_padding: bool = False,
    mode: Literal["fill", "roll", "abs"] = "fill",
    fill_values: float = 0,
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
    mode : Literal["fill", "roll", "abs"], optional
        The padding mode, by default "constant"
        - fill(padding_mode=constant) -> shift + fill
            (result[i,j] = a[i,j+n_shift*i] if j >= i else fill_values)
        - roll(padding_mode=wrap) -> shift + roll
            (a[i,j] = b[i] then result[i,j] = b[(j+n_shift*i)%len(b)])
        - abs(padding_mode=reflect) -> shift + symmetric
            (a[i,j] = b[i] then result[i,j] = b[abs(j+n_shift*i)]
            not implemented,
            do `result + result.T - result * ivy.eye(result.shape[-1])` instead
            (current behavior aims to support cut_padding = False)
    fill_values : float, optional
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
            "cut_padding is True, but s < r, "
            "which results in redundant computation.",
            stacklevel=2,
        )

    # first pad to [s, r] -> [s+r, r]
    # if cut_padding, could be [s, r] -> [s+r-1, r]
    # and therefore by mode="reflect", we get symmetric output
    mode_ = {
        "fill": "constant",
        "roll": "wrap",
        "abs": "reflect",
    }[mode]
    output = ivy.pad(
        a,
        [(0, 0)] * (len(shape) - 1) + [(0, l_row)],
        mode=mode_,
        constant_values=fill_values,
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
