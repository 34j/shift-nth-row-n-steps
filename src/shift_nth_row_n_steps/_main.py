import warnings

import ivy
from ivy import Array


def _take_slice(input: Array, start: int, end: int, axis: int) -> Array:
    axis = axis % len(input.shape)
    return input[
        (slice(None),) * axis
        + (slice(start, end),)
        + (slice(None),) * (len(input.shape) - axis - 1)
    ]


def shift_nth_row_n_steps_for_loop(
    input: Array, *, dim_row: int = -2, dim_shift: int = -1, cut_padding: bool = False
) -> Array:
    """
    Shifts the nth row n steps to the right.

    Parameters
    ----------
    input : Array
        The input tensor.
    dim_row : int, optional
        The dimension of the row to shift, by default -2
    dim_shift : int, optional
        The dimension of the shift, by default -1
    cut_padding : bool, optional
        Whether to cut additional columns, by default False

    Returns
    -------
    Array
        The shifted tensor. If the input is (..., row, ..., shift, ...),
        the output will be (..., row, ..., shift + row - 1, ...).
        [...,i,...,j,...] -> [...,i,...,j+i,...]

    """
    outputs = []
    row_len = ivy.shape(input)[dim_row]
    for i in range(ivy.shape(input)[dim_row]):
        row = _take_slice(input, i, i + 1, axis=dim_row)
        row_cut = _take_slice(row, 0, row_len - i, axis=dim_shift)
        zero_shape = list(ivy.shape(row))
        zero_shape[dim_shift] = i
        output = ivy.concat([ivy.zeros(zero_shape), row_cut], axis=dim_shift).squeeze(
            axis=dim_row
        )
        outputs.append(output)
    output = ivy.stack(outputs, axis=dim_row)
    return output


def shift_nth_row_n_steps(
    input: Array, *, dim_row: int = -2, dim_shift: int = -1, cut_padding: bool = False
) -> Array:
    """
    Shifts the nth row n steps to the right.

    Parameters
    ----------
    input : Array
        The input tensor.
    dim_row : int, optional
        The dimension of the row to shift, by default -2
    dim_shift : int, optional
        The dimension of the shift, by default -1
    cut_padding : bool, optional
        Whether to cut additional columns, by default False

    Returns
    -------
    Array
        The shifted tensor. If the input is (..., row, ..., shift, ...),
        the output will be (..., row, ..., shift + row - 1, ...).
        [...,i,...,j,...] -> [...,i,...,j+i,...]

    """
    # swap dim_row and -2, dim_shift and -1
    dim_row_ = -2
    dim_shift_ = -1
    input = ivy.moveaxis(input, (dim_row, dim_shift), (dim_row_, dim_shift_))

    shape = ivy.shape(input)
    l_row = shape[dim_row_]
    l_shift = shape[dim_shift_]
    if cut_padding and l_shift < l_row:
        warnings.warn(
            "cut_padding is True, but s < r, which results in redundant computation.",
            stacklevel=2,
        )

    # first pad to [s, r] -> [s+r, r]
    output = ivy.pad(
        input,
        [(0, 0)] * (len(shape) - 1) + [(0, l_row)],
        mode="constant",
        constant_values=0,
    )

    # flatten dim_shift_ to dim_row_
    flatten_shape = list(ivy.shape(output))
    flatten_shape[dim_shift_] = 1
    flatten_shape[dim_row_] = -1
    output = output.reshape(flatten_shape).squeeze(axis=dim_shift_)

    # remove last padding, [(s+r)*r] -> [(s+r-1)*r]
    output = _take_slice(output, 0, (l_shift + l_row - 1) * l_row, axis=dim_shift_)

    # new shape is [s+r-1,r]
    result_shape = list(shape)
    result_shape[dim_shift_] = l_shift + l_row - 1
    output = output.reshape(result_shape)

    # cut padding
    if cut_padding:
        output = _take_slice(output, 0, l_shift, axis=dim_shift_)

    # return the result
    return ivy.moveaxis(output, (dim_row_, dim_shift_), (dim_row, dim_shift))
