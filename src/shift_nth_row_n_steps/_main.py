import warnings

import ivy
from ivy import Array, NativeArray


def take_slice(a: Array | NativeArray, start: int, end: int, *, axis: int) -> Array:
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
    axis = axis % len(a.shape)
    return a[
        (slice(None),) * axis
        + (slice(start, end),)
        + (slice(None),) * (len(a.shape) - axis - 1)
    ]


def narrow(a: Array | NativeArray, start: int, length: int, *, axis: int) -> Array:
    """
    torch.narrow() in ivy.

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


def select(a: Array | NativeArray, index: int, *, axis: int) -> Array:
    """
    torch.select() (!= numpy.select()) in ivy.

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
    axis = axis % len(a.shape)
    return a[
        (slice(None),) * axis + (index,) + (slice(None),) * (len(a.shape) - axis - 1)
    ]


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
        The axisension of the row to shift, by default -2
    axis_shift : int, optional
        The axisension of the shift, by default -1
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
    row_len = ivy.shape(a)[axis_row]
    for i in range(ivy.shape(a)[axis_row]):
        row = take_slice(a, i, i + 1, axis=axis_row)
        row_cut = take_slice(row, 0, row_len - i, axis=axis_shift)
        zero_shape = list(ivy.shape(row))
        zero_shape[axis_shift] = i
        output = ivy.concat([ivy.zeros(zero_shape), row_cut], axis=axis_shift).squeeze(
            axis=axis_row
        )
        outputs.append(output)
    output = ivy.stack(outputs, axis=axis_row)
    return output


def shift_nth_row_n_steps(
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
    output = ivy.pad(
        a,
        [(0, 0)] * (len(shape) - 1) + [(0, l_row)],
        mode="constant",
        constant_values=0,
    )

    # flatten axis_shift_ to axis_row_
    flatten_shape = list(ivy.shape(output))
    flatten_shape[axis_shift_] = 1
    flatten_shape[axis_row_] = -1
    output = output.reshape(flatten_shape).squeeze(axis=axis_shift_)

    # remove last padding, [(s+r)*r] -> [(s+r-1)*r]
    output = take_slice(output, 0, (l_shift + l_row - 1) * l_row, axis=axis_shift_)

    # new shape is [s+r-1,r]
    result_shape = list(shape)
    result_shape[axis_shift_] = l_shift + l_row - 1
    output = output.reshape(result_shape)

    # cut padding
    if cut_padding:
        output = take_slice(output, 0, l_shift, axis=axis_shift_)

    # return the result
    return ivy.moveaxis(output, (axis_row_, axis_shift_), (axis_row, axis_shift))
