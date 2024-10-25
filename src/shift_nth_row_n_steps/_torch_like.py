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
