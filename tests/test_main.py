import ivy
import pytest

from shift_nth_row_n_steps._main import (
    select,
    shift_nth_row_n_steps,
    shift_nth_row_n_steps_for_loop,
)

ivy.set_backend("numpy")


def test_shift_nth_row_n_steps_match() -> None:
    input = ivy.array([[[1, 2, 3], [4, 5, 6], [7, 8, 9]]])
    expected = ivy.array([[[1, 2, 3], [0, 4, 5], [0, 0, 7]]])
    assert ivy.array_equal(shift_nth_row_n_steps(input, cut_padding=True), expected)
    assert ivy.array_equal(shift_nth_row_n_steps_for_loop(input), expected)


@pytest.mark.parametrize(
    "shape", [(1, 1, 1), (1, 1, 2), (1, 2, 1), (2, 7, 4), (2, 4, 7)]
)
@pytest.mark.parametrize(
    "axis_row,axis_shift", [(-1, -2), (-2, -1), (-3, -1), (-1, -3)]
)
@pytest.mark.parametrize("cut_padding", [True, False])
def test_shift_nth_row_n_steps(
    shape: tuple[int, ...], axis_row: int, axis_shift: int, cut_padding: bool
) -> None:
    array = ivy.ones(shape)
    shift_nth_row_n_steps(
        array, axis_row=axis_row, axis_shift=axis_shift, cut_padding=cut_padding
    )


@pytest.mark.parametrize("index", [(0, 0), (0, 1), (1, 0), (1, 1), (3, 4)])
@pytest.mark.parametrize(
    "axis_row,axis_shift",
    [(0, 1), (1, 0), (-2, -1), (-1, -2), (-3, -1), (-1, -3), (-2, -3), (-3, -2)],
)
def test_shift_nth_row_n_steps_index(
    index: tuple[int, int], axis_row: int, axis_shift: int
) -> None:
    array = ivy.random.random_uniform(shape=(5, 5, 5))
    res = shift_nth_row_n_steps(
        array, axis_row=axis_row, axis_shift=axis_shift, cut_padding=False
    )
    assert (
        res.shape[axis_shift] == array.shape[axis_shift] + array.shape[axis_row] - 1
    ), f"{array.shape=}, {res.shape=}"
    assert res.shape[:axis_shift] == array.shape[:axis_shift]
    if axis_shift != -1:
        assert res.shape[axis_shift + 1 :] == array.shape[axis_shift + 1 :]
    assert ivy.allclose(
        select(
            select(array, index[0], axis=axis_row).expand_dims(axis=axis_row),
            index[1],
            axis=axis_shift,
        ),
        select(
            select(res, index[0], axis=axis_row).expand_dims(axis=axis_row),
            index[0] + index[1],
            axis=axis_shift,
        ),
    ), f"{array=}, {res=}"
