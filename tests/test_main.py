import ivy

from shift_nth_row_n_steps._main import (
    shift_nth_row_n_steps,
    shift_nth_row_n_steps_for_loop,
)

ivy.set_backend("numpy")


def test_shift_nth_row_n_steps() -> None:
    input = ivy.array([[[1, 2, 3], [4, 5, 6], [7, 8, 9]]])
    expected = ivy.array([[[1, 2, 3], [0, 4, 5], [0, 0, 7]]])
    assert ivy.array_equal(shift_nth_row_n_steps(input, cut_padding=True), expected)
    assert ivy.array_equal(shift_nth_row_n_steps_for_loop(input), expected)
