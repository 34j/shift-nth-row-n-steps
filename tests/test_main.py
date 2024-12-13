import importlib.util
from unittest import SkipTest

import ivy
import pytest
from ivy_tests.test_ivy.helpers.assertions import assert_all_close

from shift_nth_row_n_steps._main import (
    shift_nth_row_n_steps,
    shift_nth_row_n_steps_advanced_indexing,
    shift_nth_row_n_steps_for_loop_assign,
    shift_nth_row_n_steps_for_loop_concat,
)
from shift_nth_row_n_steps._torch_like import select


@pytest.fixture(autouse=True, params=["numpy", "jax", "torch"], scope="session")
def setup(request: pytest.FixtureRequest) -> None:
    if importlib.util.find_spec(request.param) is None:
        raise SkipTest(f"{request.param} is not installed")
    ivy.set_backend(request.param)


@pytest.mark.parametrize("cut_padding", [True, False])
def test_shift_nth_row_n_steps_manual_match(cut_padding: bool) -> None:
    input = ivy.array([[[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]]])
    if cut_padding:
        expected = ivy.array([[[1, 2, 3, 4], [0, 5, 6, 7], [0, 0, 9, 10]]])
    else:
        expected = ivy.array(
            [[[1, 2, 3, 4, 0, 0], [0, 5, 6, 7, 8, 0], [0, 0, 9, 10, 11, 12]]]
        )
    assert_all_close(
        shift_nth_row_n_steps(input, cut_padding=cut_padding),
        expected,
        ivy.current_backend_str,
    )
    assert_all_close(
        shift_nth_row_n_steps_for_loop_concat(input, cut_padding=cut_padding),
        expected,
        ivy.current_backend_str,
    )
    assert_all_close(
        shift_nth_row_n_steps_advanced_indexing(input, cut_padding=cut_padding),
        expected,
        ivy.current_backend_str,
    )
    assert_all_close(
        shift_nth_row_n_steps_for_loop_assign(input, cut_padding=cut_padding),
        expected,
        ivy.current_backend_str,
    )


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
    funcs = [
        shift_nth_row_n_steps,
        shift_nth_row_n_steps_for_loop_concat,
        shift_nth_row_n_steps_for_loop_assign,
        shift_nth_row_n_steps_advanced_indexing,
    ]
    results = [
        func(  # type: ignore
            array,
            axis_row=axis_row,
            axis_shift=axis_shift,
            cut_padding=cut_padding,
            # padding_constant_values=0.1,
        )
        for func in funcs
    ]
    for i in range(len(results) - 1):
        assert_all_close(results[i], results[i + 1], ivy.current_backend_str)


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
    assert_all_close(
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
        ivy.current_backend_str,
    )


def test_custom_padding() -> None:
    raise SkipTest("Not implemented yet")
    n = 4  # type: ignore
    array = ivy.random.random_uniform(shape=(n,)).expand_dims(axis=0).repeat(n, axis=0)
    res_const = shift_nth_row_n_steps(
        array,
        axis_row=-2,
        axis_shift=-1,
        cut_padding=True,
        padding_mode="constant",
    )
    res_const = res_const + res_const.T - res_const * ivy.eye(res_const.shape[-1])
    res_wrap = shift_nth_row_n_steps(
        array, axis_row=-2, axis_shift=-1, cut_padding=True, padding_mode="reflect"
    )
    assert_all_close(res_const, res_wrap, ivy.current_backend_str)
