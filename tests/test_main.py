from math import prod
from typing import Any, Literal

import array_api_extra as xpx
import pytest
from array_api._2024_12 import ArrayNamespace
from array_api_compat import numpy, torch

from shift_nth_row_n_steps._main import (
    shift_nth_row_n_steps,
    shift_nth_row_n_steps_advanced_indexing,
    shift_nth_row_n_steps_for_loop_assign,
    shift_nth_row_n_steps_for_loop_concat,
)
from shift_nth_row_n_steps._torch_like import select


@pytest.mark.parametrize("xp", [numpy, torch])
@pytest.mark.parametrize("cut_padding", [True, False])
@pytest.mark.parametrize("mode", ["fill", "roll", "abs"])
@pytest.mark.parametrize(
    "func",
    [
        shift_nth_row_n_steps,
        shift_nth_row_n_steps_advanced_indexing,
        # shift_nth_row_n_steps_for_loop_assign,
        # shift_nth_row_n_steps_for_loop_concat,
    ],
)
def test_shift_nth_row_n_steps_manual_match(
    cut_padding: bool,
    mode: Literal["fill", "roll", "abs"],
    func: Any,
    xp: ArrayNamespace,
) -> None:
    input = xp.asarray([[[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]]])
    if mode == "fill":
        expected = xp.asarray(
            [[[1, 2, 3, 4, 0, 0], [0, 5, 6, 7, 8, 0], [0, 0, 9, 10, 11, 12]]]
        )
    elif mode == "roll":
        expected = xp.asarray(
            [[[1, 2, 3, 4, 1, 2], [3, 5, 6, 7, 8, 5], [6, 7, 9, 10, 11, 12]]]
        )
    elif mode == "abs":
        expected = xp.asarray(
            [[[1, 2, 3, 4, 3, 2], [1, 5, 6, 7, 8, 7], [6, 5, 9, 10, 11, 12]]]
        )
    if cut_padding:
        expected = expected[:, :, :4]
    if mode != "fill" and func in [
        shift_nth_row_n_steps_for_loop_assign,
        shift_nth_row_n_steps_for_loop_concat,
        shift_nth_row_n_steps_advanced_indexing,
    ]:
        pytest.skip("Not implemented")

    assert xp.all(
        xpx.isclose(
            func(input, cut_padding=cut_padding, mode=mode, fill_values=0),
            expected,
        )
    )


@pytest.mark.parametrize("xp", [numpy, torch])
@pytest.mark.parametrize(
    "shape", [(1, 1, 1), (1, 1, 2), (1, 2, 1), (2, 7, 4), (2, 4, 7)]
)
@pytest.mark.parametrize(
    "axis_row,axis_shift", [(-1, -2), (-2, -1), (-3, -1), (-1, -3)]
)
@pytest.mark.parametrize("cut_padding", [True, False])
def test_shift_nth_row_n_steps(
    shape: tuple[int, ...],
    axis_row: int,
    axis_shift: int,
    cut_padding: bool,
    xp: ArrayNamespace,
) -> None:
    array = xp.reshape(xp.arange(prod(shape)), shape)
    funcs = [
        shift_nth_row_n_steps,
        # shift_nth_row_n_steps_for_loop_concat,
        # shift_nth_row_n_steps_for_loop_assign,
        shift_nth_row_n_steps_advanced_indexing,
    ]
    results = [
        func(
            array,
            axis_row=axis_row,
            axis_shift=axis_shift,
            cut_padding=cut_padding,
            # padding_constant_values=0.1,
        )
        for func in funcs
    ]
    for i in range(len(results) - 1):
        assert xp.all(xpx.isclose(results[i], results[i + 1]))


@pytest.mark.parametrize("xp", [numpy, torch])
@pytest.mark.parametrize("index", [(0, 0), (0, 1), (1, 0), (1, 1), (3, 4)])
@pytest.mark.parametrize(
    "axis_row,axis_shift",
    [(0, 1), (1, 0), (-2, -1), (-1, -2), (-3, -1), (-1, -3), (-2, -3), (-3, -2)],
)
def test_shift_nth_row_n_steps_index(
    index: tuple[int, int], axis_row: int, axis_shift: int, xp: ArrayNamespace
) -> None:
    shape = (5, 6, 7)
    array = xp.reshape(xp.arange(prod(shape)), shape)
    res = shift_nth_row_n_steps(
        array, axis_row=axis_row, axis_shift=axis_shift, cut_padding=False
    )
    assert (
        res.shape[axis_shift] == array.shape[axis_shift] + array.shape[axis_row] - 1
    ), f"{array.shape=}, {res.shape=}"
    assert res.shape[:axis_shift] == array.shape[:axis_shift]
    if axis_shift != -1:
        assert res.shape[axis_shift + 1 :] == array.shape[axis_shift + 1 :]
    assert xp.all(
        xpx.isclose(
            select(
                xpx.expand_dims(select(array, index[0], axis=axis_row), axis=axis_row),
                index[1],
                axis=axis_shift,
            ),
            select(
                xpx.expand_dims(select(res, index[0], axis=axis_row), axis=axis_row),
                index[0] + index[1],
                axis=axis_shift,
            ),
        )
    )


# def test_custom_padding() -> None:
#     raise SkipTest("Not implemented yet")
#     n = 4  # type: ignore
#     array = xp.random.random_uniform(shape=(n,)).expand_dims(axis=0).repeat(n, axis=0)
#     res_const = shift_nth_row_n_steps(
#         array,
#         axis_row=-2,
#         axis_shift=-1,
#         cut_padding=True,
#         padding_mode="constant",
#     )
#     res_const = res_const + res_const.T - res_const * xp.eye(res_const.shape[-1])
#     res_wrap = shift_nth_row_n_steps(
#         array, axis_row=-2, axis_shift=-1, cut_padding=True, padding_mode="reflect"
#     )
#     assert xp.all(xpx.isclose(res_const, res_wrap))
