import importlib.util
from unittest import SkipTest

import pytest

from shift_nth_row_n_steps import narrow, select


@pytest.fixture(autouse=True, scope="session")
def setup() -> None:
    if importlib.util.find_spec("torch") is None:
        raise SkipTest("torch is not installed")


def test_select() -> None:
    import torch

    input = torch.randn(3, 4, 5)
    assert torch.equal(select(input, 0, axis=0), torch.select(input, 0, 0))


def test_narrow() -> None:
    import torch

    input = torch.randn(3, 4, 5)
    assert torch.equal(narrow(input, 1, 2, axis=0), torch.narrow(input, 0, 1, 2))
