import numpy as np
import typer
from cm_time import timer
from rich import print

from ._main import (
    shift_nth_row_n_steps,
    shift_nth_row_n_steps_advanced_indexing,
)

app = typer.Typer()


@app.command()
def benchmark(
    backend: str = typer.Option("numpy"),
    device: str = typer.Option("cpu"),
    dtype: str = typer.Option("float32"),
    n_end: int = typer.Option(10),
) -> None:
    """
    Benchmark the two implementations of the function.

    Parameters
    ----------
    backend : str, optional
        The backend to use, by default typer.Option("numpy")
    device : str, optional
        The device to use, by default typer.Option("cpu")
    dtype : str, optional
        The dtype to use, by default typer.Option("float32")
    n_end : int, optional
        The maximum power of 2 to use, by default typer.Option(10)

    """
    if device == "gpu":
        device = "gpu:0"
    elif device == "tpu":
        device = "tpu:0"
    if backend == "numpy":
        import numpy as xp
    elif backend == "jax":
        import jax.numpy as xp
    elif backend == "torch":
        import torch as xp
    for i in range(n_end):
        n = 2**i
        rng = np.random.default_rng()
        input = rng.uniform(size=(n, n))
        input = xp.asarray(input, dtype=dtype)
        with timer() as t1:
            shift_nth_row_n_steps(input)
        with timer() as t2:
            shift_nth_row_n_steps_advanced_indexing(input)
        print(f"{n}: propsed: {t1.elapsed:g}, advanced indexing: {t2.elapsed:g}")
