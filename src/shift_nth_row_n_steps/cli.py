import ivy
import typer
from cm_time import timer
from rich import print

from ._main import shift_nth_row_n_steps, shift_nth_row_n_steps_for_loop

app = typer.Typer()

ivy.set_backend("torch")


@app.command()
def benchmark() -> None:
    """Add the arguments and print the result."""
    for n in 2 ** ivy.arange(2, 10):
        input = ivy.random.random_uniform(shape=(n, n))
        with timer() as t1:
            shift_nth_row_n_steps(input)
        with timer() as t2:
            shift_nth_row_n_steps_for_loop(input)
        print(f"{n}: propsed: {t1.elapsed:g}, for loop: {t2.elapsed:g}")
