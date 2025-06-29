import warnings

import numpy as np
import seaborn as sns
import typer
from aquarel import load_theme
from array_api_compat import numpy, torch
from cm_time import timer
from pandas import DataFrame
from rich import print

from ._main import (
    shift_nth_row_n_steps,
    shift_nth_row_n_steps_advanced_indexing,
)

app = typer.Typer()


@app.command()
def benchmark(
    dtype: str = typer.Option("float32"),
    n_end: int = typer.Option(10),
    n_iter: int = typer.Option(10),
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
    n_iter : int, optional
        The number of iterations to run, by default typer.Option(10)

    """
    res = []
    for name, xp in [
        ("numpy", numpy),
        ("torch", torch),
        # ("jax", jnp),
    ]:
        for device in ["cpu", "cuda"]:
            # if xp == jnp:
            #     device = jax.devices(device)[0]
            for i in range(n_end):
                try:
                    n = 2**i
                    rng = np.random.default_rng()
                    input = rng.uniform(size=(n, n))
                    input = xp.asarray(input, dtype=getattr(xp, dtype), device=device)
                    with timer() as t1:
                        for _ in range(n_iter):
                            input = shift_nth_row_n_steps(input, cut_padding=True)
                    with timer() as t2:
                        for _ in range(n_iter):
                            input = shift_nth_row_n_steps_advanced_indexing(input)
                    print(
                        f"{n}: propsed: {t1.elapsed:g}, "
                        f"advanced indexing: {t2.elapsed:g}"
                    )
                    res.append(
                        {
                            "n": n,
                            "type": "Proposed Method",
                            "time": t1.elapsed / n_iter,
                            "backend": name,
                            "device": device,
                        }
                    )
                    res.append(
                        {
                            "n": n,
                            "type": "Advanced Indexing",
                            "time": t2.elapsed / n_iter,
                            "backend": name,
                            "device": device,
                        }
                    )
                except Exception as e:
                    warnings.warn(
                        f"Failed to run on {name} with {device} for n={n}: {e}",
                        stacklevel=2,
                    )
    df = DataFrame(res)
    # df = pd.pivot_table(
    #     df,
    #     index="n",
    #     columns=["backend", "device"],
    #     values="time",
    #     aggfunc="mean",
    # )
    # df.plot(ax=ax, logx=True, logy=True)
    theme = load_theme("boxy_dark")
    theme.apply()
    g = sns.FacetGrid(df, row="backend", col="device", margin_titles=True)
    g.map_dataframe(
        sns.lineplot, x="n", y="time", hue="type", style="type", markers=["o", "X"]
    ).set(xscale="log", yscale="log", xlabel="Matrix size", ylabel="Time (s)")
    g.add_legend()
    g.savefig("benchmark.webp", dpi=300)
