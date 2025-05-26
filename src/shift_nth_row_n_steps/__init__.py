__version__ = "1.0.0-rc.1"
from ._main import shift_nth_row_n_steps
from ._torch_like import create_slice, narrow, select, take_slice

__all__ = ["shift_nth_row_n_steps", "narrow", "select", "take_slice", "create_slice"]
