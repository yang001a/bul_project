"""Paper-defined case-level global sensitivity bounds."""

from __future__ import annotations

from ._validation import validate_l_max, validate_t_max
from .models import SensitivityBounds


def calculate_sensitivity(l_max: int, t_max: float) -> SensitivityBounds:
    """Calculate the paper-defined conservative vector L1 sensitivities.

    Input
    -----
    l_max:
        Maximum retained events per case after ``clip_trace``.
    t_max:
        Maximum retained seconds per adjacent transition.

    Output
    ------
    SensitivityBounds
        ``Delta_activity=Lmax``, ``Delta_edge=Lmax-1``,
        ``Delta_variant=1``, ``Delta_ST=(Lmax-1)*Tmax``, and
        ``Delta_QT=(Lmax-1)*Tmax^2``.

    Paper experiment step
    ---------------------
    Overall Step 5: use case-level add/remove neighboring logs and the paper's
    conservative vector L1 sensitivity upper bounds. No alternative or
    per-coordinate sensitivity formula is introduced here.
    """

    length_bound = validate_l_max(l_max)
    time_bound = validate_t_max(t_max)
    edge_bound = length_bound - 1
    return SensitivityBounds(
        activity=float(length_bound),
        edge=float(edge_bound),
        variant=1.0,
        time=float(edge_bound) * time_bound,
        time_squared=float(edge_bound) * time_bound**2,
        l_max=length_bound,
        t_max=time_bound,
    )
