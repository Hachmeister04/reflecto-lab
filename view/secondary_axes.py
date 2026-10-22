import numpy as np
import pyqtgraph as pg
import scipy.constants as cst

# O-mode cutoff: f_pe = f_probe, and n_e = 4*pi^2*eps0*m_e*f^2/e^2
NE_PER_HZ2 = 4 * np.pi**2 * cst.epsilon_0 * cst.m_e / cst.e**2
DENSITY_SCALE = 1e19    # densities shown in 1e19 m^-3, as in the profile plot
FREQUENCY_SCALE = 1e9   # plasma frequencies shown in GHz


def frequency_to_density(f):
    """O-mode cutoff density (m^-3) for probing frequency f (Hz)."""
    return NE_PER_HZ2 * np.asarray(f, dtype=float)**2


def density_to_frequency(ne):
    """Plasma frequency (Hz) for density ne (m^-3)."""
    return np.sqrt(np.clip(np.asarray(ne, dtype=float), 0, None) / NE_PER_HZ2)


class MappedAxis(pg.AxisItem):
    """Secondary axis labelling a monotonic function of the plot coordinate.

    The view coordinate is untouched; ticks are chosen at round values of the
    mapped quantity and placed at the matching view coordinate, so their
    spacing is non-linear. Both maps must be increasing for coordinates >= 0.
    """

    def __init__(self, orientation, to_display, from_display, **kwargs):
        super().__init__(orientation, **kwargs)
        self._to_display = to_display
        self._from_display = from_display
        # The axis carries no units, so pyqtgraph must not rescale the view range
        self.enableAutoSIPrefix(False)
        self._display_spacing = {}

    def tickValues(self, minVal, maxVal, size):
        lo, hi = sorted((minVal, maxVal))
        lo = max(lo, 0.0)
        if hi <= lo:
            return []

        d_lo = float(self._to_display(lo))
        d_hi = float(self._to_display(hi))

        levels = []
        self._display_spacing = {}
        for spacing, values in super().tickValues(d_lo, d_hi, size):
            values = np.array([v for v in values if d_lo <= v <= d_hi])
            if len(values) == 0:
                continue
            positions = self._from_display(values)
            # tickStrings only receives the view-coordinate spacing, so remember the display one
            view_spacing = float(np.min(np.diff(positions))) if len(positions) > 1 else hi - lo
            self._display_spacing[view_spacing] = spacing
            levels.append((view_spacing, list(positions)))
        return levels

    def tickStrings(self, values, scale, spacing):
        display = self._to_display(np.asarray(values, dtype=float))
        display_spacing = self._display_spacing.get(spacing)
        if not display_spacing or display_spacing <= 0:
            return [f'{v:g}' for v in display]
        decimals = max(0, int(np.ceil(-np.log10(display_spacing))))
        return [f'{v:.{decimals}f}' for v in display]


class CutoffDensityAxis(MappedAxis):
    """Probing frequency (Hz) axis labelled with the O-mode cutoff density (1e19 m^-3)."""

    def __init__(self, orientation='top', **kwargs):
        super().__init__(
            orientation,
            to_display=lambda f: frequency_to_density(f) / DENSITY_SCALE,
            from_display=lambda ne: density_to_frequency(ne * DENSITY_SCALE),
            **kwargs,
        )


class PlasmaFrequencyAxis(MappedAxis):
    """Density (1e19 m^-3) axis labelled with the plasma frequency (GHz)."""

    def __init__(self, orientation='right', **kwargs):
        super().__init__(
            orientation,
            to_display=lambda ne: density_to_frequency(ne * DENSITY_SCALE) / FREQUENCY_SCALE,
            from_display=lambda f: frequency_to_density(f * FREQUENCY_SCALE) / DENSITY_SCALE,
            **kwargs,
        )
