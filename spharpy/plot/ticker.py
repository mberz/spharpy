from matplotlib.ticker import Locator, Formatter
import spharpy
import numpy as np


class SphericalHarmonicLocator(Locator):

    def __init__(self, offset=1):
        self._offset = offset
        super().__init__()

    def __call__(self):
        dmin, dmax = self.axis.get_data_interval()

        return self.tick_values(dmin, dmax)

    def tick_values(self, vmin, vmax):
        n_min, m = spharpy.spherical.acn2nm(vmin)
        n_max, m = spharpy.spherical.acn2nm(vmax)

        orders = np.arange(n_min, n_max)
        nn_ticks = spharpy.spherical.nm2acn(orders, orders) + self._offset

        return nn_ticks


class SphericalHarmonicFormatter(Formatter):
    """Formatter for ticks representing axes with spherical harmonics.

    Parameters
    ----------
    format : str
        The format used, can be either 'n', 'n0', 'acn', or 'nm'
    """

    def __init__(
            self,
            offset=0,
            format='n',
            **kwargs):

        super().__init__(**kwargs)
        self.format = format
        self._offset = offset

    def __call__(self, x, pos=None):
        if self.format == 'n':
            string = (spharpy.spherical.acn2nm(x)[0]) + self._offset
        elif self.format == 'nm':
            string = x
        elif self.format == 'n0':
            n, m = spharpy.spherical.acn2nm(x)
            string = f'({n},{m})'

        return string
