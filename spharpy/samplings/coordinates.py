import numpy as np
from spharpy.samplings.helpers import sph2cart
from scipy.spatial import cKDTree
import pyfar


class Coordinates(object):
    """Container class for coordinates in a three-dimensional space, allowing
    for compact representation and convenient conversion into spherical as well
    as geospatial coordinate systems.
    The constructor as well as the internal representation are only
    available in Cartesian coordinates. To create a Coordinates object from
    a set of points in spherical coordinates, please use the
    Coordinates.from_spherical() method.

    """
    def __init__(self, x=None, y=None, z=None):
        """Init coordinates container

        Parameters
        ----------
        x : ndarray, double
            x-coordinate
        y : ndarray, double
            y-coordinate
        z : ndarray, double
            z-coordinate
        """

        super(Coordinates, self).__init__()
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        z = np.asarray(z, dtype=np.float64)

        if not np.shape(x) == np.shape(y) == np.shape(z):
            raise ValueError("Input arrays need to have same dimensions.")

        self._x = x
        self._y = y
        self._z = z

    @property
    def x(self):
        """The x-axis coordinates for each point.
        """
        return self._x

    @x.setter
    def x(self, value):
        self._x = np.asarray(value, dtype=np.float64)

    @property
    def y(self):
        """The y-axis coordinate for each point."""
        return self._y

    @y.setter
    def y(self, value):
        self._y = np.asarray(value, dtype=np.float64)

    @property
    def z(self):
        """The z-axis coordinate for each point."""
        return self._z

    @z.setter
    def z(self, value):
        self._z = np.asarray(value, dtype=np.float64)

    @property
    def radius(self):
        """The radius for each point."""
        return np.sqrt(self.x**2 + self.y**2 + self.z**2)

    @radius.setter
    def radius(self, radius):
        x, y, z = sph2cart(np.asarray(radius, dtype=np.float64),
                           self.elevation,
                           self.azimuth)
        self._x = x
        self._y = y
        self._z = z

    @property
    def azimuth(self):
        """The azimuth angle for each point."""
        return np.mod(np.arctan2(self.y, self.x), 2*np.pi)

    @azimuth.setter
    def azimuth(self, azimuth):
        x, y, z = sph2cart(self.radius,
                           self.elevation,
                           np.asarray(azimuth, dtype=np.float64))
        self._x = x
        self._y = y
        self._z = z

    @property
    def elevation(self):
        """The elevation angle for each point"""
        rad = self.radius
        return np.arccos(self.z/rad)

    @elevation.setter
    def elevation(self, elevation):
        x, y, z = sph2cart(self.radius,
                           np.asarray(elevation, dtype=np.float64),
                           self.azimuth)
        self._x = x
        self._y = y
        self._z = z

    @classmethod
    def from_cartesian(cls, x, y, z):
        """Create a Coordinates class object from a set of points in the
        Cartesian coordinate system.

        Parameters
        ----------
        x : ndarray, double
            x-coordinate
        y : ndarray, double
            y-coordinate
        z : ndarray, double
            z-coordinate
        """
        return Coordinates(x, y, z)

    @classmethod
    def from_spherical(cls, radius, elevation, azimuth):
        """Create a Coordinates class object from a set of points in the
        spherical coordinate system.

        Parameters
        ----------
        radius : ndarray, double
            The radius for each point
        elevation : ndarray, double
            The elevation angle in radians
        azimuth : ndarray, double
            The azimuth angle in radians
        """
        radius = np.asarray(radius, dtype=np.double)
        elevation = np.asarray(elevation, dtype=np.double)
        azimuth = np.asarray(azimuth, dtype=np.double)
        x, y, z = sph2cart(radius, elevation, azimuth)
        return Coordinates(x, y, z)

    @classmethod
    def from_array(cls, values, coordinate_system='cartesian'):
        """Create a Coordinates class object from a set of points given as
        numpy array

        Parameters
        ----------
        values : double, ndarray
            Array with shape Nx3 where N is the number of points.
        coordinate_system : string
            Coordinate convention of the given values.
            Can be Cartesian or spherical coordinates.
        """
        coords = Coordinates()
        if coordinate_system == 'cartesian':
            coords.cartesian = values
        elif coordinate_system == 'spherical':
            coords.spherical = values
        else:
            return ValueError("This coordinate system is not supported.")

        return coords

    @property
    def latitude(self):
        """The latitude angle as used in geospatial coordinates."""
        return np.pi/2 - self.elevation

    @property
    def longitude(self):
        """The longitude angle as used in geospatial coordinates."""
        return np.arctan2(self.y, self.x)

    @property
    def cartesian(self):
        """Cartesian coordinates of all points."""
        return np.vstack((self.x, self.y, self.z))

    @cartesian.setter
    def cartesian(self, value):
        """Cartesian coordinates of all points."""
        self.x = value[0, :]
        self.y = value[1, :]
        self.z = value[2, :]

    @property
    def spherical(self):
        """Spherical coordinates of all points."""
        return np.vstack((self.radius, self.elevation, self.azimuth))

    @spherical.setter
    def spherical(self, value):
        """Cartesian coordinates of all points."""
        x, y, z = sph2cart(value[0, :], value[1, :], value[2, :])
        self.cartesian = np.vstack((x, y, z))

    @property
    def n_points(self):
        """Return number of points stored in the object"""
        return self.x.size

    def merge(self, other):
        """Merge another coordinates objects into this object."""
        data = np.concatenate(
            (self.cartesian, other.cartesian),
            axis=-1
        )
        self.cartesian = data

    def find_nearest_point(self, point):
        """Find the closest Coordinate point to a given Point.
        The search for the nearest point is performed using the scipy
        cKDTree implementation.

        Parameters
        ----------
        point : Coordinates
            Point to find nearest neighboring Coordinate

        Returns
        -------
        distance : ndarray, double
            Distance between the point and it's closest neighbor
        index : int
            Index of the closest point.

        """
        kdtree = cKDTree(self.cartesian.T)
        distance, index = kdtree.query(point.cartesian.T)

        return distance, index

    def __repr__(self):
        """repr for Coordinate class

        """
        if self.n_points == 1:
            repr_string = "Coordinates of 1 point"
        else:
            repr_string = "Coordinates of {} points".format(self.n_points)
        return repr_string

    def __getitem__(self, index):
        """Return Coordinates at index
        """
        return Coordinates(self._x[index], self._y[index], self._z[index])

    def __setitem__(self, index, item):
        """Set Coordinates at index
        """
        self.x[index] = item.x
        self.y[index] = item.y
        self.z[index] = item.z

    def __len__(self):
        """Length of the object which is the number of points stored.
        """
        return self.n_points

    def to_pyfar(self):
        """Export to a pyfar Coordinates object.

        Returns
        -------
        pyfar.Coordinates
            The equivalent pyfar class object.
        """
        return pyfar.Coordinates(
            self.x,
            self.y,
            self.z,
            domain='cart',
            convention='right',
            unit='met')


class SamplingSphere(Coordinates):
    """Class for samplings on a sphere"""

    def __init__(self, x=None, y=None, z=None, n_max=None, weights=None):
        """Init for sampling class
        """
        Coordinates.__init__(self, x, y, z)
        if n_max is not None:
            self._n_max = np.int(n_max)
        else:
            self._n_max = None

        if weights is not None:
            if len(x) != len(weights):
                raise ValueError("The number of weights has to be equal to \
                        the number of sampling points.")
            self._weights = np.asarray(weights, dtype=np.double)
        else:
            self._weights = None

    @property
    def n_max(self):
        """Spherical harmonic order."""
        return self._n_max

    @n_max.setter
    def n_max(self, value):
        self._n_max = np.int(value)

    @property
    def weights(self):
        """Sampling weights for numeric integration."""
        return self._weights

    @weights.setter
    def weights(self, weights):
        if len(weights) != self.n_points:
            raise ValueError("The number of weights has to be equal to \
                    the number of sampling points.")
        self._weights = np.asarray(weights, dtype=np.double)

    @classmethod
    def from_coordinates(cls, coords, n_max=None, weights=None):
        """Generate a spherical sampling object from a coordinates object

        Parameters
        ----------
        coords : Coordinates
            Coordinate object

        Returns
        -------
        sampling : SamplingSphere
            Sampling on a sphere

        """
        return SamplingSphere(coords.x, coords.y, coords.z,
                              n_max=n_max, weights=weights)

    @classmethod
    def from_cartesian(cls, x, y, z, n_max=None, weights=None):
        """Create a Coordinates class object from a set of points in the
        Cartesian coordinate system.

        Parameters
        ----------
        x : ndarray, double
            x-coordinate
        y : ndarray, double
            y-coordinate
        z : ndarray, double
            z-coordinate
        """
        return SamplingSphere(x, y, z, n_max, weights)

    @classmethod
    def from_spherical(
            cls, radius, elevation, azimuth, n_max=None, weights=None):
        """Create a Coordinates class object from a set of points in the
        spherical coordinate system.

        Parameters
        ----------
        radius : ndarray, double
            The radius for each point
        elevation : ndarray, double
            The elevation angle in radians
        azimuth : ndarray, double
            The azimuth angle in radians
        """
        radius = np.asarray(radius, dtype=np.double)
        elevation = np.asarray(elevation, dtype=np.double)
        azimuth = np.asarray(azimuth, dtype=np.double)
        x, y, z = sph2cart(radius, elevation, azimuth)
        return SamplingSphere(x, y, z, n_max, weights)

    @classmethod
    def from_array(
            cls, values, n_max=None,
            weights=None, coordinate_system='cartesian'):
        """Create a Coordinates class object from a set of points given as
        numpy array

        Parameters
        ----------
        values : double, ndarray
            Array with shape Nx3 where N is the number of points.
        coordinate_system : string
            Coordinate convention of the given values.
            Can be Cartesian or spherical coordinates.
        """
        coords = SamplingSphere(n_max=n_max, weights=weights)
        if coordinate_system == 'cartesian':
            coords.cartesian = values
        elif coordinate_system == 'spherical':
            coords.spherical = values
        else:
            return ValueError("This coordinate system is not supported.")

        return coords

    def __repr__(self):
        """repr for SamplingSphere class
        """
        if self.n_points == 1:
            repr_string = "Sampling with {} point".format(self.n_points)
        else:
            repr_string = "Sampling with {} points".format(self.n_points)
        return repr_string

    def to_pyfar(self):

        """Export to a pyfar Coordinates object.

        Returns
        -------
        pyfar.Coordinates
            The equivalent pyfar class object.
        """
        pyfar_coords = super().to_pyfar()
        pyfar_coords.weights = self.weights / np.linalg.norm(self.weights)
        pyfar_coords.sh_order = self.n_max

        return pyfar_coords
