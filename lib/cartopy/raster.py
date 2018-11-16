# (C) British Crown Copyright 2018, Met Office
#
# This file is part of cartopy.
#
# cartopy is free software: you can redistribute it and/or modify it under
# the terms of the GNU Lesser General Public License as published by the
# Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# cartopy is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with cartopy.  If not, see <https://www.gnu.org/licenses/>.

"""
Implements RasterSource classes using images or data supplied by the user.

The matplotlib interface can make use of RasterSources via the
:meth:`cartopy.mpl.geoaxes.GeoAxes.add_raster` method.

"""

import cartopy.crs as ccrs
from cartopy.img_transform import warp_array
from cartopy.io import RasterSource, LocatedImage
import numpy as np

try:
    import pykdtree.kdtree
    _is_pykdtree = True
except ImportError:
    import scipy.spatial
    _is_pykdtree = False


class LocatedImageRasterSource(RasterSource):
    """Expose a simple image as a Raster Source object."""

    def __init__(self, image, projection):
        """
        Create a Raster Source of a geo-located simple image.

        Parameters
        ----------
        image: `cartopy.io.LocatedImage`
            An image, expressed as a LocatedImage object (containing an image
            and an extent).
        projection
            The native projection of the image.
        """
        self.image = image
        self.projection = projection
        self._geocent = self.projection.as_geocentric()

        self._kd = self._generate_kdtree()

    def _generate_kdtree(self):

        array = np.asanyarray(self.image.image)[::-1]
        extent = self.image.extent
        x = np.linspace(extent[0], extent[1], array.shape[1])
        y = np.linspace(extent[2], extent[3], array.shape[0])
        xs, ys = np.meshgrid(x, y)
        xyz = self._geocent.transform_points(self.projection,
                                             xs.flatten(),
                                             ys.flatten())

        if _is_pykdtree:
            kd = pykdtree.kdtree.KDTree(xyz)
        else:
            # Versions of scipy >= v0.16 added the balanced_tree argument,
            # which caused the KDTree to hang with this input.
            try:
                kd = scipy.spatial.cKDTree(xyz, balanced_tree=False)
            except TypeError:
                kd = scipy.spatial.cKDTree(xyz)

        return kd

    def validate_projection(self, projection):
        """
        Validate whether image is available in the target `projection`. As
        images are regridded before being shown, this is always `True`.
        """
        return True

    def fetch_raster(self, projection, extent, target_resolution):
        target_resolution = [int(np.ceil(val)) for val in target_resolution]
        # Convert image to numpy array (flipping so that origin is 'lower').
        array = np.asanyarray(self.image.image)[::-1]
        new_array, new_extent = warp_array(array, projection,
                                           source_proj=self.projection,
                                           target_res=target_resolution,
                                           source_extent=self.image.extent,
                                           target_extent=extent)
        # Don't forget to flip the image back again!
        return [LocatedImage(new_array[::-1], new_extent)]

