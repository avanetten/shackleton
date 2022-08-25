# -*- coding: utf-8 -*-
"""
Created on Fri Jun 5 16:54:23 2015

@author: avanetten
"""

# https://github.com/dwyerk/boundaries
# https://github.com/dwyerk/boundaries/blob/master/concave_hulls.ipynb
import shapely.ops
import shapely.geometry
import scipy.spatial
import numpy as np
import math
import time


###############################################################################
def add_edge(edges, edge_points, coords, i, j):
    """
    Add a line between the i-th and j-th points,
    if not in the list already
    """
    if (i, j) in edges or (j, i) in edges:
        # already added
        return
    edges.add((i, j))
    edge_points.append(coords[[i, j]])


###############################################################################
def alpha_shape(coords, alpha=100, verbose=True):
    """
    https://github.com/dwyerk/boundaries/blob/master/concave_hulls.ipynb
    see also: http://code.flickr.net/2008/10/30/the-shape-of-alpha/
    Compute the alpha shape (concave hull) of a set
    of points.
    # alpha = 0.5 for lat-lon
    @param coords: 2-d array of [[x0, y0], ..., [xn, yn]]
    @param alpha: alpha value to influence the
        gooeyness of the border. Smaller numbers
        don't fall inward as much as larger numbers.
        Too large, and you lose everything!
    return x, y coords of hull
    """
    t0 = time.time()
    if verbose:
        print("concave_hull.py - alpha_shape() - coords:", coords)
    if len(coords) < 4:
        # When you have a triangle, there is no sense
        # in computing an alpha shape.
        # return shapely.geometry.MultiPoint(list(points)).convex_hull
        return coords[:, 0], coords[:, 1]

    t1 = time.time()
    tri = scipy.spatial.Delaunay(coords)
    if verbose:
        print("concave_hull.py - alpha_shape() - tri:", tri)

    edges = set()
    edge_points = []
    # loop over triangles:
    # ia, ib, ic = indices of corner points of the
    # triangle
    for ia, ib, ic in tri.vertices:
        pa = coords[ia]
        pb = coords[ib]
        pc = coords[ic]

        # Lengths of sides of triangle
        a = math.sqrt((pa[0]-pb[0])**2 + (pa[1]-pb[1])**2)
        b = math.sqrt((pb[0]-pc[0])**2 + (pb[1]-pc[1])**2)
        c = math.sqrt((pc[0]-pa[0])**2 + (pc[1]-pa[1])**2)

        # Semiperimeter of triangle
        s = (a + b + c)/2.0

        # Area of triangle by Heron's formula
        area = math.sqrt(s*(s-a)*(s-b)*(s-c))
        if area <= 0:
            continue
        # if verbose:
        #    print("concave_hull.py - alpha_shape() - area:", area)

        circum_r = a*b*c/(4.0*area)
        # if verbose:
        #    print("concave_hull.py - alpha_shape() - circum_r:", circum_r)

        # Here's the radius filter.
        # print circum_r
        if circum_r < 1.0 / alpha:
            add_edge(edges, edge_points, coords, ia, ib)
            add_edge(edges, edge_points, coords, ib, ic)
            add_edge(edges, edge_points, coords, ic, ia)
    print("Time to compute edge_points:", time.time() - t1, "seconds")

    t1 = time.time()
    m = shapely.geometry.MultiLineString(edge_points)  # slow
    triangles = list(shapely.ops.polygonize(m))  # slow
    concave_hull = shapely.ops.cascaded_union(triangles)  # very slow
    print("Time to create hull polygon:", time.time() - t1, "seconds")
    # sometimes we will get a multipolygon, which breaks
    try:
        xhull, yhull = concave_hull.exterior.coords.xy
        xhull = xhull.tolist()
        yhull = yhull.tolist()
    except:
        xhull, yhull = [], []
    # get indices of hull points
    hull_points = np.vstack((xhull, yhull)).T.tolist()
    if verbose:
        print("concave_hull.py - alpha_shape() - hull_points:", hull_points)

    coordslist = coords.tolist()
    # print coordslist
    # print hull_points
    t1 = time.time()
    hull_indices = []
    for p in hull_points:
        idx = coordslist.index(p)
        hull_indices.append(idx)
    print("Time to find indices:", time.time() - t1, "seconds")

    print("Time to compute concave hull:", time.time() - t0, "seconds")
    # return concave_hull, edge_points
    return concave_hull, xhull, yhull, hull_indices


###############################################################################
def test_hull():
    '''Test function'''

    x = gsource.data['lon']
    y = gsource.data['lat']
    coords = np.vstack((x, y)).T

    import pylab as pl
    from matplotlib.collections import LineCollection
    for i in range(9):
        alpha = (i+1)*.1
        concave_hull, xhull, yhull, hull_indices = alpha_shape(
            coords, alpha=alpha)
        # print concave_hull
        lines = LineCollection(edge_points)
        pl.figure(figsize=(10, 10))
        pl.title('Alpha={0} Delaunay triangulation'.format(
            alpha))
        pl.gca().add_collection(lines)
        delaunay_points = coords
        pl.plot(delaunay_points[:, 0], delaunay_points[:, 1],
                'o', hold=1, color='#f16824')

        #_ = plot_polygon(concave_hull)
        _ = pl.plot(x, y, 'o', color='#f16824')

    # Extract the point values that define the perimeter of the polygon
    xhull, yhull, hull_indices = concave_hull.exterior.coords.xy
