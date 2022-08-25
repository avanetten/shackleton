#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 21 20:23:50 2017

@author: avanetten
"""

import math
import urllib
import os
import numpy as np
import rasterio
from pyproj import Proj, transform
from shapely.geometry import Polygon


###############################################################################
def global_vars():

    global_dic = {
        # edge weight for shortest path ['length', 'Travel Time (h)']
        'edge_weight':  'Travel Time (h)',
        'compute_path_width': 5,                # width of computed paths
        'mapzoom':       9,       # zoom level for GMap
        'plot_width':    1200,    # plot width in pixels
        'histo_height':  300,     # histogram plot height in pixels
        'binsize':       0.5,     # histogram bin size in hours
        'weight_mult':   3.0,     # reweight edges for secondary routes
        'crit_perc':     20.0,    # percentage of points to consider critical
        'minS':          8.0,    # min node size for computed nodes
        'maxS':          15.0,    # max node size for computed nodes
        'concave':       False,   # convex or concave hull
        # 'hull_alpha':    4.5,     # value for concave hull computation (set in define_colors_alphas)
        #'gnode_size':    3,        # size of graph intersections and endpoints
        'gnode_size':    7, #5.5,     # size of graph intersections and endpoints
    }

    return global_dic


###############################################################################
def download_osm_bbox(left, bottom, right, top):
    """ Return a filehandle to the downloaded data bbox data"""
    fp = urllib.urlopen("http://api.openstreetmap.org/api/0.6/"
                        + "map?bbox=%f,%f,%f,%f" % (left, bottom, right, top))
    return fp


###############################################################################
def download_osm_query(queryfile, osmfile):
    '''Test/visualize queries at: http://overpass-api.de/query_form.html'''
    cmdstring = 'curl -o {0} --data-binary @{1} "http://overpass-api.de' \
                + '/api/interpreter"'.format(osmfile, queryfile)
    os.system(cmdstring)


###############################################################################
def construct_poly_query(poly, queryfile, min_road_type='tertiary'):
    '''
    min_road_type is the smallest road to download, should be one of:
        motorway|trunk|primary|secondary|tertiary
    use polygons, must be in format:
    (poly:"latitude_1 longitude_1 latitude_2 longitude_2 latitude_3 longitude_3 …");
    Use http://www.the-di-lab.com/polygon/
        Create polygon on map
        Select show paths button
                (38.47, -12.30)(37.87, -9.88)(35.94, -13.05)
        Edit to remove all () and commas
        there's your poly!

    Other polygon options:
        http://www.birdtheme.org/useful/v3largemap.html
        http://codepen.io/jhawes/pen/ujdgK

    Test by inputting queries into: http://overpass-turbo.eu
    or: http://overpass-api.de/query_form.html
    '''

    # determine which roads to download:
    if min_road_type == 'motorway':
        road_type_string = '    ["highway"~"motorway"]\n'
    elif min_road_type == 'trunk':
        road_type_string = '    ["highway"~"motorway|trunk"]\n'
    elif min_road_type == 'primary':
        road_type_string = '    ["highway"~"motorway|trunk|primary"]\n'
    elif min_road_type == 'secondary':
        road_type_string = '    ["highway"~"motorway|trunk|primary|secondary"]\n'
    elif min_road_type == 'tertiary':
        road_type_string = '    ["highway"~"motorway|trunk|primary|secondary|tertiary"]\n'

    # replace commas and parentheses
    poly = poly.replace('(', '').replace(')', ' ').replace(',', '').strip()
    # construct query
    f = open(queryfile, 'w')
    f.write('(\n')
    f.write('  way\n')
    f.write(road_type_string)
    #f.write('    ["highway"~"motorway|trunk|primary|secondary|tertiary"]\n')
    f.write('    (poly:"'+poly+'");\n')
    f.write('  >;\n')
    f.write(');\n')
    f.write('out;\n')
    f.close()


###############################################################################
# Haversine formula example in Python
# Author: Wayne Dyck
def distance(lat1, lon1, lat2, lon2):
    #lat1, lon1 = origin
    #lat2, lon2 = destination
    radius = 6371  # km

    dlat = math.radians(lat2-lat1)
    dlon = math.radians(lon2-lon1)
    a = math.sin(dlat/2) * math.sin(dlat/2) + math.cos(math.radians(lat1)) \
        * math.cos(math.radians(lat2)) * math.sin(dlon/2) * math.sin(dlon/2)
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    d = radius * c

    return d


###############################################################################
def distance_euclid(x1, y1, x2, y2):
    '''Distance between two points in km, assume x,y are in meters'''

    d = np.sqrt((x2-x1)**2 + (y2-y1)**2)

    return d


###############################################################################
def distance_euclid_km(x1, y1, x2, y2):
    '''Distance between two points in km, assume x,y are in meters'''

    d = np.sqrt((x2-x1)**2 + (y2-y1)**2) / 1000.

    return d


###############################################################################
def latlon_to_wmp(lats, lons):
    '''
    for tile providers need to transform to web mercator projection 
       (meters from lat,lon = (0,0))
    https://github.com/bokeh/bokeh/issues/3858
    input should be lats, lons
    '''

    # transform lng/lat to meters
    from_proj = Proj(init="epsg:4326")
    to_proj = Proj(init="epsg:3857")
    #x, y = transform(from_proj, to_proj, -73.764214, 45.542642)
    x, y = transform(from_proj, to_proj, lons, lats)

    if type(lats) in (float, int, np.float64):
        lenlats = 1
    else:
        lenlats = len(lats)

    # print "Time to transform latlon to wmp for ", lenlats, "items:", \
    #        time.time() - t0, "seconds"

    if lenlats == 1:
        return x, y
    else:
        return np.asarray(x), np.asarray(y)


###############################################################################
def wmp_to_latlon(x_wmp, y_wmp):
    '''
    for tile providers need to transform to web mercator projection 
       (meters from lat,lon = (0,0))
    https://github.com/bokeh/bokeh/issues/3858
    input should be lats, lons
    '''

    # transform meters to lat/lon to meters
    to_proj = Proj(init="epsg:4326")
    from_proj = Proj(init="epsg:3857")
    #x, y = transform(from_proj, to_proj, -73.764214, 45.542642)
    lons, lats = transform(from_proj, to_proj, x_wmp, y_wmp)

    if type(x_wmp) in (float, int, np.float64):
        lenlats = 1
    else:
        lenlats = len(x_wmp)

    # print "Time to transform latlon to wmp for ", lenlats, "items:", \
    #        time.time() - t0, "seconds"

    if lenlats == 1:
        return lats, lons
    else:
        return np.asarray(lats), np.asarray(lons)


###############################################################################
def lin_scale(inarr, minS, maxS):
    '''Output an array of sizes for in arrr, scaled linearly from minS
    to maxS'''

    if len(inarr) == 0:
        return [], 0, 0

    mincount, maxcount = min(inarr), max(inarr)
    if maxcount == mincount:
        return len(inarr)*[maxS], 1, 1

    A = (maxS - minS) / (maxcount - mincount)
    B = maxS - A*maxcount
    sizes = A*np.array(inarr) + B
    return sizes, A, B


###############################################################################
def log_scale(inarr, minS, maxS, verbose=True):
    '''Output an array of sizes for in arrr, scaled logarithmically from minS
    to maxS. Assume integers, so add 1.5 to avoid log(1) = 0'''
    # scale sizes logarithmically
    # S = A*log(Counts) + B
    # A = (maxS - minS) / Log[maxC/minC]
    # B = maxS - A log[maxC]

    if len(inarr) == 0:
        return [], 0, 0

    mincount, maxcount = min(inarr), max(inarr)

    if mincount == maxcount:
        sizes = np.mean([minS, maxS]) * np.ones(len(inarr))
        if verbose:
            print("log_scale() - sizes:", sizes)
        return sizes, 0, 0

    else:

        # add 1.5 to arr so we don't have log(1) = 0
        inarr = 1.5 + np.array(inarr)
        mincount, maxcount = min(inarr), max(inarr)

        if mincount == 0:
            mincount = maxcount / 10.
        inarr_clip = np.clip(inarr, mincount, None)

        A = (maxS - minS) / np.log10(maxcount/mincount)
        B = maxS - A*np.log10(maxcount)
        sizes = A*np.log10(inarr_clip) + B
        if verbose:
            print("log_scale() - inarr:", inarr)
            print("log_scale() - inarr_clip:", inarr_clip)
            print("log_scale() - sizes:", sizes)
    return sizes, A, B


###############################################################################
def log_transform(x, A, B):
    '''Output log transform, assume A, B are set in log_scale()'''

    # scale sizes logarithmically
    # S = A*log(Counts) + B
    # A = (maxS - minS) / Log[maxC/minC]
    # B = maxS - A log[maxC]

    size = A*np.log10(x) + B

    return size


###############################################################################
# http://stackoverflow.com/questions/17106819/accessing-python-dict-values-with-the-key-start-characters
def value_by_key_prefix(d, partial):
    '''Access dict values with incomplete keys, as long as there 
    are not more than one entry for a given string. Use key prefix'''

    default_val = 'unknown'
    matches = []
    matches = [val for key, val in d.items() if key.startswith(partial)]
    # for key, val in d.items():
    #    if type(key) == list:
    #        keycheck = key[0]
    #    else:
    #        keycheck = key
    #    print ("key, partial:", key, partial)
    #    if keycheck.startswith(partial):
    #        matches.append(val)
    #matches = [val for key, val in d.iteritems() if key.startswith(partial)]

    if not matches:
        #raise KeyError(partial)
        # instead of raising an error, use default value
        return default_val
    elif len(matches) > 1:
        raise ValueError('{} matches more than one key'.format(partial))
    else:
        return matches[0]


###############################################################################
def query_kd(kdtree, kd_idx_dic, x, y, k=2):
    '''
    Query the kd-tree for nearest neighbor to the data point with positions
    given by x_wmp, y_wmp
    Return nearest node name, distance, nearest node index
    '''
    point = np.array([x, y])
    dists, idxs = kdtree.query(point, k=k)

    dist = dists[0]
    idx = idxs[0]
    node_name = kd_idx_dic[idx]

    return node_name, dist, idx


###############################################################################
def query_kd_latlon(kdtree, kd_idx_dic, lat, lon, k=10):
    '''
    Query the kd-tree for nearest neighbor to the data point with positions
    given by lat, lon
    Return nearest node name, distance, nearest node index
    !! Answers will be inexact since we're not computing on the surface of 
    a sphere, not on plane!!
    So query 10 nearest, and select nearest using distance calculation!
    '''
    point = np.array([lon, lat])
    dists0, idxs = kdtree.query(point, k=k)

    # get all actual distances
    dists_km = [distance(lat, lon, kdtree.data[i][1], kdtree.data[i][0])
                for i in idxs]
    i_tmp = np.argmin(dists_km)
    dist = dists_km[i_tmp]
    idx = idxs[i_tmp]
    node_name = kd_idx_dic[idx]

    return node_name, dist, idx


###############################################################################
def query_kd_ball(kdtree, kd_idx_dic, x, y, r_m, verbose=False):
    '''
    Query the kd-tree for neighbors within a distance r of the point
    Return nearest node names, distances, nearest node indexes.
    Assume coords are euclidean
    '''

    r_ball = 1.0 * r_m
    # increase ball radius to make sure we dont miss anything?
    # r_ball = 1.5 * r_m

    point = np.array([x, y])
    idxs_unsort = np.array(kdtree.query_ball_point(point, r=r_ball))
    # check distances
    dists_m_unsort = np.asarray([distance_euclid(x, y, kdtree.data[i][0],
                                                 kdtree.data[i][1]) for i in idxs_unsort])
    # sort by increasing distance
    sort_idxs = np.argsort(dists_m_unsort)
    idxs = idxs_unsort[sort_idxs]
    dists_m = dists_m_unsort[sort_idxs]
    # idxs = idxs_unsort[np.argsort(dists_m)]
    if verbose:
        print("    sort_idxs:", sort_idxs)
        print("    idxs_unsort:", idxs_unsort)
        print("    idxs:", idxs)
    # keep only points within distance and greaater than 0
    f0 = np.where((dists_m <= r_m) & (dists_m > 0))
    idxs_refine = list(np.asarray(idxs)[f0])
    dists_m_refine = list(dists_m[f0])
    node_names = [kd_idx_dic[i] for i in idxs_refine]

    return node_names, idxs_refine, dists_m_refine


###############################################################################
def query_kd_ball_latlon(kdtree, kd_idx_dic, lat, lon, r_km):
    '''
    Query the kd-tree for neighbors within a distance r of the point
    given by lat, lon
    Return nearest node names, distances, nearest node indexes
    !! Answers will be inexact since we're not computing on the surface of 
    a sphere, not on plane!!
    So increase approximate radius, and check distances
    '''

    # compute r in lat, lon units
    # at the latitude of Iraq/Syria, a differenct in lat of 0.01 = 1.1119 km
    # and a difference in lon of 0.01 = 0.8996 km
    # so for now, assume anything a change of 0.01 in coords ~ 1 km
    r_ball = r_km * 0.01
    # increase ball radius to make sure we dont miss anything
    r_ball = 1.5 * r_ball

    point = np.array([lon, lat])
    idxs = kdtree.query_ball_point(point, r=r_ball)
    # check distances
    dists_km = np.asarray([distance(lat, lon, kdtree.data[i][1],
                                    kdtree.data[i][0]) for i in idxs])
    # keep only points within distance and greaater than 0
    f0 = np.where((dists_km <= r_km) & (dists_km > 0))
    idxs_refine = list(np.asarray(idxs)[f0])
    dists_km_refine = list(dists_km[f0])
    node_names = [kd_idx_dic[i] for i in idxs_refine]

    return node_names, idxs_refine, dists_km_refine


###############################################################################
def clean_dic(dic):
    # clean tags of unwanted characters (non-unicode)
    for (key, val) in zip(dic.keys(), dic.values()):
        # print ("key, val", key, val
        # replace non-ascii characters (slow but apparently necessary for creating graphs)
        if type(val) == str or type(val) == unicode:
            #newval = "".join(i for i in val if ord(i)<128)
            #newval = val.decode('ascii', 'ignore')
            #newval = val.encode('ascii', 'ignore').decode('ascii')
            #newval = val.encode('utf-8', 'ignore')
            # do nothing!
            newval = val
            if newval != val:
                print("original value", val, "updated value:", newval)
            dic[key] = newval

    return dic


###############################################################################
# https://hub.mybinder.org/user/bokeh-bokeh-notebooks-yqh9m44y/notebooks/tutorial/09%20-%20Geographic%20Plots.ipynb
def wgs84_to_web_mercator(df, lon="lon", lat="lat"):
    """Converts decimal longitude/latitude to Web Mercator format"""
    k = 6378137
    df["x"] = df[lon] * (k * np.pi/180.0)
    df["y"] = np.log(np.tan((90 + df[lat]) * np.pi/360.0)) * k
    return df


###############################################################################
def geotiff_boundaries(image_path):
    '''Extract lat,lon boundary coords for a geotiff, return a polygon'''
    
    btmp = rasterio.open(image_path).bounds
    left, bottom, right, top = btmp
    points = [(left, bottom), (left, top), (right, top), (right, bottom)]
    polygon = Polygon(points)
    return polygon
    

###############################################################################
def ps_coords_to_osmnx_request(lats, lons):
    '''From a list of lats and lons, construct a shapely polygon and execute
    the osmnx command to download the osm street network
    https://github.com/gboeing/osmnx/blob/master/osmnx/core.py
    def graph_from_polygon(polygon, network_type='all_private', simplify=True,
                       retain_all=False, truncate_by_edge=False, name='unnamed',
                       timeout=180, memory=None,
                       max_query_area_size=50*1000*50*1000,
                       clean_periphery=True, infrastructure='way["highway"]',
                       custom_filter=None):
    polygon : shapely Polygon or MultiPolygon
        the shape to get network data within. coordinates should be in units of
        latitude-longitude degrees.
    !! Actully, coords should be in longitude, latitude !!
       (see https://github.com/gboeing/osmnx/blob/master/tests/test_osmnx.py
            test_get_network_methods())
    '''

    print("Creating osmnx graph...")
    points = [(lon, lat) for (lat, lon) in zip(lats, lons)]
    #points = [(lat, lon) for (lat,lon) in zip(lats, lons)]
    polygon = Polygon(points)
    if verbose:
        print("polygon.coords:", list(polygon.exterior.coords))

    return polygon