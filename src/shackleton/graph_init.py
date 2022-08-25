# -*- coding: utf-8 -*-
"""
Created on Wed Dec 17 11:09:58 2014
@author: avanetten
"""

import osmnx as ox
import time
import networkx as nx
import os
import sys
import random
import matplotlib
import itertools
matplotlib.use('tkagg')

import graph_utils
import utils

###############################################################################
global_dic = utils.global_vars()


###############################################################################
def create_cresi_graph(pkl, plot=False, verbose=True):

    G = nx.read_gpickle(pkl)

    # plot graph with matplotlib
    if plot:
        ox.plot_graph(G, save=True, filename='ox_plot')

    # Graph properties
    if verbose:

        print("Num G_gt_init.nodes():", len(G.nodes()))
        print("Num G_gt_init.edges():", len(G.edges()))

        # print random node prop
        node = random.choice(list(G.nodes()))
        print((node, "G random node props:", G.nodes[node]))

        # print random edge properties
        edge_tmp = random.choice(list(G.edges()))
        print(edge_tmp, "G random edge props:",
              G.edges[edge_tmp[0], edge_tmp[1], 0])
        # print("Edges in G: ", G.edges(data=True))

        # also determine number of segments
        nsegs = 0
        for u, v, data in G.edges(data=True):
            # print ("u,v,data:", u,v,data)
            if 'geometry' in data.keys():
                geom = data['geometry']
                npoints = len(list(geom.coords))
                nsegs += npoints - 1
            else:
                nsegs += 1
                print("missing geom")
                break
        print("N segs:", nsegs)

    return G


###############################################################################
def create_osm_graph(bbox=None, poly_shapely=None, pkl=None,
                     network_type='drive',
                     to_crs={'init': 'epsg:3857'},
                     # speed_dict=int(35),
                     speed_mph_key='speed_mph',
                     speed_mps_key='speed_m/s',
                     travel_time_s_key='travel_time',
                     travel_time_key='Travel Time (h)',
                     road_type_key='highway',
                     plot=False, simplify=True,
                     verbose=True):
    '''Import a graph with osmnx
    bbox =  [lat0, lat1, lon0, lon1]
    poly_shapely: shapely Polygon or MultiPolygon
        the shape to get network data within. coordinates should be in units of
        latitude-longitude degrees.
    to_crs=None to use utm, or {'init': 'epsg:3857'} to use wms
    speed_dict is a convertion dictionary to get speed for each edge, if it's
    just {35}, assume all routes have the same speed limit'''

    if verbose:
        print("Executing create_osm_graph...")

    if bbox:
        # bbox = (north, south, east, west)
        [lat1, lat0, lon1, lon0] = bbox
        G0_tmp = ox.graph_from_bbox(lat1, lat0, lon1, lon0,
                                    network_type=network_type,
                                    simplify=simplify)

    elif poly_shapely:
        G0_tmp = ox.graph_from_polygon(poly_shapely,
                                       network_type=network_type,
                                       simplify=simplify)

    elif pkl:
        G0_tmp = nx.read_gpickle(pkl)

    # project graph
    if verbose:
        print("Projecting graph...")
    # https://github.com/gboeing/osmnx/blob/master/osmnx/projection.py
    G1_tmp = ox.project_graph(G0_tmp, to_crs=to_crs)

    # ensure linestrings
    if verbose:
        print("Creating linestrings...")
    G = graph_utils.create_edge_linestrings(G1_tmp.to_undirected())

    if verbose:
        print("Create speed limit and travel times...")
    for u, v, data in G.edges(data=True):
        # print (data)

        # get speed
        speed_mph = infer_max_speed_edge_data(data,
                                              road_type_key=road_type_key)
        # old version
        # if type(speed_dict) == int:
        #    speed_mph = speed_dict
        # else:
        #    print ("Still need to write some code in create_osm_graph...")
        #    return

        speed_mps = 0.44704 * speed_mph
        travel_time_s = data['length'] / speed_mps
        travel_time_h = travel_time_s / 3600
        # assign values
        data[speed_mph_key] = speed_mph
        data[speed_mps_key] = speed_mps
        data[travel_time_s_key] = travel_time_s
        data[travel_time_key] = travel_time_h

    # add linestrings
    print("Creating edge linestrings...")
    G = graph_utils.create_edge_linestrings(G, remove_redundant=True, verbose=False)
    # G = apls.create_edge_linestrings(G, remove_redundant=True, verbose=False)

    # plot graph with matplotlib
    if plot:
        ox.plot_graph(G, save=True, filename='ox_plot')

    # Graph properties
    if verbose:

        print("Num G_gt_init.nodes():", len(G0_tmp.nodes()))
        print("Num G_gt_init.edges():", len(G0_tmp.edges()))

        # print random node prop
        node = random.choice(list(G.nodes()))
        print((node, "G random node props:", G.nodes[node]))

        # print random edge properties
        edge_tmp = random.choice(list(G.edges()))
        print(edge_tmp, "G random edge props:",
              G.edges[edge_tmp[0], edge_tmp[1], 0])
        # print("Edges in G: ", G.edges(data=True))

        # also determine number of segments
        nsegs = 0
        for u, v, data in G.edges(data=True):
            # print ("u,v,data:", u,v,data)
            if 'geometry' in data.keys():
                geom = data['geometry']
                npoints = len(list(geom.coords))
                nsegs += npoints - 1
            else:
                nsegs += 1
                print("missing geom")
                break
        print("N segs:", nsegs)

    return G


###############################################################################
def _node_props(G, weight='Travel Time (h)', compute_all=False):
    '''
    THIS HAS NOT BEEN UPDATED TO NX 2+!
    Add node properties to graph
    Assume encoding is unicode: utf-8
    node properties of interest: lat, lon, ntype, deg, eigen_centrality
    Very slow to compute all centrality values
    '''
    t0 = time.time()
    skip_betweenness = False

    # Compute a few properties
    # http://en.wikipedia.org/wiki/Centrality

    # Timing numbers for refined graph around Mosul:
    # Number of nodes: 4509
    # Number of edges: 6142
    # Time to compute degree centrality: 0.00338315963745 seconds
    # Time to compute Katz centrality: 2.45788598061 seconds
    # Time to compute closeness centrality: 123.56230402 seconds
    # Time to compute betweenness centrality: 215.34570694 seconds
    # Time to compute node properties: 343.798242092 seconds

    # web mercator projection
    print("weight:", weight)
    # print random edge properties
    edge_tmp = random.choice(list(G.edges()))
    print("G random edge props:", edge_tmp, ":",
          G.edges[edge_tmp[0], edge_tmp[1], 0])
    # print random node prop
    node = random.choice(list(G.nodes()))
    print(("G random node props:", node, ":", G.nodes[node]))

    # degree centrality O(v^2)
    # The degree centrality for a node v is the fraction of nodes
    # it is connected to.
    t1 = time.time()
    deg_centr_dic = nx.degree(G)
    # print ("Time to compute degree centrality:", time.time() - t1, "seconds")

   # # Eigenvector centrality is a measure of the influence of a node in a
   # # network fastish
   # t1 = time.time()
   # if compute_all:
   #     try:
   #         eigen_centr_dic = nx.eigenvector_centrality(G, weight=weight)
   #         print ("Time to compute eigenvector centrality:", time.time() - t1, "seconds")
   #     except:
   #         eigen_centr_dic = dict.fromkeys( G.nodes(), 'N/A' )
   # else:
   #     eigen_centr_dic = dict.fromkeys( G.nodes(), 'N/A' )
   #
   # # Katz centrality
   # # Generalization of degree centrality. Katz centrality measures the number
   # # of all nodes that can be connected through a path, while the
   # # contributions of distant nodes are penalized. variant of eigenvector
   # # centrality
   # # not super fast
   # t1 = time.time()
   # if compute_all:
   #     katz_centr_dic = nx.katz_centrality_numpy(G, weight=weight)
   #     print ("Time to compute Katz centrality:", time.time() - t1, "seconds")
   # else:
   #     katz_centr_dic = dict.fromkeys(G.nodes(), 0 )

    # closeness centrality: Closeness centrality at a node is 1/average
    # distance to all other nodes
    # slow!
    t1 = time.time()
    if compute_all:
        close_centr_dic = nx.closeness_centrality(G, distance=weight)
        print("Time to compute closeness centrality:",
              time.time() - t1, "seconds")
    else:
        close_centr_dic = dict.fromkeys(G.nodes(), 0)

    # Betweenness centrality O(v^3)
    # quantifies the number of times a node acts as a
    # bridge along the shortest path between two other nodes
    # extremely slow!
    t1 = time.time()
    if compute_all and not skip_betweenness:
        between_centr_dic = nx.betweenness_centrality(G, weight=weight)
        print("Time to compute betweenness centrality:",
              time.time() - t1, "seconds")
    else:
        between_centr_dic = dict.fromkeys(G.nodes(), 0)

    # add to graph
    t1 = time.time()
    for i, (n, data) in enumerate(list(G.nodes(data=True))):
        data['Degree'] = deg_centr_dic[n]
        # G.node[n]['Degree'] = deg_centr_dic[n]
        if compute_all:
            # data['Eigenvector Centrality'] = eigen_centr_dic[n]
            # data['Katz Centrality'] = katz_centr_dic[n]
            data['Closeness Centrality'] = close_centr_dic[n]
            data['Betweenness Centrality'] = between_centr_dic[n]
        # if (i % 1000) == 0:
        #    print i, "G.node[n]", G.node[n]
    print("Time to add node properties to graph:", time.time() - t1, "seconds")

    print("Time to compute node properties:", time.time() - t0, "seconds")

    # return G, deg_centr_dic, eigen_centr_dic, katz_centr_dic
    return G


###############################################################################
def infer_max_speed_edge_data(edge_props, road_type_key='highway',
                              default_max_speed=31.404):
    '''Assign max speed based on road type (mph)'''

    # mph
    speed_dic = {
        'motorway':     65.,
        'trunk':        55.,
        'primary':      45.,
        'secondary':    35.,
        'tertiary':     30.,
        'residential':  25.,
    }

    if 'highway' not in list(edge_props.keys()):
        return default_max_speed

    else:
        try:
            road_type = edge_props[road_type_key]
            speed = speed_dic[road_type]
        except KeyError:
            speed = default_max_speed
        return speed


###############################################################################
def infer_max_speed_road_type(road_type):
    '''Assign max speed based on road type'''

    default_max_speed = 60.404     # km/h
    speed_dic = {
        'motorway':     105.,
        'trunk':        100.,
        'primary':      90.,
        'secondary':    60.,
        'tertiary':     50.,
        'residential':  40.,
    }

    # 'Access dict values with uncomplete keys, as long as there
    # are not more than one entry for a given string. Use key prefix'
    matches = [val for key, val in speed_dic.iteritems()
               if key.startswith(road_type)]
    if not matches:
        # raise KeyError(partial)
        # instead of raising an error, use default value
        return default_max_speed
    elif len(matches) > 1:
        raise ValueError('{} matches more than one key'.format(road_type))
    else:
        return matches[0]


###############################################################################
def _edge_props(e, attr_dic):
    '''
    Add edge properties of edge e = (source, target) to attr_dic
    Use edge attribute dictionary for calculations
    Assume encoding is unicode: utf
    attributes of interest:
    Road Type, Bridge, Road Name, ref, Num Lanes, Max Speed (km/h),
        Travel Time (h), Path Length (km), Edge Length (km), e_id, link
    '''

    # t0 = time.time()
    s, t = e
    # print ("s,t, attr_dic", s, t, attr_dic

#    if attr_dic['e_id'].startswith('31242722-4'):
#        print ("edge, s, t", e, s, t
#        print ("edge_props", attr_dic
#        print '\n'

    # link
    attr_dic['Link'] = s + ' <=> ' + t

    # road type
    if 'highway' in attr_dic:
        attr_dic['Road Type'] = attr_dic['highway']
    else:
        attr_dic['Road Type'] = 'unknown'

    # bridge
    if 'bridge' in attr_dic:
        attr_dic['Bridge'] = True
    else:
        attr_dic['Bridge'] = False

    # road name, first try english name
    if 'name:en' in attr_dic:
        attr_dic['Road Name'] = attr_dic['name:en']
    else:
        if 'name' in attr_dic:
            attr_dic['Road Name'] = attr_dic['name']
        else:
            attr_dic['Road Name'] = 'unknown'

    # ref
    if 'ref' not in attr_dic:
        attr_dic['ref'] = 'unknown'

    # num lanes
    if 'lanes' in attr_dic:
        try:
            attr_dic['Num Lanes'] = int(attr_dic['lanes'])
        except TypeError:
            attr_dic['Num Lanes'] = 1
    else:
        attr_dic['Num Lanes'] = 1

#    G.edge[s][t]['Path Length (km)'] = G.edge[s][t]['path length (km)']
#    G.edge[s][t]['Edge Length (km)'] = G.edge[s][t]['edge length (km)']

    # max speed
    # By default, values will be interpreted as kilometres per hour.
    # If the speed limit should be specified in a different unit the unit
    # can be added to the end of the value, separated by a space
    # e.g. 45 mph
    # couls also be 100:110
    if 'maxspeed' in attr_dic:
        sraw = str(attr_dic['maxspeed'])
        # optional, strip whitespace
        sraw = sraw.replace(' ', '').lower()
        # split number and string values
        # eg: ["".join(x) for _, x in itertools.groupby("dfsd98 sd8f68as7df56", key=str.isdigit)]
        slist = ["".join(x)
                 for _, x in itertools.groupby(sraw, key=str.isdigit)]
        # if only one element, in km/h
        if len(slist) == 0:
            attr_dic['Max Speed (km/h)'] = float(sraw)
        # else, test if in mph
        else:
            # try to parse max speed, if cannot parse, revert to
            try:
                sval = float(slist[0])
                if 'mph' not in slist:
                    # just take first element
                    attr_dic['Max Speed (km/h)'] = sval
                else:
                    # if found mph, convert
                    attr_dic['Max Speed (km/h)'] = sval * 1.60934
            except:
                # can't parse, so guess max speed
                attr_dic['Max Speed (km/h)'] = infer_max_speed(attr_dic['Road Type'])
    # if no maxspeed, guess max speed
    else:
        attr_dic['Max Speed (km/h)'] = infer_max_speed(attr_dic['Road Type'])

    # travel time
    attr_dic['Travel Time (h)'] = \
        float(attr_dic['Edge Length (km)']) / \
        float(attr_dic['Max Speed (km/h)'])

    # print ("Time to compute edge properties:", time.time() - t0, "seconds"

    return attr_dic

