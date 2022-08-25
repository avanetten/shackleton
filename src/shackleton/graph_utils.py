#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 11 23:17:26 2018

@author: Adam Van Etten
"""

from shapely.geometry import Point, LineString
import scipy.spatial
import bokeh.plotting as bk
import geopandas as gpd
import pandas as pd
import numpy as np
import networkx as nx
import osmnx as ox
import scipy.spatial
import time
import sys
import os
import copy
import importlib

import graph_funcs
import bokeh_utils
import utils


###############################################################################
### Data ingest
###############################################################################
###############################################################################
def load_YOLT(test_predictions_gdf, G=None, categories=[],
                 min_prob=0.05, scale_alpha_prob=True, max_dist_m=10000,
                 dist_buff=5000, nearest_edge=True, search_radius_mult=5,
                 max_rows=10000, randomize_goodness=False,
                 verbose=False, super_verbose=False):
    '''Load data from output of YOLT
    Columns:
        Loc_Tmp	Prob	Xmin	Ymin	Xmax	Ymax	Category	
        Image_Root_Plus_XY	Image_Root	Slice_XY	Upper	Left
        Height	Width	Pad	Im_Width	Im_Height	Image_Path
        Xmin_Glob	Xmax_Glob	Ymin_Glob	Ymax_Glob
        Xmin_wmp, Xmax_wmp, Ymin_wmp, Ymax_wmp
    # buses are good (+1) trucks are bad (-1), cars are neutral (0)
    '''

    t0 = time.time()
    print("Executing load_YOLT()...")
    print("  max_dist_m:", max_dist_m)
    print("Test predictions gdf:", test_predictions_gdf)

    # colors for plotting
    color_dic, alpha_dic = bokeh_utils.define_colors_alphas()

    # read gdf
    df_raw = gpd.read_file(test_predictions_gdf)
    
    # rename cols
    df_raw = df_raw.rename(columns={"prob": "Prob", "category": "Category"})
    
    # filter length
    df_raw = df_raw[:max_rows]
    print("Len raw predictions:", len(df_raw))

    # make sure index starts at 1, not 0
    df_raw.index = np.arange(1, len(df_raw) + 1)
    print("Unique Categories:", np.unique(df_raw['Category'].values))
    cat_value_counts = df_raw['Category'].value_counts()
    print("Category Value Counts:", cat_value_counts)
    
    # filter out probabilities
    df = df_raw[df_raw['Prob'] >= min_prob]

    # filter out categories
    if len(categories) > 0:
        df = df.loc[df['Category'].isin(categories)]
    
    # get bounds of geoms
    xmin_list, xmax_list, ymin_list, ymax_list = [], [], [], []
    for geom_tmp in df['geometry']:
        minx, miny, maxx, maxy = geom_tmp.bounds
        xmin_list.append(minx)
        xmax_list.append(maxx)
        ymin_list.append(miny)
        ymax_list.append(maxy)
    df['Xmin_wmp'] = xmin_list
    df['Xmax_wmp'] = xmax_list
    df['Ymin_wmp'] = ymin_list
    df['Ymax_wmp'] = ymax_list

    print("Len predictions csv:", len(df))

    # # rename wmp columns to what run_gui.py expects
    # df = df.rename(index=int, columns={
    #     'x0_wmp': 'Xmin_wmp',
    #     'x1_wmp': 'Xmax_wmp',
    #     'y0_wmp': 'Ymin_wmp',
    #     'y1_wmp': 'Ymax_wmp',
    # })

    df['xmid'] = 0.5*(df['Xmin_wmp'].values + df['Xmax_wmp'].values)
    df['ymid'] = 0.5*(df['Ymin_wmp'].values + df['Ymax_wmp'].values)

    # each box has a count (and num) of 1
    df['count'] = np.ones(len(df))
    df['num'] = np.ones(len(df))

    # set lat lons
    lats, lons = utils.wmp_to_latlon(df['xmid'].values, df['ymid'].values)
    df['lat'] = lats
    df['lon'] = lons

    # determine colors
    colors = [color_dic[cat] for cat in df['Category'].values]
    # V0
    # colors = [color_dic['goodnode_color'] if v > 0 else color_dic['badnode_color'] for v in df['Val'].values]
    df.insert(len(df.columns), "color", colors)
    # df['color'] = colors

    # set Val
    # buses are good (+1) trucks are bad (-1), cars are neutral (0)
    vals = compute_goodness(df, randomize=randomize_goodness)
    df['Val'] = vals

    # vals = np.zeros(len(df))
    # pos_idxs = np.where(df['Category'].values == 'Bus')
    # neg_idxs = np.where(df['Category'].values == 'Truck')
    # vals[pos_idxs] = 1
    # vals[neg_idxs] = -1
    # df['Val'] = vals
    ## V0 (random)
    ## asssign a random value of 0 or 1
    ## df_raw['Val'] = np.random.randint(0,2,size=len(df_raw))

    # get kdtree
    # if G is not None:
    #    kd_idx_dic, kdtree = G_to_kdtree(G)
    #    # get graph bounds
    #    xmin0, ymin0, xmax0, ymax0 = get_G_extent(G)
    #    xmin, ymin = xmin0 - dist_buff, ymin0 - dist_buff
    #    xmax, ymax = xmax0 + dist_buff, ymax0 + dist_buff
    #    print ("xmin, xmax, ymin, ymax:", xmin, xmax, ymin, ymax)

    # set alpha
    if scale_alpha_prob:
        # scale between 0.3 and 0.8
        prob_rescale = df['Prob'] / np.max(df['Prob'].values)
        alphas = 0.3 + 0.5 * prob_rescale
    else:
        alphas = 0.75
    # set alpha column
    df.insert(len(df.columns), "line_alpha", alphas)
    df.insert(len(df.columns), "fill_alpha", alphas-0.1)
    #df['line_alpha'] = alphas
    #df['fill_alpha'] = alphas - 0.15

    # create dictionary of gdelt node properties
    df.insert(len(df.columns), "nearest_osm", '')
    df.insert(len(df.columns), "dist", 0.0)
    df.insert(len(df.columns), "status", [
              'good' if v > 0 else 'bad' for v in df['Val']])
    #df['nearest_osm'] = ''
    #df['dist'] = 0.0
    #df['status'] = ['good' if v > 0 else 'bad' for v in df['Val']]

    # create sets of nodes
    s0 = set([])
    idx_rem = []
    g_node_props_dic = {}
    xmids, ymids, dists, nearests = [], [], [], []

    # iterate through rows to determine nearest edge to each box
    # need to speed this up!!!
    # https://gis.stackexchange.com/questions/222315/geopandas-find-nearest-point-in-other-dataframe
    # https://github.com/CosmiQ/solaris/blob/master/solaris/utils/geo.py
    if nearest_edge:
        print("Inserting new nodes for each YOLT data point...")
        X = df['xmid'].values
        Y = df['ymid'].values

        # get graph bounds
        xmin0, ymin0, xmax0, ymax0 = get_G_extent(G)
        xmin, ymin = xmin0 - dist_buff, ymin0 - dist_buff
        xmax, ymax = xmax0 + dist_buff, ymax0 + dist_buff
        print("xmin, xmax, ymin, ymax:", xmin, xmax, ymin, ymax)

        # construct a dictionary to track altered edges
        #dict_edge_altered = dict.fromkeys(list(G_.edges()), [])
        dict_edge_altered = {}
        for e in G.edges():
            dict_edge_altered[e] = []
        #print (dict_edge_altered)

        # get edges
        # time all at once
        tt0 = time.time()
        spacing_dist = 5  # spacing distance (meters)
        nearest_edges = ox.get_nearest_edges(
            G, X, Y, method='kdtree', dist=spacing_dist)
        print("Time to get {} nearest edges = {}".format(
            len(nearest_edges), time.time()-tt0))

        tt1 = time.time()
        for i, (index, row) in enumerate(df.iterrows()):

            # if index > 23:
            #    return

            insert_node_id = int(-1 * index)
            #insert_node_id = -1 + int(-1 * index)
            if (i % 100 ) == 0:
                print("index:", index, "/", len(df))
            #print ("  row:", row)
            #cat, prob = row['Category'], row['Prob']
            Xmin_wmp, Xmax_wmp, Ymin_wmp, Ymax_wmp = \
                row['Xmin_wmp'], row['Xmax_wmp'], row['Ymin_wmp'], row['Ymax_wmp']
            #row['x0_wmp'], row['x1_wmp'], row['y0_wmp'], row['y1_wmp']
            xmid, ymid = (Xmin_wmp + Xmax_wmp)/2., (Ymin_wmp + Ymax_wmp)/2.
            #print ("Xmin_wmp, Xmax_wmp, Ymin_wmp, Ymax_wmp:", Xmin_wmp, Xmax_wmp, Ymin_wmp, Ymax_wmp)
            #print (" (Xmin_wmp < xmin):",  (Xmin_wmp < xmin))
            #print (" (Ymin_wmp < ymin):",  (Ymin_wmp < ymin))
            #print (" (Xmax_wmp > xmax):",  (Xmax_wmp > xmax))
            #print (" (Ymax_wmp > ymax):",  (Ymax_wmp > ymax))

            # skip if too far from max coords of G.nodes (much faster than full
            # distance calculation)
            if (Xmin_wmp < xmin) or (Ymin_wmp < ymin) \
                    or (Xmax_wmp > xmax) or (Ymax_wmp > ymax):
                idx_rem.append(index)
                print("Xmin_wmp, Xmax_wmp, Ymin_wmp, Ymax_wmp:",
                      Xmin_wmp, Xmax_wmp, Ymin_wmp, Ymax_wmp)
                print(" (Xmin_wmp < xmin):",  (Xmin_wmp < xmin))
                print(" (Ymin_wmp < ymin):",  (Ymin_wmp < ymin))
                print(" (Xmax_wmp > xmax):",  (Xmax_wmp > xmax))
                print(" (Ymax_wmp > ymax):",  (Ymax_wmp > ymax))
                return
                continue

            best_edge = nearest_edges[i]
            if (i % 500) == 0:
                print("edge:", i, "/", len(df), "best_edge:", best_edge)
            # we 1-indexed the df
            # best_edge = nearest_edges[index-1]
            [u, v] = best_edge[:2]
            if verbose:
                print("u,v:", u, v)
            point = (xmid, ymid)

            # Get best edge
            # check if edge even exists
            #   if so, insert into the edge
            if G.has_edge(u, v):
                best_edge = (u, v)
            elif G.has_edge(v, u):
                best_edge = (v, u)
            else:
                # we'll have to use the newly added edges that won't show up in
                #  ne (the list of closest edges)
                if verbose:
                    print("edge {} DNE!".format((u, v)))
                edges_new = dict_edge_altered[(u, v)]
                if verbose:
                    print("edges_new:", edges_new)
                # get nearest edge from short list
                best_edge, min_dist, best_geom = \
                    get_closest_edge_from_list(G, edges_new,
                                                    Point(xmid, ymid),
                                                    verbose=verbose)

            # insert point
            if verbose:
                print("best edge:", best_edge)
            G, node_props, min_dist, edge_list, edge_props, rem_edge \
                = insert_point_into_edge(G, best_edge, point,
                                              node_id=int(insert_node_id),
                                              max_distance_meters=max_dist_m,
                                              allow_renaming_once=False,
                                              verbose=verbose,
                                              super_verbose=super_verbose)
            if verbose:
                print("min_dist:", min_dist)
            # if an edge has been updated, load new props into dict_edge_altered
            if len(edge_list) > 0:
                z = dict_edge_altered[(u, v)]
                # remove rem_edge item from dict
                if len(rem_edge) > 0 and rem_edge in z:
                    z.remove(rem_edge)
                #print ("z:", z)
                # z.append(edge_list)  # can't append or all values get updated!!!
                val_tmp = z + edge_list
                # update dict value
                dict_edge_altered[(u, v)] = val_tmp
                if super_verbose:
                    print("\n", u, v)
                    print("edge_list:", edge_list)
                    print("dict_edge_altered[(u,v)] :",
                          dict_edge_altered[(u, v)])
                    #print ("  dict", dict_edge_altered)

            # maybe if distance is too large still plot it, but don't include
            #   in analytics
            if min_dist > max_dist_m:
                if verbose:
                    print("dist > max_dist_m")
                node_name = 'Null'
                xmids.append(xmid)
                ymids.append(ymid)
                dists.append(min_dist)
                node_name_tmp = 'null'
                nearests.append(node_name_tmp)
                continue

            else:
                if verbose:
                    print("Updating df values")
                    print("node_props:", node_props)
                node_name = insert_node_id  # node_props['osmid']
                xmids.append(xmid)
                ymids.append(ymid)
                dists.append(min_dist)
                nearests.append(node_name)

                # update node properties if nearest node in s0, else create new
                # If node_name is in s0, there could be multiple reports for the same
                # location.  For now, just keep the report with the largest count
                if node_name not in s0:
                    g_node_props_dic[node_name] = \
                        {'index': [index], 'dist': [min_dist]}
                    s0.add(node_name)
                    # if node_name < 0:
                    #    insert_node_id += -1

                else:
                    if verbose:
                        print("node name", node_name, "already in s0:")
                    g_node_props_dic[node_name]['index'].append(index)
                    g_node_props_dic[node_name]['dist'].append(min_dist)

        # add travel time
        G = add_travel_time(G)

    # iterate through rows to determine nearest node to each box
    else:
        print("Assigning existing intersection/endpoint for each YOLT data point...")
        for index, row in df.iterrows():
            print("index:", index, "/", len(df))
            #cat, prob = row['Category'], row['Prob']
            Xmin_wmp, Xmax_wmp, Ymin_wmp, Ymax_wmp = \
                row['Xmin_wmp'], row['Xmax_wmp'], row['Ymin_wmp'], row['Ymax_wmp']
            xmid, ymid = (Xmin_wmp + Xmax_wmp)/2., (Ymin_wmp + Ymax_wmp)/2.
            #print ("Xmin_wmp, Xmax_wmp, Ymin_wmp, Ymax_wmp:", Xmin_wmp, Xmax_wmp, Ymin_wmp, Ymax_wmp)
            #print (" (Xmin_wmp < xmin):",  (Xmin_wmp < xmin))
            #print (" (Ymin_wmp < ymin):",  (Ymin_wmp < ymin))
            #print (" (Xmax_wmp > xmax):",  (Xmax_wmp > xmax))
            #print (" (Ymax_wmp > ymax):",  (Ymax_wmp > ymax))

            # skip if too far from max coords of G.nodes (much faster than full
            # distance calculation)
            if (Xmin_wmp < xmin) or (Ymin_wmp < ymin) \
                    or (Xmax_wmp > xmax) or (Ymax_wmp > ymax):
                idx_rem.append(index)
                print("Xmin_wmp, Xmax_wmp, Ymin_wmp, Ymax_wmp:",
                      Xmin_wmp, Xmax_wmp, Ymin_wmp, Ymax_wmp)
                print(" (Xmin_wmp < xmin):",  (Xmin_wmp < xmin))
                print(" (Ymin_wmp < ymin):",  (Ymin_wmp < ymin))
                print(" (Xmax_wmp > xmax):",  (Xmax_wmp > xmax))
                print(" (Ymax_wmp > ymax):",  (Ymax_wmp > ymax))
                return
                continue

            # find nearest osm node, update dataframe values
            node_name, dist, kd_idx = utils.query_kd(
                kdtree, kd_idx_dic, xmid, ymid)
            #print ("node_name, dist, kd_idx:", node_name, dist, kd_idx)

            # remove if distance is too large, continue loop
            if dist > max_dist_m:
                idx_rem.append(index)
                print("dist > max_dist_m")
                print("  dist:", dist)
                # return
                continue

            xmids.append(xmid)
            ymids.append(ymid)
            dists.append(dist)
            nearests.append(node_name)
            #df.loc[index, 'Xmid_wmp'] = xmid
            #df.loc[index, 'Ymid_wmp'] = ymid
            #df.loc[index, 'dist'] = dist
            #df.loc[index, 'nearest_node'] = node_name
            ##df.set_value(index, 'Xmid_wmp', xmid)
            ##df.set_value(index, 'Ymid_wmp', ymid)
            ##df.set_value(index, 'dist', dist)
            ##df.set_value(index, 'nearest_osm', node_name)

            # update osm node properties if nearest node in s0, else create new
            # If node_name is in s0, there could be multiple reports for the same
            # location.  For now, just keep the report with the largest count
            if node_name not in s0:
                g_node_props_dic[node_name] = \
                    {'index': [index], 'dist': [dist]}
                s0.add(node_name)

            else:
                g_node_props_dic[node_name]['index'].append(index)
                g_node_props_dic[node_name]['dist'].append(dist)

            #colortmp = color_dic['badnode_color']
            #df.set_value(index, 'color', colortmp)

    # remove unnneeded indexes of df
    print("df.index:", df.index)
    print("idx_rem", idx_rem)
    print("len of original df", len(df))
    if len(idx_rem) > 0:
        df = df.drop(np.unique(idx_rem))
        #df = df.drop(df.index[np.unique(idx_rem)])
    print("len of refined df", len(df))

    df['index'] = df.index.values
    df['Xmid_wmp'] = xmids
    df['Ymid_wmp'] = ymids
    df['dist'] = dists
    df['nearest_node'] = nearests
    #df.loc[index, 'Xmid_wmp'] = xmid
    #df.loc[index, 'Ymid_wmp'] = ymid
    #df.loc[index, 'dist'] = dist
    #df.loc[index, 'nearest_node'] = node_name

    #print ("df:", df)
    #print ("df.columns:", df.columns)

    # reset index?
    #   (Don't want to since we track the index in g_nodes_props dic)
    #df.index = range(len(df))

    # create a columndatasource
    # (drop 'geometry' col since bokeh can't handle shapely Polygons
    # df2 = df.drop(columns=['geometry'])
    # print("df2:", df2)
    source_YOLT = bk.ColumnDataSource(df.drop(columns=['geometry']))
    # source_YOLT = bk.ColumnDataSource(df)

    #G_gt_init = create_edge_linestrings(G_gt_init0.to_undirected())
    #print("Num G_gt_init.nodes():", len(G_gt_init.nodes()))
    #print("Num G_gt_init.edges():", len(G_gt_init.edges()))

    print("Time to load YOLT data:", time.time() - t0, "seconds")
    return G, df, source_YOLT, g_node_props_dic, cat_value_counts


###############################################################################
def compute_goodness(df, randomize=False):
    """
    Compute whether a node is good (+1), netural (0), or bad (-1)
    """

    if randomize:
        outarr = np.random.randint(-1, 2, size=len(df))
    else:
        # buses are good (+1) trucks are bad (-1), cars are neutral (0)
        outarr = np.zeros(len(df))
        pos_idxs = np.where(df['Category'].values == 'Bus')
        neg_idxs = np.where(df['Category'].values == 'Truck')
        outarr[pos_idxs] = 1
        outarr[neg_idxs] = -1

    return outarr


###############################################################################
def density_speed_conversion(N, min_val=0.25):
    """
    Fraction to multiply speed by if there are N nearby vehicles
    """
    
    z = 1.0 - 0.04 * N
    return max(z, min_val)
        
   # # v0
   # if N < 2:
   #     return 1.0
   # elif N == 2:
   #     return 0.95
   # elif N == 3:
   #     return 0.90
   # elif N == 4:
   #     return 0.85
   # elif N == 5:
   #     return 0.80
   # elif N == 6:
   #     return 0.75
   # elif N == 7:
   #     return 0.70
   # elif N == 8:
   #     return 0.65
   # elif N == 9:
   #     return 0.6
   # elif N == 10:
   #     return 0.55
   # elif N == 11:
   #     return 0.50
   # elif N == 12:
   #     return 0.45
   # elif N == 13:
   #     return 0.4
   # elif N == 14:
   #     return 0.35
   # elif N == 15:
   #     return 0.3
   # else:
   #     return 0.25


###############################################################################
def compute_traffic(G, query_radius_m=50, min_nearby=2, max_edge_len=50,
                    verbose=False):
    """
    Compute updated edge speeds based on density of cars

    Notes
    -----
    Assume nodes near cars have been added, with a negative index

    Arguments
    ---------
    G : networkx graph
        Input graph
    query_radiums_m : float
        Radius to query around each negative node for other negative nodes.
        Defaults to ``200`` (meters).
    min_query : int
        Number of nearby cars required to consider trafficy.
        Defaults to ``2``.
    max_edge_len : float
        Maximum edge length to consider reweighting.
        Defaults to ``50`` (meters).
    verbose : bool
        Switch to print relevant values

    Returns
    -------
    edge_update_dict : dict
        Dictionary with fraction of original speed for appropriate edges.
        Example entry:
             (-999, 258910932): 0.4}
             # (-999, 258910932): {'orig_mps': 11.176, 'new_mps': 3.3528}}
    """

    print ("Computing traffic...")
    # get all nodes less than zero (corresponding to a vehicle)
    neg_nodes = [n for n in list(G.nodes()) if n < 0]
    if verbose:
        print("compute_traffic(): neg_nodes:", neg_nodes)

    kd_idx_dic, kdtree = G_to_kdtree(G, node_subset=set(neg_nodes))
    if verbose:
        print("compute_traffic(): kd_idx_dic:", kd_idx_dic)

    # iterate throught each neg node
    edge_update_dict = {}
    edge_altered_set = set([])
    # can't search only negative nodes, because it's possible that normal
    # nodes will have tons of nearby cars too
    # for i, n in enumerate(neg_nodes):
    for i, n in enumerate(list(G.nodes())):
        x, y = G.nodes[n]['x'], G.nodes[n]['y']
        # get number of vehicles near each node
        node_names, idxs_refine, dists_m_refine = utils.query_kd_ball(
            kdtree, kd_idx_dic, x, y, query_radius_m, verbose=False)
        n_nearby = len(node_names)

        if n_nearby > min_nearby:
            if verbose:
                print("i, n:", i, n)
                print("node_names:", node_names)
                print("dists:", dists_m_refine)

            # get edges coincident on nodes
            coinc_e = G.edges(n)
            if verbose:
                print("coinc_e:", coinc_e)
            # check coincident edges
            for (u, v) in coinc_e:
                if verbose:
                    print("u, v,", u, v)
                if (u, v) not in edge_altered_set:
                    # data = G.get_edge_data(u, v)
                    # print("data:", data)
                    line_len = G.edges[u, v, 0]['length']
                    if line_len <= max_edge_len:
                        speed_frac = density_speed_conversion(n_nearby)
                        edge_update_dict[(u, v)] = speed_frac
                        if verbose:
                            print("line_len", line_len)
                            print("speed_frac:", speed_frac)
                        # speed_mps = G.edges[u, v, 0]['speed_m/s']
                        # new_speed = speed_mps * speed_frac
                        # if verbose:
                        #    print("line_len", line_len)
                        #    print("speed_mps:", speed_mps)
                        #    print("new_speed:", new_speed)
                        # edge_update_dict[(u, v)] = {'orig_mps': speed_mps,
                        #                             'new_mps': new_speed}
                        edge_altered_set.add((u, v))

    if verbose:
        print("edge_altered_set:", edge_altered_set)
        print("edge_update_dict:", edge_update_dict)

    return edge_update_dict


###############################################################################
def update_Gweights(G, update_dict,
                    speed_key1='inferred_speed_mph',
                    speed_key2='speed2',
                    edge_id_key='uv',
                    congestion_key='congestion',
                    travel_time_key='Travel Time (h)',
                    travel_time_key2='Travel Time (h) Traffic',
                    travel_time_key_default='Travel Time (h) default',
                    verbose=False):
    """Update G and esource"""

    # color_dic, alpha_dic = define_colors_alphas()
    update_keys = set(list(update_dict.keys()))

    # update speed
    for j, (u, v, data) in enumerate(G.edges(data=True)):
        speed1 = data[speed_key1]
        if (u, v) in update_keys:
            frac = update_dict[(u, v)]
        elif (v, u) in update_keys:
            frac = update_dict[(v, u)]
            # if verbose:
            #    print("u, v, frac:", u, v, frac)
        else:
            frac = 1
        speed2 = speed1 * frac
        data[speed_key2] = speed2
        # set congestion
        data[congestion_key] = frac
    # update travel time
    G = add_travel_time(G, speed_key=speed_key1,
                        travel_time_key=travel_time_key,
                        speed_key2=speed_key2,
                        travel_time_key2=travel_time_key2,
                        travel_time_key_default=travel_time_key_default)

    return G


###############################################################################
def load_gdelt(gdelt_infile, kdtree, kd_idx_dic, max_dist_km=1000000,
               rand_alpha=False):
    '''Import gdelt data and return a dataframe of data as well as a dictionary
    of nearest osm nodes to the gdelt data.  dictionary has score and count
    as values. Also create a bokeh ColumnDataSource
    There might be more than one gdelt datapoint with the same closest node,
    so we need to aggregate the datapoints
    in this case track the index of the gdelt datapoint with a lower count, 
    and remove it later
    osm_node_props_dic is a dictionary of properties of the the nearest 
    osm nodes to gdelt datapoints with key=node_name, 
        vals = index, score, count
    max_dist_km is the maximum distance to the nearest road node.  If 
    the nearest node is greater than this distance, remove the gdelt point
    Skip GDELT data if it's more than 1 degree from the extent of G.nodes(), 
    assume we aren't too near the poles'''

    t0 = time.time()
    # size limits for plotting
    minS, maxS = 5, 30
    shape = 'circle'
    color_dic, alpha_dic = define_colors_alphas()

    # ActionGeo Lat	ActionGeo Long	Avg. Goldstein Scale	Number of Records
    dfgdelt = pd.read_excel(gdelt_infile)

    # set alpha
    if rand_alpha:
        alpha_coeffs = 0.2 + np.random.rand(len(dfgdelt))
    else:
        alpha_coeffs = np.ones(len(dfgdelt)),

    # get min, max coords of kdtree
    min_lon0, max_lon0 = np.min(kdtree.data[:, 0]), np.max(kdtree.data[:, 0])
    min_lat0, max_lat0 = np.min(kdtree.data[:, 1]), np.max(kdtree.data[:, 1])
    # define a 0.5 degree bugger around the extent of G.nodes()
    # assume we aren't near the poles!!!
    degree_buffer = 0.5

    min_lon, max_lon = min_lon0 - degree_buffer, max_lon0 + degree_buffer
    min_lat, max_lat = min_lat0 - degree_buffer, max_lat0 + degree_buffer

    # create sets of nodes
    s0 = set([])
    idx_rem = []
    osm_node_props_dic = {}

    # create dictionary of gdelt node properties
    dfgdelt['nearest_osm'] = ''
    dfgdelt['dist'] = 0.0
    dfgdelt['status'] = ''
    dfgdelt['color'] = ''
    for index, row in dfgdelt.iterrows():
        # assign row values
        [lat, lon, score, count, b0, b1, b2, b3] = row.values

        # skip if too far from max coords of G.nodes (much faster than full
        # distance calculation)
        if ((lat > max_lat or lat < min_lat) and
                (lon > max_lon or lon < min_lon)):
            idx_rem.append(index)
            # print ("lat, lon", lat, lon
            continue

        # find nearest osm node, update dataframe values
        node_name, dist, idx = query_kd(kdtree, kd_idx_dic, lat, lon)

        # remove if distance is too large, continue loop
        if dist > max_dist_km:
            idx_rem.append(index)
            continue

        dfgdelt.set_value(index, 'dist', dist)
        dfgdelt.set_value(index, 'nearest_osm', node_name)

        # update osm node properties if nearest node in s0, else create new
        # If node_name is in s0, there could be multiple reports for the same
        # location.  For now, just keep the report with the largest count
        if node_name not in s0:
            osm_node_props_dic[node_name] = \
                {'index': index, 'score': score, 'count': count}
            s0.add(node_name)

        else:
            # else node is already in s0, so update values
            count0 = osm_node_props_dic[node_name]['count']
            score0 = osm_node_props_dic[node_name]['score']
            index0 = osm_node_props_dic[node_name]['index']
            # update properties if count > existing count
            if count > count0:
                print("GDELT: Replacing values for osm_node:", node_name,
                      "new vals:", score, count, "old vals:", score0, count0)
            else:
                count = count0
                score = score0
            # set dataframe and dic values
            dfgdelt.set_value(index, 'Avg. Goldstein Scale', score)
            dfgdelt.set_value(index, 'Number of Records', count)
            osm_node_props_dic[node_name] = \
                {'index': index, 'score': score, 'count': count}
            # add old index to list of indexes to be removed
            idx_rem.append(index0)

        # update status, color
        if score < 0:
            status = 'bad'
            colortmp = color_dic['badnode_color']
        else:
            status = 'good'
            colortmp = color_dic['goodnode_color']
        dfgdelt.set_value(index, 'status', status)
        dfgdelt.set_value(index, 'color', colortmp)

    # remove unnneeded indexes of dfgelt
    # print ("idx_rem", idx_rem
    print("len of original gdelt df", len(dfgdelt))
    dfgdelt = dfgdelt.drop(dfgdelt.index[np.unique(idx_rem)])
    print("len of refined gdelt df", len(dfgdelt))

    # reset index
    dfgdelt.index = range(len(dfgdelt))

    # create a size column
    numarr = dfgdelt['Number of Records'].values
    sizes, A, B = log_scale(numarr, minS, maxS)
    dfgdelt['sizes'] = sizes
    # print ("minS, maxS", minS, maxS
    # print ("mincount, maxcount", mincount, maxcount
    # print ("A, B", A, B
    # print ("sizes", sizes

    # create label as number of recoords
    dfgdelt['label'] = dfgdelt['Number of Records'].values

    # sort by label, so that id column will be descending by label
    dfgdelt.sort(columns='label', inplace=True, ascending=False)
    # create id column
    nid = np.asarray(['GDELT'+str(i) for i in range(len(dfgdelt))])
    dfgdelt['nid'] = nid

    # get web mercator projection
    x_wmp, y_wmp = utils.latlon_to_wmp(
        dfgdelt['ActionGeo Lat'].values, dfgdelt['ActionGeo Long'].values)
    dfgdelt['x'] = x_wmp
    dfgdelt['y'] = y_wmp

    # create a columndatasource
    source_gdelt = bk.ColumnDataSource(
        data=dict(
            nid=dfgdelt['nid'].values,
            name=dfgdelt['nid'].values,
            count=dfgdelt['Number of Records'].values,
            val=dfgdelt['Avg. Goldstein Scale'].values,
            lat=dfgdelt['ActionGeo Lat'].values,
            lon=dfgdelt['ActionGeo Long'].values,
            x_wmp=dfgdelt['x'].values,
            y_wmp=dfgdelt['y'].values,
            num=dfgdelt['Number of Records'].values,
            score=dfgdelt['Avg. Goldstein Scale'].values,
            color=dfgdelt['color'].values,
            dist=dfgdelt['dist'].values,
            nearest=dfgdelt['nearest_osm'].values,
            status=dfgdelt['status'].values,
            size=dfgdelt['sizes'].values,
            label=dfgdelt['label'].values,
            alpha=alpha_dic['gdelt'] * alpha_coeffs,  # np.ones(len(dfgdelt)),
            shape=np.ones(len(dfgdelt))
        )
    )

    print("Time to load GDELT data:", time.time() - t0, "seconds")
    return dfgdelt, osm_node_props_dic, source_gdelt


###############################################################################
def add_travel_time(G_,
                    length_key='length',  # meters
                    speed_key='inferred_speed_mph',
                    travel_time_key='Travel Time (h)',
                    travel_time_s_key='travel_time_s',
                    speed_key2='speed2',
                    travel_time_key2='Travel Time (h) Traffic',
                    default_speed=31.404,   # mph
                    travel_time_key_default='Travel Time (h) default',
                    verbose=False):
    '''Add travel time estimate to each edge
    if speed_key does not exist, use default
    Default speed is 31.404 mph'''

    for i, (u, v, data) in enumerate(G_.edges(data=True)):
        if speed_key in data:
            speed_mph = data[speed_key]
        else:
            data['inferred_speed'] = default_speed
            data[speed_key] = default_speed
            speed_mph = default_speed

        if verbose:
            print("data[length_key]:", data[length_key])
            print("speed:", speed_mph)

        speed_mps = 0.44704 * speed_mph
        travel_time_s = data['length'] / speed_mps
        travel_time_h = travel_time_s / 3600

        data[travel_time_s_key] = travel_time_s
        data[travel_time_key] = travel_time_h

        # get weights for speed2
        if speed_key2 in data:
            speed_mph2 = data[speed_key2]
        else:
            speed_mph2 = speed_mph
        if verbose:
            print("speed2:", speed_mph2)
        speed_mps2 = 0.44704 * speed_mph2
        travel_time_s2 = data['length'] / speed_mps2
        travel_time_h2 = travel_time_s2 / 3600
        data[travel_time_key2] = travel_time_h2

        # get weights for default
        speed_mps3 = 0.44704 * default_speed
        travel_time_s3 = data['length'] / speed_mps3
        travel_time_h3 = travel_time_s3 / 3600
        data[travel_time_key_default] = travel_time_h3

        # print(data)
        # print ("travel_time_h, travel_time_h2, travel_time_h3:",
        #     travel_time_h, travel_time_h2, travel_time_h3)

    return G_


###############################################################################
def get_aug(G, df_YOLT, kdtree, kd_idx_dic, r_m=200.,
            special_nodes=set([]), node_size=6, shape='circle', verbose=False):
    '''Find osm nodes within a radiums in meters of (r_m) of data points
    special nodes are ones we don't want to augment'''

    t0 = time.time()
    color_dic, alpha_dic = bokeh_utils.define_colors_alphas()

    # make kdtree of YOLT data?
    #df_xy_arr = np.stack((df_YOLT['Xmid_wmp'].values, df_YOLT['Ymid_wmp'].values), axis=1)
    #kdtree_YOLT = scipy.spatial.KDTree(df_xy_arr)

    # iterate through bboxes
    names, vals, cats, idxs, meters = [], [], [], [], []
    names_closest = []
    bad_idxs = []
    # will contain elements with [df_index, distance, arr_idx]
    node_dist_dic = {}
    arr_idx = 0
    for df_index, row in df_YOLT.iterrows():
        if verbose:
            print("df_index:", df_index)
        cat, val = row['Category'], row['Val']
        x, y = row['Xmid_wmp'], row['Ymid_wmp']
        # get nearest points
        names_tmp, idxs_tmp, ms_tmp = utils.query_kd_ball(
            kdtree, kd_idx_dic, x, y, r_m)
        if len(names_tmp) > 0:

            # check if mulitple points have the same distance!
            # if so, reorder so that the index of the desired point 
            # (-1 *df_index) is deemed the closest node
            if (len(names_tmp) > 1) and (ms_tmp[0] == ms_tmp[1]) and (int(-1*df_index) in names_tmp):
                if verbose:
                    print("coincident nodes, reorder so that -1*index is first!")
                    print("  names:", names_tmp)
                    print("  dists:", ms_tmp)
                # mylist.insert(0, mylist.pop(mylist.index(targetvalue)))
                idx_tmp_ro = names_tmp.index(int(-1*df_index))
                names_tmp.insert(0, names_tmp.pop(idx_tmp_ro))
                idxs_tmp.insert(0, idxs_tmp.pop(idx_tmp_ro))
                ms_tmp_copy = ms_tmp.copy()
                ms_tmp.insert(0, ms_tmp.pop(idx_tmp_ro))
                if verbose:
                    print("  names_resort:", names_tmp)
                    print("  dists_resort:", ms_tmp)
                # ordering of meters shouldn't change, if so we screwed up!
                if ms_tmp != ms_tmp_copy:
                    print("reorder screwed up, returning!")
                    return

            # check that each point is not further from an already assigned node
            for j, (name_tmp, dist_tmp, idx_tmp) in enumerate(zip(names_tmp, ms_tmp, idxs_tmp)):
                if name_tmp in set(node_dist_dic.keys()):
                    # if new point is closer than old point, update
                    if dist_tmp < node_dist_dic[name_tmp][1]:
                        # first set node closest if j == 0
                        if j == 0:
                            names_closest.append(name_tmp)
                            if name_tmp > 0:
                                print ("name > 0, why?, returning")
                                print("  names:", names_tmp)
                                print("  dists:", ms_tmp)
                                print("  idxs:", idxs_tmp)
                                return
                        names.append(name_tmp)
                        vals.append(val)
                        cats.append(cat)
                        idxs.append(df_index)
                        meters.append(dist_tmp)
                        # mark old point as to be reomoved
                        bad_idxs.append(node_dist_dic[name_tmp][2])
                        # update dict
                        node_dist_dic[name_tmp] = [df_index, dist_tmp, arr_idx]
                        arr_idx += 1
                    else:
                        continue

                else:
                    # first set node closest if j == 0
                    if j == 0:
                        names_closest.append(name_tmp)
                    names.append(name_tmp)
                    vals.append(val)
                    cats.append(cat)
                    idxs.append(df_index)
                    meters.append(dist_tmp)
                    node_dist_dic[name_tmp] = [df_index, dist_tmp, arr_idx]
                    arr_idx += 1

    names = np.array(names)  # [str(n) + '_aug' for n in names])
    vals = np.array(vals)
    idxs = np.array(idxs)
    meters = np.array([str(np.round(m, 1)) + 'm' for m in meters])
    bbox_labels = np.array(['bbox=' + str(z) for z in idxs])
    cats = np.array(cats)
    bad_idxs = np.sort(np.unique(bad_idxs))

    # remove extraneuous elements
    names = np.delete(names, bad_idxs)
    vals = np.delete(vals, bad_idxs)
    idxs = np.delete(idxs, bad_idxs)
    meters = np.delete(meters, bad_idxs)
    bbox_labels = np.delete(bbox_labels, bad_idxs)
    cats = np.delete(cats, bad_idxs)

    # now assign lists
    # find indices of good (>0) and bad (<0) nodes
    f_nbad = np.where(vals < 0)
    f_ngood = np.where(vals > 0)

    # extract values
    ngood_aug = names[f_ngood]
    nbad_aug = names[f_nbad]
    ngood = np.array(list(set(ngood_aug).intersection(set(names_closest))))
    nbad = np.array(list(set(nbad_aug).intersection(set(names_closest))))
    # extract values
    # good
    #ngood = names[f_ngood]
    #ngood_aug = ngood
    # bad
    #nbad = names[f_nbad]
    #nbad_aug = nbad

    if verbose:
        print("  vals:", vals)
        print("  names:", names)
        print("  names_closest:", names_closest)
        print("  f_nbad:", f_nbad)
        print("  f_ngood:", f_ngood)
        print("ngood_aug:", ngood_aug)
        print("nbad_aug:", nbad_aug)
        print("ngood:", ngood)
        print("nbad:", nbad)

    print("Augmented points: len nbad", len(nbad), "len nbad_aug",
          len(nbad_aug))
    print("Augmented points: len ngood", len(ngood), "len ngood_aug",
          len(ngood_aug))

    # store in columndatasources
    cds_good_aug = bokeh_utils.set_nodes_source(G, ngood_aug, size=node_size,
                                                fill_alpha=alpha_dic['aug'],
                                                shape=shape,
                                                label=idxs[f_ngood],
                                                count=meters[f_ngood],
                                                # val=vals[f_ngood],
                                                val=bbox_labels[f_ngood],
                                                name=ngood_aug,
                                                color=color_dic['goodnode_aug_color'])
    cds_bad_aug = bokeh_utils.set_nodes_source(G, nbad_aug, size=node_size,
                                               fill_alpha=alpha_dic['aug'],
                                               shape=shape,
                                               label=idxs[f_nbad],
                                               count=meters[f_nbad],
                                               # val=vals[f_nbad],
                                               val=bbox_labels[f_nbad],
                                               name=nbad_aug,
                                               color=color_dic['badnode_aug_color'])

    auglist = [ngood, nbad, ngood_aug, nbad_aug, cds_good_aug, cds_bad_aug]

    print("Time to compute auglist:", time.time() - t0, "seconds")
    return auglist


###############################################################################
def get_aug_gdelt(G, dfgdelt, kdtree, kd_idx_dic, r_km=2., special_nodes=set([]),
                  node_size=5, shape='circle'):
    '''Find osm nodes within r_km of data points
    special nodes are ones we don't want to augment'''

    t0 = time.time()
    color_dic, alpha_dic = define_colors_alphas()
    # compute good and bad nodes
    # find bad nodes
    nid = dfgdelt['nid'].values
    count = dfgdelt['Number of Records'].values
    val = dfgdelt['Avg. Goldstein Scale'].values
    nearest = dfgdelt.nearest_osm.values
    # find indices of good and bad nodes
    f_nbad = np.where(val < 0)
    f_ngood = np.where(val >= 0)
    # extract values
    # bad
    nbad = nearest[f_nbad]
    nbad_nid = nid[f_nbad]
    nbad_count = count[f_nbad]
    nbad_val = val[f_nbad]
    # good
    ngood = nearest[f_ngood]
    ngood_nid = nid[f_ngood]
    ngood_count = count[f_ngood]
    ngood_val = val[f_ngood]

    # create set of gdelt nodes
    set_gdelt = set(np.append(ngood, nbad))
    ####################
    # expand node selection based on nearby nodes
    nbad_extra, ngood_extra = [], []
    nbad_extra_nid, ngood_extra_nid = [], []
    nbad_extra_km, ngood_extra_km = [], []
    nbad_aug_set = set([])
    ngood_aug_set = set([])
    nbad_set = set(nbad)
    ngood_set = set(ngood)
    # print ("len(nbad)", len(nbad)
    # print ("nbad", nbad

    for i, n in enumerate(nbad):
        # skip augmenting if n is in special_nodes
        if n in special_nodes:
            continue
        lat, lon = G.nodes[n]['lat'], G.nodes[n]['lon']
        names, idxs, kms = query_kd_ball(kdtree, kd_idx_dic, lat, lon, r_km)
        aug_nids = [nbad_nid[i] + '_aug_' + str(j) for j in range(len(names))]
        nbad_extra.extend(names)
        nbad_extra_km.extend(kms)
        nbad_extra_nid.extend(aug_nids)
    for i, n in enumerate(ngood):
        # skip augmenting if n is in special_nodes
        if n in special_nodes:
            continue
        lat, lon = G.nodes[n]['lat'], G.nodes[n]['lon']
        names, idxs, kms = query_kd_ball(kdtree, kd_idx_dic, lat, lon, r_km)
        aug_nids = [nbad_nid[i] + '_aug_' + str(j) for j in range(len(names))]
        ngood_extra.extend(names)
        ngood_extra_km.extend(kms)
        ngood_extra_nid.extend(aug_nids)

    # test for overlapping points in nbad_extra, ngood_extra
    nb_arr, ng_arr = np.asarray(nbad_extra), np.asarray(ngood_extra)
    nb_km_arr, ng_km_arr = np.asarray(
        nbad_extra_km), np.asarray(ngood_extra_km)
#    for bad_idx, item in enumerate(nb_arr):
#        inds = np.where(nb_arr == 0)[0]
#        if len(z) <= 1:
#            continue
#        else:
#            # find smallest distance if multiple entries
#            kms = nb_km_arr[inds]
#            inds_inds = np.argsort(kms)
#            # sort array indexes, then take all but first (min distance)
#            rem_is = inds[inds_inds][1:]
#            # att this to bad_rem
#            bad_rem.extend(list(rem_is))

    def rem_dups0(n_arr, km_arr):
        rem_list = []
        for i, item in enumerate(n_arr):
            if item in nbad_set or item in ngood_set:
                rem_list.extend([i])
                continue
#            if item == '3138576166':
#                print ("i, item", i, item
            inds = np.where(n_arr == item)[0]
            if len(inds) <= 1:
                continue
            else:
                # print ("inds", inds
                # find smallest distance if multiple entries
                kms = km_arr[inds]
                inds_inds = np.argsort(kms)
                # sort array indexes, then take all but first (min distance)
                rem_is = inds[inds_inds][1:]
                # att this to bad_rem
                rem_list.extend(list(rem_is))

        return rem_list

    bad_rem = rem_dups0(nb_arr, nb_km_arr)
    good_rem = rem_dups0(ng_arr, ng_km_arr)
    # print ("bad_rem", bad_rem

    # remove items
    nbad_extra = list(np.delete(nbad_extra, bad_rem))
    ngood_extra = list(np.delete(ngood_extra, good_rem))
    nbad_extra_nid = list(np.delete(nbad_extra_nid, bad_rem))
    ngood_extra_nid = list(np.delete(ngood_extra_nid, good_rem))
    nbad_extra_km = list(np.delete(nbad_extra_km, bad_rem))
    ngood_extra_km = list(np.delete(ngood_extra_km, good_rem))

    # test if any points are in both good_extra and bad_extra
    bad_rem, good_rem = [], []
    ngood_extra_set = set(ngood_extra)
    for bad_idx, item in enumerate(nbad_extra):
        if item in ngood_extra_set:
            # test which is closer
            good_idx = ngood_extra.index(item)
            dist_good = ngood_extra_km[good_idx]
            dist_bad = nbad_extra_km[bad_idx]
            if dist_bad >= dist_good:
                bad_rem.append(bad_idx)
            else:
                good_rem.append(good_idx)
    # remove overlapping items from lists
    nbad_extra = list(np.delete(nbad_extra, bad_rem))
    ngood_extra = list(np.delete(ngood_extra, good_rem))
    nbad_extra_nid = list(np.delete(nbad_extra_nid, bad_rem))
    ngood_extra_nid = list(np.delete(ngood_extra_nid, good_rem))
    nbad_extra_km = list(np.delete(nbad_extra_km, bad_rem))
    ngood_extra_km = list(np.delete(ngood_extra_km, good_rem))

    # set aug count, val arrays as emtpy string
    nbad_extra_count = np.asarray(
        [str(round(km, 2))+'km' for km in nbad_extra_km])
    nbad_extra_val = np.asarray(len(nbad_extra)*['bad_aug'])
    ngood_extra_count = np.asarray(
        [str(round(km, 2))+'km' for km in ngood_extra_km])
    ngood_extra_val = np.asarray(len(ngood_extra)*['good_aug'])
    # combine
    # bad
    nbad_aug = np.concatenate((nbad, np.asarray(nbad_extra)))
    nbad_aug_nid = np.concatenate((nbad_nid, np.asarray(nbad_extra_nid)))
    nbad_aug_count = np.concatenate((nbad_count, nbad_extra_count))
    nbad_aug_val = np.concatenate((nbad_val, nbad_extra_val))
    # good
    ngood_aug = np.concatenate((ngood, np.asarray(ngood_extra)))
    ngood_aug_nid = np.concatenate((ngood_nid, np.asarray(ngood_extra_nid)))
    ngood_aug_count = np.concatenate((ngood_count, ngood_extra_count))
    ngood_aug_val = np.concatenate((ngood_val, ngood_extra_val))

    # test
    #nzz, czz = np.unique(nbad_aug, return_counts=True)
    #toozz = np.where(czz > 1)
    #nzz_too = nzz[toozz]
    # print ("nzz_too", nzz_too

    print("Augmented points: len nbad", len(nbad), "len nbad_aug",
          len(nbad_aug))
    print("Augmented points: len ngood", len(ngood), "len ngood_aug",
          len(ngood_aug))

    # auglist = [ngood, nbad, ngood_aug, nbad_aug, ngood_aug_nid, nbad_aug_nid,
    #        ngood_aug_count, nbad_aug_count, ngood_aug_val, nbad_aug_val]

    # store in columndatasources
    cds_good_aug = bokeh_utils.set_nodes_source(G, ngood_aug, size=node_size,
                                                fill_alpha=alpha_dic['aug'],
                                                shape=shape, label=[],
                                                count=list(ngood_aug_count),
                                                val=list(ngood_aug_val),
                                                name=list(ngood_aug_nid),
                                                color=color_dic['goodnode_aug_color'])
    cds_bad_aug = bokeh_utils.set_nodes_source(G, nbad_aug, size=node_size,
                                               fill_alpha=alpha_dic['aug'],
                                               shape=shape, label=[],
                                               count=list(nbad_aug_count),
                                               val=list(nbad_aug_val),
                                               name=list(nbad_aug_nid),
                                               color=color_dic['badnode_aug_color'])

    auglist = [ngood, nbad, ngood_aug, nbad_aug, cds_good_aug, cds_bad_aug]

    print("Time to compute auglist:", time.time() - t0, "seconds")
    return auglist


###############################################################################
def choose_target(G, skiplists, direction='north'):
    '''Choose target node in G.nodes() thats not in skiplists
    pick the northernmost (or other direction) of all nodes
    direction options are 'north, south, east, west'
    '''

    # keep only nodes not in skiplists
    skipset = set([item for sublist in skiplists for item in sublist])
    Gset = set(G.nodes())
    nodes = np.asarray(list(Gset-skipset))
    # extract locations,
    xs, ys = [], []
    for n in nodes:
        xs.append(G.nodes[n]['x'])
        ys.append(G.nodes[n]['y'])

    # order by x
    fx = np.argsort(xs)
    nodes_xsort = nodes[fx]

    # order by lon
    fy = np.argsort(ys)
    nodes_ysort = nodes[fy]

    if direction.lower() == 'north':
        node = nodes_ysort[-1]
    elif direction.lower() == 'south':
        node = nodes_ysort[0]
    elif direction.lower() == 'east':
        node = nodes_xsort[-1]
    elif direction.lower() == 'west':
        node = nodes_xsort[0]
    else:
        node = None

    return node


###############################################################################
def choose_target_latlon(G, skiplists, direction='north'):
    '''Choose target node in G.nodes() thats not in skiplists
    pick the northernmost (or other direction) of all nodes
    direction options are 'north, south, east, west'
    '''

    # keep only nodes not in skiplists
    skipset = set([item for sublist in skiplists for item in sublist])
    Gset = set(G.nodes())
    nodes = np.asarray(list(Gset-skipset))
    # extract locations,
    lats, lons = [], []
    for n in nodes:
        lons.append(G.nodes[n]['lon'])
        lats.append(G.nodes[n]['lat'])

    # order by lat
    flat = np.argsort(lats)
    nodes_latsort = nodes[flat]

    # order by lon
    flon = np.argsort(lons)
    nodes_lonsort = nodes[flon]

    if direction.lower() == 'north':
        node = nodes_latsort[-1]
    elif direction.lower() == 'south':
        node = nodes_latsort[0]
    elif direction.lower() == 'east':
        node = nodes_lonsort[-1]
    elif direction.lower() == 'west':
        node = nodes_lonsort[0]
    else:
        node = None

    return node


###############################################################################
def G_to_csv(G, outfile, delim='|'):
    '''
    Create csv for import into Tableau
    Assume encoding is unicode: utf-8
    '''

    print("Printing networkx graph to csv...")
    t0 = time.time()

    header = ['E_ID', 'Node', 'Source', 'Target', 'Link', 'Node Lat',
              'Node Lat2', 'Node Lon', 'Node Type', 'Node Degree',
              'Node Eigenvector Centrality', 'Road Type', 'Road Name', 'Bridge',
              'Ref', 'Num Lanes', 'Edge Length (km)', 'Path Length (km)',
              'Max Speed (km/h)', 'Travel Time (h)']
#    header = ['E_ID', 'Node', 'Source', 'Target', 'Link', 'Node Lat', \
#        'Node Lat2', 'Node Lon', 'Source Lat', 'Source Lon', 'Target Lat', \
#        'Target Lon', 'Road Type', 'Road Name', \
#        'Ref', 'Num Lanes', 'Path Length (km)', 'Max Speed (km/h)', \
#        'Travel Time (h)']
    colno = len(header)

    # compute node properties
    G = shackleton_create_g.node_props(G)

    #fout = open(outfile, 'w')
    #fout = codecs.open(outfile, "w", "utf-8")
    #fout.write(stringfromlist(header, delim) + '\n')

    # csv writer can't handle unicode!
    #writer = csv.writer(fout, delimiter = delim)

    # another option: unicodecsv
    fout = open(outfile, 'w')
    writer = unicodecsv.writer(fout, encoding='utf-8', delimiter=delim)
    writer.writerow(header)

    for i, e in enumerate(G.edges()):
        s, t = e
        e_props = G.edge[s][t]
        s_props = G.nodes[s]
        t_props = G.nodes[t]

        if G.edge[s][t]['e_id'].startswith('31242722'):
            print("i, edge, s, t", i, e, s, t)
            print("edge_props", G.edge[s][t])
            print("s_props", G.nodes[s])
            print("t_props", G.nodes[t])
            print('\n')

        # node properties
        slat, slon = s_props['lat'], s_props['lon']
        tlat, tlon = t_props['lat'], t_props['lon']
        stype, ttype = s_props['ntype'], t_props['ntype']
        sdeg, tdeg = s_props['deg'], t_props['deg']
        # G.nodes[s]['Eigenvector Centrality'], G.nodes[t]['Eigenvector Centrality']
        seig, teig = 0, 0,
        # edge properties
        link = e_props['Link']
        e_id = e_props['e_id']
        roadtype = e_props['Road Type']
        roadname = e_props['Road Name']
        bridge = e_props['Bridge']
        ref = e_props['ref']
        numlanes = e_props['Num Lanes']
        edgelen = e_props['Edge Length (km)']
        pathlen = e_props['Path Length (km)']
        #roadlen = e_props['Road Length (km)']
        maxspeed = e_props['Max Speed (km/h)']
        traveltime = e_props['Travel Time (h)']

        # initial row
        row = [e_id, s, s, t, link, slat, slat, slon, stype, sdeg, seig, roadtype,
               roadname, bridge, ref, numlanes, edgelen, pathlen, maxspeed, traveltime]
        if len(row) != colno:
            print("len header", len(header), "header", header)
            print("malformed row!:", "len", len(row), row)
            return
        # convert all ints and floats to strings prior to printing
        for j, item in enumerate(row):
            if type(item) == float or type(item) == int:
                string = str(item)
                row[j] = string  # .encode('utf-8', 'ignore')
            # print ("item, type(item)", row[i], type(row[i])

        # also enter target - source row
        rowrev = [e_id, t, s, t, link, tlat, tlat, tlon, ttype, tdeg, teig, roadtype,
                  roadname, bridge, ref, numlanes, edgelen, pathlen, maxspeed, traveltime]
        if len(rowrev) != colno:
            print("len header", len(header), "header", header)
            print("malformed reverse row!:", "len", len(rowrev), rowrev)
            return
        # convert all ints and floats to strings prior to printing
        for k, item in enumerate(rowrev):
            if type(item) == float or type(item) == int:
                rowrev[k] = str(item)

        writer.writerow(row)
        writer.writerow(rowrev)

        #outstring = stringfromlist(row, delim)
        #outstring2 = stringfromlist(rowrev, delim)
        #fout.write(outstring + '\n')
        #fout.write(outstring2 + '\n')

        # if (i % 1000) == 0:
        #    print i, "row:", row

    fout.close()

    print("Time to print", outfile, "graph to csv:", time.time() - t0, "seconds")


###############################################################################
def G_to_kdtree(G, coords='wmp', node_subset=[]):
    '''
    Create kd tree from node positions
    (x, y) = (lon, lat)
    return kd tree and kd_idx_dic
    kd_idx_dic maps kdtree entry to node name: kd_idx_dic[i] = n (n in G.nodes())
    node_subset is a list of nodes to consider, use all if []
    '''
    nrows = len(G.nodes())
    ncols = 2
    kd_idx_dic = {}
    arr = np.zeros((nrows, ncols))
    # populate node array
    t1 = time.time()

    i = 0
    # for i, (node, data) in enumerate(list(G.nodes(data=True))):
    for (node, data) in list(G.nodes(data=True)):
        if len(node_subset) > 0:
            if node not in node_subset:
                continue
        if coords == 'wmp':
            x, y = data['x'], data['y']
        elif coords == 'latlon':
            x, y = data['lon'], data['lat']
        arr[i] = [x, y]
        kd_idx_dic[i] = node
        i += 1

    # for i,n in enumerate(G.nodes()):
    #    n_props = G.nodes[n]
    #    lat, lon = n_props['lat'], n_props['lon']
    #    x, y = lon, lat
    #    arr[i] = [x,y]
    #    kd_idx_dic[i] = n

    # now create kdtree from numpy array
    kdtree = scipy.spatial.KDTree(arr)

    print("Time to create k-d tree:", time.time() - t1, "seconds")

    return kd_idx_dic, kdtree  # , arr


###############################################################################
def get_G_extent(G):
    '''Get extent of graph
    return [x0, y0, x1, y1]'''

    node_Xs = [float(x) for _, x in G.nodes(data='x')]
    node_Ys = [float(y) for _, y in G.nodes(data='y')]

    xmin, xmax = np.min(node_Xs), np.max(node_Xs)
    ymin, ymax = np.min(node_Ys), np.max(node_Ys)

    return xmin, ymin, xmax, ymax


###############################################################################
def make_ecurve_dic(G):
    '''Create dictionary of plotting coordinates for each graph edge
        node_Ys = [float(y) for _, y in G.nodes(data='y')]
    '''

    ecurve_dic = {}

    for u, v, data in G.edges(data=True):
        #print ("u,v,data:", u,v,data)
        if 'geometry' in data.keys():
            geom = data['geometry']
            coords = np.array(list(geom.coords))
            xs = coords[:, 0]
            ys = coords[:, 1]
        else:
            xs = np.array([G.nodes[u]['x'], G.nodes[v]['x']])
            ys = np.array([G.nodes[u]['y'], G.nodes[v]['y']])

        ecurve_dic[(u, v)] = (xs, ys)
        # also include reverse edge in list of coords?
        #ecurve_dic[(v,u)] = (xs, ys)

    return ecurve_dic


###############################################################################
def get_osm_data(poly, filenames, download_osm=True):
    '''Get osm data'''

    t00 = time.time()
    [root, outroot, indir, outdir, gdelt_infile, osmfile, queryfile,
     html_raw, html_ref, html_ref_straight, html_gdelt, html_aug] = filenames

    # Download map
    if download_osm:
        print("Downloading Open Street Maps data...")
        construct_poly_query(poly, queryfile)
        t0 = time.time()
        download_osm_query(queryfile, osmfile)
        print("Time to download graph =", time.time()-t0, "seconds")

    # Create osm, dictionaries
    t0 = time.time()
    osm = shackleton_create_g.OSM(osmfile)  # extremely fast
    node_histogram, ntype_dic, X_dic = shackleton_create_g.intersect_dic(osm)
    print("Time to create osm, dictionaries:", time.time()-t0, "seconds")

    ###################
    # full graph
    t1 = time.time()
    G0 = shackleton_create_g.read_osm(osm)
    t2 = time.time()
    print("Time to create raw graph =", t2-t1, "seconds")
    # print to csv
    #graph_out = outdir + root + '_full.graphml'
    #csvout =  outdir + root + '_full.csv'
    #G_to_csv(G0, csvout)
    ###################

    ###################
    # refined graph
    # test_refine_osm(osmfile)
    t2 = time.time()
    G, ecurve_dic = shackleton_create_g.refine_osm(
        osm, node_histogram, ntype_dic, X_dic)  # (osmfile)
    # print ("ecurve_dic", ecurve_dic
    # print ("G", G
    t3 = time.time()
    print("Time to create refined graph =", t3-t2, "seconds")
    # print to csv
    #graph_out = outdir + root + '.graphml'
    #csvout =  outdir + root + '.csv'
    #G_to_csv(G, csvout)
    ###################

    ###################
    # crate kdtree of nodes
    kd_idx_dic, kdtree = G_to_kdtree(G)
    ###################

    print("Total time to load data:", time.time()-t00, "seconds")
    # return data
    return osm, node_histogram, ntype_dic, X_dic, G0, G, ecurve_dic, \
        kd_idx_dic, kdtree


###############################################################################
### From apls.py
###############################################################################
def create_edge_linestrings(G, remove_redundant=True, verbose=False):
    '''Ensure all edges have 'geometry' tag, use shapely linestrings
    If identical edges exist, remove extras'''

    # clean out redundant edges with identical geometry
    edge_seen_set = set([])
    geom_seen = []
    bad_edges = []

    G_ = G.copy()
    for i, (u, v, key, data) in enumerate(G_.edges(keys=True, data=True)):
        # create linestring if no geometry reported
        if 'geometry' not in data:
            sourcex, sourcey = G_.nodes[u]['x'],  G_.nodes[u]['y']
            targetx, targety = G_.nodes[v]['x'],  G_.nodes[v]['y']
            line_geom = LineString([Point(sourcex, sourcey),
                                    Point(targetx, targety)])
            data['geometry'] = line_geom

            # get reversed line
            coords = list(data['geometry'].coords)[::-1]
            line_geom_rev = LineString(coords)
            #G_.edge[u][v]['geometry'] = lstring

        else:
            # check which direction linestring is travelling (it may be going from
            # v -> u, which means we need to reverse the linestring)
            # otherwise splitting this edge yields a tangled edge
            line_geom = data['geometry']
            coords = list(data['geometry'].coords)[::-1]
            line_geom_rev = LineString(coords)

        # flag redundant edges
        if remove_redundant:
            if i == 0:
                edge_seen_set = set([(u, v)])
                edge_seen_set.add((v, u))
                geom_seen.append(line_geom)

            else:
                if ((u, v) in edge_seen_set) or ((v, u) in edge_seen_set):
                    # test if geoms have already been seen
                    for geom_seen_tmp in geom_seen:
                        if (line_geom == geom_seen_tmp) \
                                or (line_geom_rev == geom_seen_tmp):
                            bad_edges.append((u, v, key))
                            if verbose:
                                print("\nRedundant edge:", u, v, key)
                else:
                    edge_seen_set.add((u, v))
                    geom_seen.append(line_geom)
                    geom_seen.append(line_geom_rev)

    if remove_redundant:
        if verbose:
            print("\nedge_seen_set:", edge_seen_set)
            print("redundant edges:", bad_edges)
        for (u, v, key) in bad_edges:
            try:
                G_.remove_edge(u, v, key)
            except:
                if verbose:
                    print("Edge DNE:", u, v, key)
                pass

    return G_


###############################################################################
def cut_linestring(line, distance, verbose=False):
    '''
    Cuts a line in two at a distance from its starting point
    http://toblerity.org/shapely/manual.html#linear-referencing-methods
    '''
    """
    Cuts a shapely linestring at a specified distance from its starting point.

    Notes
    ----
    Return orignal linestring if distance <= 0 or greater than the length of
    the line.
    Reference:
        http://toblerity.org/shapely/manual.html#linear-referencing-methods

    Arguments
    ---------
    line : shapely linestring
        Input shapely linestring to cut.
    distanct : float
        Distance from start of line to cut it in two.
    verbose : boolean
        Switch to print relevant values to screen.  Defaults to ``False``.

    Returns
    -------
    [line1, line2] : list
        Cut linestrings.  If distance <= 0 or greater than the length of
        the line, return input line.
    """

    if verbose:
        print("Cutting linestring at distance", distance, "...")
    if distance <= 0.0 or distance >= line.length:
        return [LineString(line)]

    # iterate through coorda and check if interpolated point has been passed
    # already or not
    coords = list(line.coords)
    for i, p in enumerate(coords):
        pdl = line.project(Point(p))
        if verbose:
            print(i, p, "line.project point:", pdl)
        if pdl == distance:
            return [
                LineString(coords[:i+1]),
                LineString(coords[i:])]
        if pdl > distance:
            cp = line.interpolate(distance)
            return [
                LineString(coords[:i] + [(cp.x, cp.y)]),
                LineString([(cp.x, cp.y)] + coords[i:])]

    # if we've reached here then that means we've encountered a self-loop and
    # the interpolated point is between the final midpoint and the the original
    # node
    i = len(coords) - 1
    cp = line.interpolate(distance)
    return [
        LineString(coords[:i] + [(cp.x, cp.y)]),
        LineString([(cp.x, cp.y)] + coords[i:])]


###############################################################################
def get_closest_edge_from_list(G_, edge_list_in, point, verbose=False):
    '''Return closest edge to point, and distance to said edge, from a list
    of possible edges
    Just discovered a similar function: 
        https://github.com/gboeing/osmnx/blob/master/osmnx/utils.py#L501'''

    # get distances from point to lines
    dist_list = []
    edge_list = []
    geom_list = []
    p = point # Point(point_coords)
    #print ("edge_list:", edge_list_in)
    for i, (u, v) in enumerate(edge_list_in):
        data = G_.get_edge_data(u ,v)
        
        if verbose:
            print(("get_closest_edge_from_list()  u,v,data:", u,v,data))
            #print ("data[0]:", data[0])
            #print ("data[0].keys:", data[0].keys())
            #print ("data[0]['geometry']:", data[0]['geometry'])
            #print(("  type data['geometry']:", type(data['geometry'])))
            
        try:
            line = data[0]['geometry']
        except:
            line = data['geometry']
            #line = data['attr_dict']['geometry']
        geom_list.append(line)
        dist_list.append(p.distance(line))
        edge_list.append([u, v])
    # get closest edge
    min_idx = np.argmin(dist_list)
    min_dist = dist_list[min_idx]
    best_edge = edge_list[min_idx]
    best_geom = geom_list[min_idx]

    return best_edge, min_dist, best_geom


###############################################################################
def insert_point_into_edge(G_, edge, point, node_id=100000,
                           max_distance_meters=10,
                           allow_renaming_once=False,
                           verbose=False, super_verbose=False):
    '''
    Insert a new node in the edge closest to the given point, if it is
    within max_distance_meters.  Assume all edges have a linestring geometry
    http://toblerity.org/shapely/manual.html#object.simplify
         
     Sometimes the point to insert will have the same coordinates as an 
     existing point.  If allow_renaming_once == True, relabel the existing 
     node once (after that add new nodes coincident and with edge length 0)
     # actually, renaming screws up dictionary of closest edges!!
     
    Return updated G_, 
                node_props, 
                min_dist,
                edges ([u1,v1], [u2,v2]), 
                list of edge_props,
                removed edge
    '''

    # check if node_id already exists in G
    G_node_set = set(G_.nodes())
    if node_id in G_node_set:
        print ("node_id:", node_id, "already in G, cannot insert node!")
        return
    
    # check if edge even exists
    u, v = edge
    if not G_.has_edge(u,v):
        print ("edge {} DNE!".format((u,v)))
        return
 
    p = Point(point[0], point[1])
    edge_props = G_.get_edge_data(u,v)
    try:
        line_geom = edge_props['geometry']
    except:
        line_geom = edge_props[0]['geometry']
    min_dist = p.distance(line_geom)
                 
    if verbose:
        print("Inserting point:", node_id, "coords:", point)
        print("best edge:", edge)
        print ("edge_props:", edge_props)
        print("  best edge dist:", min_dist)
        #u_loc = [G_.nodes[u]['x'], G_.nodes[u]['y']]
        #v_loc = [G_.nodes[v]['x'], G_.nodes[v]['y']]
        #print("ploc:", (point.x, point.y))
        #print("uloc:", u_loc)
        #print("vloc:", v_loc)
    
    if min_dist > max_distance_meters:
        if verbose:
            print("min_dist > max_distance_meters, skipping...")
        return G_, {}, min_dist, [], [], ()
    
    else:
        # update graph
        
        ## skip if node exists already
        #if node_id in G_node_set:
        #    if verbose:
        #        print("Node ID:", node_id, "already exists, skipping...")
        #    return #G_, {}, -1, -1

        # Length along line that is closest to the point
        line_proj = line_geom.project(p)

        # Now combine with interpolated point on line
        new_point = line_geom.interpolate(line_geom.project(p))
        x, y = new_point.x, new_point.y
        
        #################
        # create new node
        
        try:
            # first get zone, then convert to latlon
            _, _, zone_num, zone_letter = utm.from_latlon(G_.nodes[u]['lat'],
                                                          G_.nodes[u]['lon'])
            # convert utm to latlon
            lat, lon = utm.to_latlon(x, y, zone_num, zone_letter)
        except:
            lat, lon = y, x
        
        # set properties
        #props = G_.nodes[u]
        node_props = {'highway': 'insertQ',
                 'lat':     lat,
                 'lon':     lon,
                 'osmid':   node_id,
                 'x':       x,
                 'y':       y}
        ## add node
        ##G_.add_node(node_id, **node_props)
        #G_.add_node(node_id, **node_props)
        #
        ## assign, then update edge props for new edge
        #data = G_.get_edge_data(u ,v)
        #_, _, edge_props_new = copy.deepcopy(list(G_.edges([u,v], data=True))[0])
        ## remove extraneous 0 key
        
        #print ("edge_props_new.keys():", edge_props_new)
        #if list(edge_props_new.keys()) == [0]:
        #    edge_props_new = edge_props_new[0]
 
        # cut line
        split_line = cut_linestring(line_geom, line_proj)
        #line1, line2, cp = cut_linestring(line_geom, line_proj)
        if split_line == None:
            print("Failure in cut_linestring()...")
            print("type(split_line):", type(split_line))
            print("split_line:", split_line)
            print("line_geom:", line_geom)
            print("line_geom.length:", line_geom.length)
            print("line_proj:", line_proj)
            print("min_dist:", min_dist)
            return G_, {}, min_dist, [], [], ()

        if verbose:
            print("split_line:", split_line)
        
        #if cp.is_empty:        
        if len(split_line) == 1:
            if verbose:
                print("split line empty, min_dist:", min_dist)
            # get coincident node
            outnode = ''
            outnode_x, outnode_y = -1, -1
            x_p, y_p = new_point.x, new_point.y
            x_u, y_u = G_.nodes[u]['x'], G_.nodes[u]['y']
            x_v, y_v = G_.nodes[v]['x'], G_.nodes[v]['y']
            #if verbose:
            #    print "x_p, y_p:", x_p, y_p
            #    print "x_u, y_u:", x_u, y_u
            #    print "x_v, y_v:", x_v, y_v
            
            # sometimes it seems that the nodes aren't perfectly coincident,
            # so see if it's within a buffer
            buff = 0.05 # meters
            if (abs(x_p - x_u) <= buff) and (abs(y_p - y_u) <= buff):
                outnode = int(u)
                outnode_x, outnode_y = x_u, y_u
                # set node_props x,y as existing node
                node_props['x'] = outnode_x
                node_props['y'] = outnode_y
                #return G_, node_props, min_dist, [], [], ()
            elif (abs(x_p - x_v) <= buff) and (abs(y_p - y_v) <= buff):
                outnode = int(v)
                outnode_x, outnode_y = x_v, y_v
                # set node_props x,y as existing node
                node_props['x'] = outnode_x
                node_props['y'] = outnode_y
                #return G_, node_props, min_dist, [], [], ()
            else:
                print("Error in determining node coincident with node: " \
                + str(node_id) + " along edge: " + str(edge))
                print("x_p, y_p:", x_p, y_p)
                print("x_u, y_u:", x_u, y_u)
                print("x_v, y_v:", x_v, y_v)
                return #G_, (), {}, [], []
            
            if verbose:
                print ("u, v, outnode:", u, v, outnode)
                #print ("allow remaning?", allow_renaming)
                
            # if the line cannot be split, that means that the new node 
            # is coincident with an existing node.  Relabel, if desired
            # only relabel if the node value is positive.  If it's negative,
            # we'll invoke the next clause and add a new node and edge of 
            # length 0.  We do this because there could be multiple objects 
            # that have the nearest point in the graph be an original node,
            # so we'll relabel it once
            if outnode > 0 and allow_renaming_once:
                node_props = G_.nodes[outnode]
                # A dictionary with the old labels as keys and new labels as values. A partial mapping is allowed.
                mapping = {outnode: node_id}
                Gout = nx.relabel_nodes(G_, mapping)
                if verbose:
                    print("Swapping out node ids:", mapping)
                return Gout, node_props, min_dist, [], [], ()
            
            else:
            #elif 1 > 2:
                # new node is already added, presumably at the exact location
                # of an existing node.  So just  make
                # an edge from new node to existing node, length should be 0.0
                line1 = LineString([new_point, Point(outnode_x, outnode_y)])
                best_data = G_.get_edge_data(u ,v)[0]
                edge_props_line1 = copy.deepcopy(best_data)
                #edge_props_line1 = edge_props.copy()         
                edge_props_line1['length'] = line1.length
                edge_props_line1['geometry'] = line1
                edge_props_line1['travel_time'] = 0.0
                # make sure length is zero
                if line1.length > buff:
                    print("Nodes should be coincident and length 0!")
                    print ("  line1.length:", line1.length)
                    print ("  x_u, y_u :", x_u, y_u )
                    print ("  x_v, y_v :", x_v, y_v )
                    print ("  x_p, y_p :", x_p, y_p )
                    print ("  new_point:", new_point)
                    print ("  Point(outnode_x, outnode_y):", Point(outnode_x, outnode_y))
                    return
                
                if verbose:
                    print("add edge of length 0 from new node to nearest existing node")
                    print ("line1.length:", line1.length)
                G_.add_node(node_id, **node_props)
                # print ("type node_id:", type(node_id))
                # print ("type outonde:", type(outnode))
                # print (edge_props_line1:", edge_props_line1)
                G_.add_edge(node_id, outnode, **edge_props_line1)
                return G_, node_props, min_dist, \
                        [(node_id, outnode)], [edge_props_line1], \
                        ()
        
        
        else:
            # add node
            G_.add_node(node_id, **node_props)
            
            # assign, then update edge props for new edge
            best_data = G_.get_edge_data(u ,v)[0]
            edge_props_new = copy.deepcopy(best_data)
            if verbose:
                print ("edge_props_new:", edge_props_new)

            #_, _, edge_props_new = copy.deepcopy(list(G_.edges([u,v], data=True))[0])
            # remove extraneous 0 key
                # else, create new edges
                
            line1, line2 = split_line

            # get distances
            #print ("insert_point(), G_.nodes[v]:", G_.nodes[v])
            u_loc = [G_.nodes[u]['x'], G_.nodes[u]['y']]
            v_loc = [G_.nodes[v]['x'], G_.nodes[v]['y']]
            # compare to first point in linestring
            geom_p0 = list(line_geom.coords)[0]
            # or compare to inserted point? [this might fail if line is very
            #    curved!]
            #geom_p0 = (x,y)
            dist_to_u = scipy.spatial.distance.euclidean(u_loc, geom_p0)
            dist_to_v = scipy.spatial.distance.euclidean(v_loc, geom_p0)
            # reverse edge order if v closer than u
            if dist_to_v < dist_to_u:
                line2, line1 = split_line
                
            if verbose:
                print("Creating two edges from split...")
                print("   original_length:", line_geom.length)
                print("   line1_length:", line1.length)
                print("   line2_length:", line2.length)
                print("   u, dist_u_to_point:", u, dist_to_u)
                print("   v, dist_v_to_point:", v, dist_to_v)
                print("   min_dist:", min_dist)

            # add new edges
            edge_props_line1 = edge_props_new.copy()
            edge_props_line1['length'] = line1.length
            edge_props_line1['geometry'] = line1
            # remove geometry?
            #edge_props_line1.pop('geometry', None) 
            # line2
            edge_props_line2 = edge_props_new.copy()
            edge_props_line2['length'] = line2.length
            edge_props_line2['geometry'] = line2
            # remove geometry?
            #edge_props_line1.pop('geometry', None) 

            # insert edge regardless of direction
            #G_.add_edge(u, node_id, **edge_props_line1)
            #G_.add_edge(node_id, v, **edge_props_line2)
            
            # check which direction linestring is travelling (it may be going from
            # v -> u, which means we need to reverse the linestring)
            # otherwise new edge is tangled
            geom_p0 = list(line_geom.coords)[0]
            dist_to_u = scipy.spatial.distance.euclidean(u_loc, geom_p0)
            dist_to_v = scipy.spatial.distance.euclidean(v_loc, geom_p0)
            #if verbose:
            #    print "dist_to_u, dist_to_v:", dist_to_u, dist_to_v
            if dist_to_u < dist_to_v:
                G_.add_edge(u, node_id, **edge_props_line1)
                G_.add_edge(node_id, v, **edge_props_line2)
                edge_list = [(u, node_id), (node_id, v)]
            else:
                G_.add_edge(node_id, u, **edge_props_line1)
                G_.add_edge(v, node_id, **edge_props_line2)
                edge_list = [(node_id, u), (v, node_id)]

            if verbose:
                print("insert edges:", u, '-',node_id, 'and', node_id, '-', v)
                         
            # remove initial edge
            rem_edge = (u,v)
            if verbose:
                print ("removing edge:", rem_edge)
            G_.remove_edge(u, v)
                        
            return G_, node_props, min_dist, edge_list, \
                            [edge_props_line1, edge_props_line2], rem_edge
