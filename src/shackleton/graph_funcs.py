# -*- coding: utf-8 -*-
"""
Created on Wed Dec 17 11:09:58 2014
@author: avanetten
"""

import sys, time, os, csv, math, webbrowser
# import  unicodecsv
import scipy.spatial
import numpy as np
import copy
import pandas as pd
import networkx as nx
import bokeh.plotting as bk
from shapely.geometry import Point, LineString
from scipy import stats
from sklearn.gaussian_process import GaussianProcessRegressor
from scipy.ndimage import gaussian_filter1d, gaussian_filter
from matplotlib import pyplot as pl
from matplotlib import cm
from bokeh.plotting import figure, curdoc
from bokeh.models import (Plot, \
    GMapPlot, GMapOptions, Range1d, LinearAxis, Grid, DataRange1d, \
    PanTool, WheelZoomTool, BoxZoomTool, ResetTool, BoxSelectTool, \
    HoverTool, #ResizeTool, 
    SaveTool, Legend, Axis, CategoricalAxis, 
    UndoTool, RedoTool,
    CategoricalTicker, FactorRange,
    #WMTSTileSource, BBoxTileSource,
    ColumnDataSource)#, GeoJSOptions, GeoJSPlot)
from bokeh.models.glyphs import (Square, Circle, Triangle, InvertedTriangle, 
                    Segment, Patch, Diamond, Text, Rect, Patches, MultiLine)
from bokeh.tile_providers import STAMEN_TONER, STAMEN_TERRAIN, CARTODBPOSITRON_RETINA
#from bokeh.io import curstate, set_curdoc
from bokeh.document import Document
# from pyproj import Proj, transform
from bokeh.layouts import column

import bokeh_utils
import graph_init
import concave_hull
from utils import (distance, latlon_to_wmp, lin_scale, log_scale,
                   log_transform, value_by_key_prefix, query_kd,
                   query_kd_ball, construct_poly_query,
                   download_osm_query, global_vars)
from bokeh_utils import (
    G_to_bokeh,
    define_colors_alphas,
    get_route_glyphs_all,
    get_paths_glyph_line,
    get_paths_glyph_seg,
    get_nodes_glyph,
    get_gdelt_glyph,
    get_bbox_glyph,
    add_hover_save,
    set_routes_title,
    plot_nodes,
    plot_histo,
    keep_bbox_indices
    )

global_dic = global_vars()
csv.field_size_limit(1000000000)
###############################################################################    


###############################################################################
### Routing
###############################################################################
def get_path_pos(G, paths, lengths, skipset=set()):
    '''Create path position arrays from path, length arrays
    Optionally, skip items in skipset'''
    
    edgelist, ex0, ey0, ex1, ey1, elat0, elon0, elat1, elon1, elen =\
                        [],[],[],[],[],[],[],[],[],[]
    edgeset = set()
    for path,length in zip(paths,lengths):
        # for each path, plot compute segments
        for l in range(len(path)-1):
            n0, n1 = path[l], path[l+1]
            ed, edrev = (n0,n1), (n1,n0)
            # only add edge if it's not already in the list
            if ed not in edgeset and edrev not in edgeset \
                        and ed not in skipset and ed not in skipset:
                ex0.append(G.nodes[n0]['x'])
                ey0.append(G.nodes[n0]['y'])                    
                ex1.append(G.nodes[n1]['x'])
                ey1.append(G.nodes[n1]['x'])                     
                elon0.append(G.nodes[n0]['lon'])
                elat0.append(G.nodes[n0]['lat'])                    
                elon1.append(G.nodes[n1]['lon'])
                elat1.append(G.nodes[n1]['lat'])  
                elen.append(length)
                edgelist.append(ed)
                edgeset.add(ed)
            
    emx = [(a + b)/2. for a, b in zip(ex0, ex1)] #np: 0.5*(ex0+ex1)
    emy = [(a + b)/2. for a, b in zip(ey0, ey1)] #np: 0.5*(ey0+ey1)

    # convert to numpy arrays (except for edgelist)
    ex0 = np.asarray(ex0)
    ex1 = np.asarray(ex1)
    ey0 = np.asarray(ey0)
    ey1 = np.asarray(ey1)
    elat0 = np.asarray(elat0)
    elat1 = np.asarray(elat1)
    elon0 = np.asarray(elon0)
    elon1 = np.asarray(elon1)
    emx = np.asarray(emx)
    emy = np.asarray(emy)
    elen = np.asarray(elen)
    
    return ex0, ey0, ex1, ey1, elat0, elon0, elat1, elon1, emx, emy, elen, edgelist
    
    
###############################################################################
def get_path_pos_elist(G, input_e, len_key, skipset=set()):
    '''
    Similar to get_path_pos() but with different input
    Create path position arrays from path, length arrays
    Optionally, skip items in skipset
    Input should be a list of edges of form [(s0,t0), (s1,t1)]
    len_key is the dictionary key for edgelength'''
    
    edgelist, ex0, ey0, ex1, ey1, elat0, elon0, elat1, elon1, elen =\
                        [],[],[],[],[],[],[],[],[],[]
    edgeset = set()
    for edge in input_e:
        n0, n1 = edge
        ed, edrev = (n0,n1), (n1,n0)
        # only add edge if it's not already in the list
        if ed not in edgeset and edrev not in edgeset \
                    and ed not in skipset and ed not in skipset:
            ex0.append(G.nodes[n0]['x'])
            ey0.append(G.nodes[n0]['y'])                    
            ex1.append(G.nodes[n1]['x'])
            ey1.append(G.nodes[n1]['x'])                     
            elon0.append(G.nodes[n0]['lon'])
            elat0.append(G.nodes[n0]['lat'])                    
            elon1.append(G.nodes[n1]['lon'])
            elat1.append(G.nodes[n1]['lat'])                      
            elen.append(G.edge[n0][n1][len_key])
            edgelist.append(ed)
            edgeset.add(ed)
            
    emx = [(a + b)/2. for a, b in zip(ex0, ex1)] #np: 0.5*(ex0+ex1)
    emy = [(a + b)/2. for a, b in zip(ey0, ey1)] #np: 0.5*(ey0+ey1)
    
    # convert to numpy arrays (except for edgelist)
    ex0 = np.asarray(ex0)
    ex1 = np.asarray(ex1)
    ey0 = np.asarray(ey0)
    ey1 = np.asarray(ey1)
    elat0 = np.asarray(elat0)
    elat1 = np.asarray(elat1)
    elon0 = np.asarray(elon0)
    elon1 = np.asarray(elon1)
    emx = np.asarray(emx)
    emy = np.asarray(emy)
    elen = np.asarray(elen)
    
    return ex0, ey0, ex1, ey1, elat0, elon0, elat1, elon1, emx, emy, elen, edgelist


###############################################################################
def get_edge_coords():
    pass


###############################################################################
def get_ecurves(edgelist, ecurve_dic):
    '''Take the edgelist from get_path_pos and get the curves from ecurve_dic
    for each edge'''
    
    elx0 = []
    ely0 = []
    ellat0 = []
    ellon0 = []

    for edge in edgelist:
        (s,t) = edge
        edge_rev = (t,s)
        # retrieve original waypoints of edge
        try:
            xtmp, ytmp = ecurve_dic[edge]
        except:
            xtmp, ytmp = ecurve_dic[edge_rev]
#        # not sure which order s,t will be in since graph is undirected
#        try:
#            x_orig, y_orig = ecurve_dic[(s,t)]
#        except:
#            lat_orig, lon_orig = ecurve_dic[(t,s)]
                                            
        # append arrays to line
        elx0.append(xtmp)
        ely0.append(ytmp)
        ellon0.append([]) #(lon_orig)
        ellat0.append([]) #(lat_orig)
        
    return elx0, ely0, ellat0, ellon0    


###############################################################################
def get_ecurves_latlon(edgelist, ecurve_dic):
    '''Take the edgelist from get_path_pos and get the curves from ecurve_dic
    for each edge'''
    
    elx0 = []
    ely0 = []
    ellat0 = []
    ellon0 = []

    for edge in edgelist:
        (s,t) = edge
        # retrieve original waypoints of edge
        # not sure which order s,t will be in since graph is undirected
        try:
            lat_orig, lon_orig = ecurve_dic[(s,t)]
        except:
            lat_orig, lon_orig = ecurve_dic[(t,s)]

        # convert coords if needed
        xtmp, ytmp = latlon_to_wmp(lat_orig, lon_orig)                                            
                                            
        # append arrays to line
        elx0.append(xtmp)
        ely0.append(ytmp)
        ellon0.append(lon_orig)
        ellat0.append(lat_orig)
        
    return elx0, ely0, ellat0, ellon0    
    
    
#http://bytes.com/topic/python/answers/686846-dict-trick        
# can also use dic.get(key, default_value)
###############################################################################
def compute_paths(G, nodes, ecurve_dic=None, target=None, skipnodes=[], \
            weight='Travel Time (h)', alt_G=None, goodroutes=None,
            verbose=False):
    '''Compute and plot all shortest paths between nodes
    if target=None, compute all paths between set(nodes), else only find paths 
    between set(nodes) and target
    skipnodes is an option to remove certain nodes from the graph because 
        nodes may be impassable or blocked
    all plotting is external to this function
    return esource, sourcenodes, paths, lengths, sourcenode_vals, missingnodes
    alt_G is a separate graph (with original edge weights) to use for 
        computing path lengths'''

    if verbose:
        print("compute_paths() - weight:", weight)

    color_dic, alpha_dic = define_colors_alphas()
    global_dic = global_vars()
    
    # make sure target is not in skipnodes!
    if len(skipnodes) != 0 and target is not None:
        skipnodes = list( set(skipnodes)  - set([target]) ) 
    
    # set path color
    if goodroutes == True:
        line_color = color_dic['compute_path_color_good']
    elif goodroutes == False:
        line_color = color_dic['compute_path_color_bad']
    else:
        line_color=color_dic['compute_path_color']
    # set widths
    const_width = True    # all paths are same width
    line_width = global_dic['compute_path_width']
    # set alpha
    if alt_G:
        line_alpha = alpha_dic['compute_paths_sec']
    else:   
        line_alpha = alpha_dic['compute_paths']
    #line_alpha=0.4
 
#    # copy graph  (otherwise changes are global?????)
#    G2 = G#.copy()
#    # remove desired nodes
#    if len(skipnodes) != 0:
#        G2.remove_nodes_from(skipnodes)

    # copy graph  (otherwise changes are global?????)
    # remove desired nodes
    if len(skipnodes) != 0:
        G2 = G.copy()
        G2.remove_nodes_from(skipnodes)
    else:
        G2 = G

    if target in skipnodes:
        print ("ERROR!!")
        print ("target", target)
        print ("skipnodes", skipnodes)
        
    # plot routes from nodes to target
    if target:
        t1 = time.time()
        lengthd, pathd = nx.single_source_dijkstra(G2, source=target, weight=weight) 
        # not all nodes may be reachable from N0, so find intersection
        sourcenodes = list(set(pathd.keys()).intersection(set(nodes)))
        missingnodes = list(set(nodes) - set(sourcenodes))
        #missingnodes_count = [g_node_props_dic[n]['count'] for n in missingnodes]
        paths = [pathd[k] for k in sourcenodes]
        lengths = [lengthd[k] for k in sourcenodes]
        # if alt_G, recompute lengths
        if alt_G:
            lengths = compute_path_lengths(alt_G, paths, weight=weight)
        # set vals as lengths
        sourcenode_vals = [str(round(l,2)) + 'H' for l in lengths]
            
    # plot LOCs between set of nodes
    else:     
        # alternative: overkill to compute all paths, is actually faster...
        t1 = time.time()
        paths, lengths, sourcenode_vals, sourcenodes, missingnodes = [],[], [], [], []
        #missingnodes = set([])
        for k,n0 in enumerate(nodes):
            # get all paths from source
            lengthd, pathd = nx.single_source_dijkstra(G2, source=n0, weight=weight) 
            # check if node is cut off from all other nodes, if so
            # the path dictionary will only contain the source node
            
            # not all nodes may be reachable from N0, so find intersection
            startnodes = list(set(pathd.keys()).intersection(set(nodes)) - set([n0]))
            #print ("n0, len(startnodes)", n0, len(startnodes)
            if len(startnodes) == 0:
                missingnodes.append(n0)
                print ("Node", n0, "cut off from other nodes"  )   
                continue 
            else:
                sourcenodes.append(n0)                
            pathsn = [pathd[k] for k in startnodes]
            lengthsn = [lengthd[k] for k in startnodes]
            if alt_G:
                lengthsn = compute_path_lengths(alt_G, pathsn, weight=weight)
            # set vals as medial path length 
            val = "Median time to nodes: " + str(round(np.median(lengthsn),2)) + 'H'
            sourcenode_vals.append(val)
            #print ("pathsn", pathsn
            #print ("lengthn", lengthsn
            paths.extend(pathsn)
            lengths.extend(lengthsn)
    
    if verbose:
        print ("compute_paths() - lengths:", lengths)
            
    print ("Time to compute paths:", time.time() - t1, "seconds")
    # Time to compute paths: 1.38523888588 seconds

    # create the paths   
    # assume np arrays, not list (except for edgelist)        
    #ex0, ey0, ex1, ey1, emx, emy, elen, edgelist = get_path_pos(G2, paths, \
    #        lengths, skipset=set())   
    ex0, ey0, ex1, ey1, elat0, elon0, elat1, elon1, emx, emy, elen, edgelist =\
            get_path_pos(G2, paths, lengths, skipset=set())               

    # for constant width!
    if const_width:
        ewidth = line_width*np.ones(len(ex0))#len(ex0)*[line_width]

    # create columndatasource
    esource = bk.ColumnDataSource(
        data=dict(
            ex0=ex0, ey0=ey0,
            ex1=ex1, ey1=ey1,
            elat0=elat0, elon0=elon0,
            elat1=elat1, elon1=elon1,
            emx=emx, emy=emy,
            ewidth=ewidth,
            ecolor=np.asarray(len(ex0)*[line_color]),
            ealpha=line_alpha*np.ones(len(ex0))#len(ex0)*[line_alpha]
        )
    )    
    
    # add coords for MultiLine if desired
    if ecurve_dic is not None:
        elx0, ely0, ellat0, ellon0 = get_ecurves(edgelist, ecurve_dic)  
        # add to esource
        esource.data['elx0'] = elx0
        esource.data['ely0'] = ely0
        esource.data['ellon0'] = ellon0
        esource.data['ellat0'] = ellat0
       
    return esource, sourcenodes, paths, lengths, sourcenode_vals, missingnodes
    
    
###############################################################################
def reweight_paths(G, bestpaths, weight='Travel Time (h)', weight_mult=3.):
    '''    Multiply bestpath edge weights in G by weight_mult
    The returned graph Gc will have altered edge weights so that secondary
    paths can be computed'''

    '''Compute secondary paths in G
    Multiply edge weights by weight_mult, and recompute best paths
    return best paths with these altered weights
    path lengths will be incorrect'''
    
    #print ("paths", bestpaths
    # copy graph (and remove skipnodes?)
    Gc = G.copy()
    #if len(skipnodes) != 0:
    #    Gc.remove_nodes_from(skipnodes)  
    # add weights to paths already seen
    # flatten paths
    edgeset = set()
    for path in bestpaths:
        for i in range(len(path)-2):
            s,t = path[i], path[i+1]
            # add edge and reversed edge to edgeset
            edgeset.add((s,t))
            edgeset.add((t,s))
    
    #print ("edgeset", edgeset
    #print ("len edgeset", len(edgeset)
    # increase weights to Gc
    seen_edges = set()
    for edge in edgeset:
        (s, t) = edge
        # since G is not directed, skip reversed edges
        if (t, s) in seen_edges:
            continue
        else:
            try :
                w0 = Gc.edges[s,t][weight]
                Gc.edges[s,t][weight] = w0 * weight_mult
            except:
                w0 = Gc.edges[s,t,0][weight]
                Gc.edges[s,t,0][weight] = w0 * weight_mult                
            #w0 = Gc.edges[s][t][weight]
            #Gc.edges[s][t][weight] = w0 * weight_mult
            seen_edges.add(edge)
            
    return Gc


###############################################################################
def compute_path_lengths(G, paths, weight='Travel Time (h)'):
    '''compute length of known paths'''
    lengths_out = np.zeros(len(paths))#len(paths) * [0.0]
    for i,path in enumerate(paths):
        edges = [(path[j], path[j+1]) for j in range(len(path)-2)]
        try:
            length = np.sum([G.edges[e[0], e[1]][weight] for e in edges])
        except:
            length = np.sum([G.edges[e[0], e[1], 0][weight] for e in edges])

        #ƒlength = np.sum([G.edges[e[0]][e[1]][weight] for e in edges])
        lengths_out[i] = length
    return lengths_out


###############################################################################
def get_route_sources(G, end_nodes, auglist, g_node_props_dic,
                      ecurve_dic=None, target=None,
                      skipnodes=[], goodroutes=True,
                      compute_secondary_routes=False,
                      compute_subgraph_centrality=True,
                      binsize=0.5,
                      edge_weight='Travel Time (h)',  # 'Travel Time (h)'
                      crit_perc=None,
                      target_size_mult=2):
    '''Compute columndatasources given route parameters'''

    t0 = time.time()
    color_dic, alpha_dic = define_colors_alphas()
    global_dic = global_vars()

    # expand auglist
    [ngood, nbad, ngood_aug, nbad_aug, cds_good_aug, cds_bad_aug] = auglist

    # set colors (set line_color within compute_paths)
    if goodroutes == True:
        #line_color = color_dic['compute_path_color_good']
        sourcenode_color = color_dic['goodnode_color']
    elif goodroutes == False:
        #line_color = color_dic['compute_path_color_bad']
        sourcenode_color = color_dic['badnode_color']
    else:
        sourcenode_color = color_dic['source_color']

    t00 = time.time()
    # define target source
    if target:
        source_target = bokeh_utils.set_nodes_source(G, [target], 
                    size=target_size_mult * global_dic['maxS'], 
                    color=color_dic['target_color'], 
                    fill_alpha=alpha_dic['target'], shape='square',       
                    label=['Target'], count=[], val=['Target'],
                    name=[target])          
    else:
        source_target = bokeh_utils.set_nodes_source_empty()
    print ("Time to compute target cds:", time.time()-t00, "seconds")
            
    # compute paths
    t01 =  time.time()
    print ("get_route_sources: len(G.nodes):", len(G.nodes))
    esource, sourcenodes, paths, lengths, sourcenode_vals, missingnodes =  \
                        compute_paths(G, end_nodes, ecurve_dic=ecurve_dic, 
                            target=target, \
                            skipnodes=skipnodes, \
                            weight=edge_weight, #global_dic['edge_weight'], 
                            alt_G=None, goodroutes=goodroutes) 
    print ("Time to compute paths cds:", time.time()-t01, "seconds")
    print ("   len(lengths):", len(lengths))
    print ("   len(sourcenodes):", len(sourcenodes))
    print ("   len(unique sourcenodes):", len(np.unique(sourcenodes)))
    print ("   sourcenodes:", sourcenodes)
    print ("   end_nodes:", end_nodes)

    #######
    # optional, compute secondary routes
    if not compute_secondary_routes:
        esourcerw = bk.ColumnDataSource(data=dict())
    else:
        t02 = time.time()
        Grw = reweight_paths(G, paths, 
                            weight=edge_weight, #global_dic['edge_weight'], 
                            weight_mult=global_dic['weight_mult'] )
    
        # now compute new paths
        # path lengths will be wrong because the weight is increased, so we
        # need to recompute path lengths after the path is returned
        # this is accompltished in compute_paths, using alt_G=G
        esourcerw, sourcenodesrw, pathsrw, lengthsrw, sourcenode_valsrw, \
            missingnodesrw =  \
                        compute_paths(Grw, end_nodes, ecurve_dic=ecurve_dic,
                                    target=target, \
                                    skipnodes=skipnodes, \
                                    weight=edge_weight, #global_dic['edge_weight'], 
                                    alt_G=G, goodroutes=goodroutes) 
        print ("Time to compute path_sec cds:", time.time()-t02, "seconds")

        # !! might want to update sourcenode_vals with new sourcenode_valsrw 
        #    for example, add another hover column for use in plot_nodes() !!
        # !! Could also append pathsrw to paths for compute_crit_nodes() and
        #   subgraph !!
    #######
                            
    t03 = time.time()
    # compute critical nodes, sorting by counts of nodes in paths
    if crit_perc is not None:
        crit_p = crit_perc
    else:
        crit_p = global_dic['crit_perc']
    # flatten array
    node_flat = [item for sublist in paths for item in sublist]
    crit_nodes, crit_counts = compute_crit_nodes(G, node_flat, \
            plot_perc=crit_p, sortlist=None) 
    crit_size, Ac, Bc = lin_scale(crit_counts, global_dic['minS'],
                                   global_dic['maxS'])
    source_crit = bokeh_utils.set_nodes_source(G, crit_nodes, size=crit_size, 
                                   color=color_dic['crit_node_color'], 
                                   fill_alpha=alpha_dic['crit_node'],
                                   #shape='square', 
                                   label=crit_counts, count=crit_counts, 
                                   val=[], name=crit_nodes)            
    print ("Time to compute crit cds:", time.time()-t03, "seconds")


    # get missing nodes source
    t04 = time.time()
    missingnodes_count = [1 for n in missingnodes]
    #missingnodes_count = [len(g_node_props_dic[n]['index']) for n in missingnodes]
    #missingnodes_count = [g_node_props_dic[n]['count'] for n in missingnodes]
    Nm = len(missingnodes)
    source_missing = bokeh_utils.set_nodes_source(G, missingnodes, size=global_dic['maxS'], 
                                    color=color_dic['missing_node_color'], 
                                    fill_alpha=alpha_dic['missing_node'],
                                    #shape='square', 
                                    label=Nm*['Obstructed!'], 
                                    count=missingnodes_count, 
                                    val=Nm*['Obstructed'],
                                    name=missingnodes)
    print ("Time to compute missing cds:", time.time()-t04, "seconds")
    

    # get sourcenodes source
    t05 = time.time()
    counts = []
    for itmp,n in enumerate(sourcenodes):
        print ("   ", itmp, " sourcenodeL", n)
        if n in list(g_node_props_dic.keys()):
            counts.append(len(g_node_props_dic[n]['index']))
        else:
            counts.append(0)
    #counts = [len(g_node_props_dic[n]['index']) for n in sourcenodes]
    #counts = [g_node_props_dic[n]['count'] for n in sourcenodes]
    print ("counts:", counts)
    sizes, A, B = log_scale(counts, global_dic['minS'], global_dic['maxS'])   
    print ("sizes:", sizes)
    source_sourcenodes = bokeh_utils.set_nodes_source(G, sourcenodes, size=sizes, 
                                    color=sourcenode_color, 
                                    fill_alpha=alpha_dic['end_node'],
                                    #shape='circle', 
                                    label=counts, 
                                    count=counts, 
                                    val=sourcenode_vals,
                                    name=sourcenodes)
    print ("Time to compute sourcenode cds:", time.time()-t05, "seconds")

    # histograms
    # Take paths, lengths, and counts and determine rate of arrival
    if target:
        x, y = lengths, counts        
    else:
        # if we want the histogram of all paths, use x = lengths
        #x, y = lengths, []
        
        # more useful, take the median length of paths to each source
        # take sourcenode_vals, and parse out string, turn to float
        svals = [float(a[22:-1]) for a in sourcenode_vals]
        #print ("get_route_source() - sourcenode_vals:", sourcenode_vals)
        #print ("get_route_source() - svals:", svals)
        x, y = svals, []
    # get bins, counts
    t06 = time.time()
    print ("get route sources(): x,y:", x, y)
    bins, bin_str, digitized, binc_arr, cumc_arr = make_histo_arrs(x, y=y, \
                                      binsize=binsize)
    print ("get route sources(): binsize:", binsize)
    print ("get route sources(): bins:", bins)
    print ("get route sources(): digitized:", digitized)
    print ("get route sources(): binc_arr:", binc_arr)
    print ("get route sources(): cumc_arr:", cumc_arr)

    # create sources 
    #source_histo_cum = get_histo_source(bin_str, cumc_arr, 
    source_histo_cum = bokeh_utils.get_histo_source(bins, cumc_arr, 
                                      color=color_dic['histo_cum_color'], 
                                      alpha=alpha_dic['histo_cum'], width=0.99, 
                                      legend='Cumulative', x_str_arr=bin_str)
    #source_histo_bin = get_histo_source(bin_str, binc_arr, 
    source_histo_bin = bokeh_utils.get_histo_source(bins, binc_arr, 
                                      color=color_dic['histo_bin_color'], 
                                      alpha=alpha_dic['histo_bin'], width=0.99, 
                                      legend='Binned', x_str_arr=bin_str)
    print ("get_route_sources(): source_histo_bin.data:", source_histo_bin.data)
    print ("Time to compute histo cds(s):", time.time()-t06, "seconds")

        

    # module B: Subgraph centrality values
    ######################################        
    # add another layer with Gsub centrality values
    if not compute_subgraph_centrality:
        source_subgraph_centrality = bokeh_utils.set_nodes_source_empty()
  
        
    else:
        # compute:
        # set sizes, labels as closeness
        t07 = time.time()
        Gsub = create_subgraph(G, paths,  weight=edge_weight) 
        nsub = np.asarray(Gsub.nodes())
        ########
        # compute betweenness, closeness centrality
        # https://networkx.github.io/documentation/stable/reference/algorithms/generated/networkx.algorithms.centrality.betweenness_centrality.html#networkx.algorithms.centrality.betweenness_centrality
        #  Betweenness centrality of a node v
        #  is the sum of the fraction of all-pairs shortest paths that pass through v
        # https://networkx.github.io/documentation/stable/reference/algorithms/generated/networkx.algorithms.centrality.closeness_centrality.html#networkx.algorithms.centrality.closeness_centrality
        #  Closeness centrality [1] of a node u is the reciprocal of the 
        #  average shortest path distance to u over all n-1 reachable nodes.
        
        ## updated method
        #cntr_between_dict = nx.betweenness_centrality(Gsub, weight=edge_weight)
        #cntr_close_dict = nx.closeness_centrality(Gsub, distance=edge_weight)
        #subbetwn = [round(cntr_between_dict[n],2) for n in nsub]
        #subclose = [round(cntr_close_dict[n],2) for n in nsub]
        #print ("subbetwn:", subbetwn )
        #print ("subclose:", subclose )
        #nsub1, closes = compute_crit_nodes(G, nsub, plot_perc=10, \
        #        sortlist=subclose)          
        #subval = ['Betweenness: '+ \
        #    str(round(cntr_between_dict[n],2)) + '/' + 
        #    str(round(max(subbetwn),2)) for n in nsub1]
        #subcount = ['Closeness: '+ \
        #    str(round(cntr_close_dict[n],2)) + '/' + 
        #    str(round(max(subclose),2)) for n in nsub1]
        #print ("subval:", subval)
        #print ("subclose:", subclose)

        ########
        # old method
        subbetw0 = [round(Gsub.nodes[n]['Betweenness Centrality'],2) for n in nsub]
        subclose0 = [round(Gsub.nodes[n]['Closeness Centrality'],2) for n in nsub]
        #print ("Min, Max Betweenness Centrality:", min(subbetw0), max(subbetw0))
        #print ("Min, Max Closeness Centrality:", min(subclose0), max(subclose0))
        # keep only critial points
        #nsub1, betweens = compute_crit_nodes(G, nsub, plot_perc=10, \
        #        sortlist=subbetw0)
        nsub1, closes = compute_crit_nodes(G, nsub, plot_perc=10, \
                sortlist=subclose0)                
        subval = ['Betweenness: '+ \
            str(round(Gsub.nodes[n]['Betweenness Centrality'],2)) + '/' + 
            str(round(max(subbetw0),2)) for n in nsub1]
        subcount = ['Closeness: '+ \
            str(round(Gsub.nodes[n]['Closeness Centrality'],2)) + '/' + 
            str(round(max(subclose0),2)) for n in nsub1]
        #sublabel = [
        #    str(round(Gsub.nodes[n]['Closeness Centrality'],2)) + '/' + 
        #    str(round(max(subbetw0),2)) for n in nsub1]  
        ########
        
        subsizes, A, B = lin_scale(closes, global_dic['minS'], 
                                        global_dic['maxS'])  
        source_subgraph_centrality = bokeh_utils.set_nodes_source(G, nsub1, size=subsizes, 
                                    color=color_dic['node_centrality_color'], 
                                    fill_alpha=alpha_dic['end_node'],
                                    #shape='diamond', 
                                    label=[],#sublabel, 
                                    count=subcount, 
                                    val=subval,
                                    name=nsub1)
        print ("Time to compute subgraph cds:", time.time()-t07, "seconds")
    ######################################

    
    #print ("esource.data", esource.data
    #print ("source_histo_cum.data", source_histo_cum.data
    #print ("source_crit.data", source_crit.data

    print ("Time to compute route sources:", time.time()-t0, "seconds")

    return esource, esourcerw, source_target, source_sourcenodes, source_crit,\
                source_missing, source_histo_cum, source_histo_bin, \
                source_subgraph_centrality     


###############################################################################                            
def path_counts(G, end_nodes, auglist, skipnodes=[], goodroutes=True,
                target=None, compute_secondary_routes=True,
                edge_weight='Travel Time (h)'):
    '''Return counts of nodes traversed on paths
    Similar to compute_crit_nodes, though this function also computes paths'''
    
    global_dic = global_vars()
    #half_sec_counts = True      # switch to half secondary route counts

    # compute paths
    t01 =  time.time()
    esource, sourcenodes, paths, lengths, sourcenode_vals, missingnodes =  \
                        compute_paths(G, end_nodes, target=target, \
                            skipnodes=skipnodes, \
                            weight=edge_weight, #global_dic['edge_weight'], 
                            alt_G=None, goodroutes=goodroutes) 
    print ("Time to compute paths cds:", time.time()-t01, "seconds")

    #######
    # optional, compute secondary routes
    if compute_secondary_routes:
        t02 = time.time()
        Grw = reweight_paths(G, paths, 
                             weight=edge_weight, #global_dic['edge_weight'], 
                             weight_mult=global_dic['weight_mult'] )  
        # now compute new paths
        esourcerw, sourcenodesrw, pathsrw, lengthsrw, sourcenode_valsrw, \
            missingnodesrw =  \
                        compute_paths(Grw, end_nodes, target=target, \
                                    skipnodes=skipnodes, \
                                    weight=edge_weight, #global_dic['edge_weight'], 
                                    alt_G=G, goodroutes=goodroutes) 
        print ("Time to compute path_sec cds:", time.time()-t02, "seconds")
    #######
                            
    #t03 = time.time()
    # compute critical nodes, sorting by counts of nodes in paths
    node_flat = [item for sublist in paths for item in sublist]
    if compute_secondary_routes:
        node_flatrw = [item for sublist in pathsrw for item in sublist]
        # combine two lists
        node_flat = node_flat + node_flatrw
    # compute all nodes and counts
    crit_perc = 100
    crit_nodes, crit_counts = compute_crit_nodes(G, node_flat, \
            plot_perc=crit_perc, sortlist=None) 
            
    return crit_nodes, crit_counts
    

###############################################################################
### Graph props                 
###############################################################################
def create_subgraph(G, paths, weight='Travel Time (h)'):
    '''Compute properties of subgraph by removing paths from G'''
    
    # Create list of all nodes of interest 
    path_flat = set([item for sublist in paths for item in sublist])
    # make sure that sourcenodes are in path_flat
    #path_flat.update(sourcenodes)

    # find the nodes not in path_flat
    rem_nodes = set(G.nodes()) - path_flat
    #print ("len(rem_nodes)", len(rem_nodes)
    
    # copy graph and remove desired nodes 
    G3 = G.copy()
    G3.remove_nodes_from(list(rem_nodes))
    print ("Number of nodes in subgraph:", len(G3.nodes()))
    # compute ALL properties 
    G3 = graph_init._node_props(G3, weight=weight, compute_all=True) 
    return G3
       
       
###############################################################################
def compute_crit_nodes(G, nlist, plot_perc=20, sortlist=None):
    '''From a list of nodes, compute the critical nodes by count
    if sortlist=None, sort and group by unique counts of items in nlist
    else, sort nlist by sortlist
    keep only the top plot_perc percentage of points
    return sorted nlist and countlist'''

    if sortlist is None:    
        # determine critical nodes in paths
        # flatten array
        #node_flat = [item for sublist in paths for item in sublist]
        #freq = scipy.stats.itemfreq(node_flat)
        unique0, counts0 = np.unique(nlist, return_counts=True)
    else:
        unique0, counts0 = np.asarray(nlist), np.asarray(sortlist)
        
    # sort descending by max counts
    f0 = np.argsort(counts0)[::-1]
    crit_nodes = unique0[f0]
    crit_counts = counts0[f0]
    #####
    # optional: remove nodes with degree of <=2
    rem_idx = []
    for i,n in enumerate(crit_nodes):
        deg = G.degree[n]
        #print ( "compute_crit_nodes():' deg", deg)
        #deg = G.nodes[n]['deg']
        if deg <= 2:
            rem_idx.append(i)
    crit_nodes = np.delete(crit_nodes, rem_idx)
    crit_counts = np.delete(crit_counts, rem_idx)
    #####
    # keep all nodes above a percentile threshold (already filtered above...)
    #thresh = scipy.percentile(crit_counts, 80) 
    #f1 = np.where(crit_counts >= thresh)
    #crit_nodes = crit_nodes[f1]
    #crit_counts = crit_counts[f1]
    ####
    numN = int(len(unique0) * (plot_perc / 100.))
    #print ("Top", int(plot_perc), "%, critical nodes, counts:", \
    #        zip(crit_nodes[:numN], crit_counts[:numN])
    print ("Top 10, critical nodes, counts:", \
            (crit_nodes[:10], crit_counts[:10]))
            
    return crit_nodes[:numN], crit_counts[:numN]


###############################################################################
def make_histo_arrs(x, y=[], binsize=0.5, verbose=False):
    '''Create binned and cumulative histograms
    If y=None, just use number of counts per xbin'''

    if verbose:
        print ("make_histo_arrs() - lenx:", len(x))
        print ("make_histo_arrs() - leny:", len(y))
        print ("make_histo_arrs() - x:", x)
        print ("make_histo_arrs() - y:", y)
        
    # first, bin by path length
    if len(x) == 0:
        return [], [], [], [], []
    bins = np.arange(binsize, max(x)+binsize, binsize)
    # reformate bins_out to reflect the midpoint of each bin
    bins_out = bins - 0.5*binsize
    # bin_str must be list since bokeh range setting cannot handle np.array!
    bin_str = np.asarray([str(round(b-binsize,2)) + '-' + str(round(b,2)) + 'H' for b in bins])
    #print ("bin_str", bin_str
    # default is right = False...
    digitized = np.digitize(x, bins, right=True)
    #digitized = np.digitize(x, bins)
    if len(y) == 0:
        # simply count the number of times each bin occors
        binc_arr = np.bincount(digitized)
    else:        
        # if we have a target, find number of 'y' per time period
        binc_arr = np.zeros(len(bins))
        npy = np.asarray(y)
        # loop through bins    
        for i,b in enumerate(bins):
            idxs = np.where(digitized==i)[0]
            if len(idxs) != 0:
                binc_arr[i] = np.sum(npy[idxs])
    # compute cumulative array
    cumc_arr = np.cumsum(binc_arr)
    
    return bins_out, bin_str, digitized, binc_arr, cumc_arr


###############################################################################
def compute_hull(G, nlist, concave=True, coords='wmp', concave_alpha=200):
    '''Compute hull of points, by default compute concave, not convex hull'''
    t0 = time.time()
    if len(nlist) == 0:
        return [],[],[]
    #nlist = np.unique(np.asarray(startnodes)[idxs])
    if coords == 'latlon':
        points = np.asarray([[G.nodes[n]['lon'],G.nodes[n]['lat']] for n in nlist])
    elif coords == 'wmp':
        points = np.asarray([[G.nodes[n]['x'],G.nodes[n]['y']] for n in nlist])
    else:
        print ("Unknown coords in compute_hull")
        return
        
    if concave:
       hull, hullx, hully, hull_indices = \
                                  concave_hull.alpha_shape(points, 
                                                           alpha=concave_alpha)
       hulln = [nlist[idx] for idx in hull_indices]
       #print ("hulln", hulln
       #print ("hullx", hullx
       #print ("hully", hully
     
    else:
        hull = scipy.spatial.ConvexHull(points)
        hulln = nlist[hull.vertices]
        hullx, hully = points[hull.vertices,0], points[hull.vertices,1]
    
    print ("Total time to compute hull:", time.time() - t0, "seconds")
    return np.asarray(hulln), np.asarray(hullx), np.asarray(hully)
  
   
###############################################################################
def point_inside_hull(hullx, hully, testx, testy):
#http://stackoverflow.com/questions/16750618/whats-an-efficient-way-to-find-if-a-point-lies-in-the-convex-hull-of-a-point-cl
    '''Test if testx, testy is inside the hull.  Do this by adding the point to
    the hull vertices, and computing a new_hull.  If the added point is one
    of the vertices of the new_hull, the point is outside the original hull'''
 
    # Create array of [[x0,y0], [x1,y1], ... [xtest, ytest]]
    new_points = np.vstack( (np.append(hullx, [testx]), np.append(hully, [testy]) )).T
    ## creat 2-d array of hull points
    #points = np.vstack((hullx, hully)).T
    ## add new point to points
    #new_points = np.append(points, [[testx, testy]], axis=0)   
  
    # find new hull 
    new_hull = scipy.spatial.ConvexHull(new_points)
    
    # Vertices = Indices of points forming the vertices of the convex hull
    # if the index of the added point (len(points)) is in new_hull.vertices,
    # the hull changed and the point is external
    if len(hullx) in set(new_hull.vertices):
        return False
    else:
        return True


###############################################################################
def gdelt_inside_hull_v0(source_bbox, hullx, hully, coords='latlon'):
    '''See which gdelt points are inside the hull
    !! Could also just see if source_bbox.data['nearest'] is in the points used 
        to construct the hull !!'''
    
    if coords == 'latlon':
        gxarr = source_bbox.data['lon']
        gyarr = source_bbox.data['lat']
    elif coords == 'wmp':
        gxarr = source_bbox.data['x']
        gyarr = source_bbox.data['y']    
    else:
        print ("Unknown coords in gdelt_insude_hull_v0()")
        return
    
    inside_indices = []
    for i,(gx, gy) in enumerate(zip(gxarr, gyarr)):
        #print gx, gy
        if point_inside_hull(hullx, hully, gx, gy):
            inside_indices.append(i)
            
    return inside_indices
 
 
###############################################################################
def bbox_inside_hull_v1(nlist, source_bbox, ignore=[]):
    '''See which bbox points have source_bbox.data['nearest'] in nlist'''
    
    gnearest = source_bbox.data['nearest_osm']
    gcount = source_bbox.data['count']
    gstatus = source_bbox.data['status']
    nset = set(nlist)
        
    good_indices, good_count = [], []
    bad_indices, bad_count = [], []
    for i,(gnear, gc, gstat) in enumerate(zip(gnearest, gcount, gstatus)):
        # skip if nearest osm node not in nlist
        if gnear not in nset:
            continue
        # else, add values to lists
        if gstat == 'good':
            ilist = good_indices
            clist = good_count
        else:
            ilist = bad_indices
            clist = bad_count
        ilist.append(i)
        clist.append(gc)

    return good_indices, good_count, bad_indices, bad_count     


###############################################################################
def compute_risk(G_, source_nodes, g_node_props_dic, target_nodes=[],
                 skipnodes=[], ignore_nodes=[], weight='Travel Time (h)',
                 coords='wmp', kdtree=None, kd_idx_dic=None, r_m=50., 
                 size_up=3,  # increases size by this number
                 verbose=False):
    '''
    Very similar to compute_paths()
    compute threat to target_nodes from source_nodes
    Compute path length from target_nodes to source_nodes
        (backward to make sure we don't delete target_node and its aug_points
        from graph)
    if target_nodes = [], use all nodes
    Then compute risk
        For now, defined as # forces / time
    skipnodes denote nodes to skip (make sure we don't remove target and
        augmented nodes for target!)
    ignore_nodes are nodes to ignore
    '''

    global_dic = global_vars()
    color_dic, alpha_dic = define_colors_alphas()

    # create list of all nodes not in input nodes
    if len(target_nodes) == 0:
        target_nodes = list(G_.nodes())
    targetnodes1 = list(set(target_nodes) - set(source_nodes) - set(ignore_nodes))
    if verbose:
        print ("compute_risk(): targetnodes1:", targetnodes1)
    # init risk to zero for all nodes
    risk_dic = dict.fromkeys(targetnodes1, 0.01)   

    t1 = time.time()
    #paths, lengths, sourcenode_vals, sourcenodes, missingnodes = [],[], [], [], []
    missingnodes = []
    for k,n0 in enumerate(targetnodes1):
        if verbose:
            print("compute_risk() n:", n0)    
            print ("compute_risk() len(skipnodes):",  len(skipnodes))
            
        if len(skipnodes) == 0:
            G2 = G_
        else:
            #########
            # remove skipnodes from graph, but make sure target and its augmented
            # nodes remain
            if coords == 'wmp':
                names, idxs, kms = query_kd_ball(kdtree, kd_idx_dic, 
                                                 G_.nodes[n0]['x'],
                                                 G_.nodes[n0]['y'], 
                                                 r_m)
                if verbose:
                    print("compute_risk() names:", names)
            else:
                names, idxs, kms = utils.query_kd_ball_latlon(kdtree, 
                                                        kd_idx_dic, 
                                                        G_.nodes[n0]['lat'],
                                                        G_.nodes[n0]['lon'], 
                                                        r_m / 1000.)
            names.append(n0)                                 
            # remove names from skipnodes
            skipnodes0 = list( set(skipnodes)  - set(names) )
            #print ("n0", n0
            #print ("len(names)", len(names)
            #print ("len G.nodes", len(G.nodes())
            #print ("len(skipnodes)", len(skipnodes)
            #print ("len(skipnodes0)", len(skipnodes0)
            #print ("n0 in skipnodes0?", n0 in skipnodes0
            G2 = G_.copy()
            G2.remove_nodes_from(skipnodes0)                                 
            #########                       
                                         
        # compute paths                             
        # get all paths from source
        lengthd, pathd = nx.single_source_dijkstra(G2, source=n0, weight=weight) 
        # try to remove n0 from dic
        try:
            del lengthd[n0]
            del pathd[n0]
        except:
            pass
        n1_list = lengthd.keys()
        # check if node is cut off from all other nodes, if so
        # the path dictionary will only contain the source node
        if n1_list == [n0] or len(n1_list) == 0:
            missingnodes.append(n0)
            print ("Node", n0, "cut off from other nodes"   )  
            continue

        ## test
        #for n in n1_list:
        #    if lengthd[n] == 0:
        #        print ("n0, n1, length", n0, n, lengthd[n]

        #print ("len(n1_list)", len(n1_list)
        
        #t2 = time.time()
        # keep only source_nodes in n1_list
        n1_list = list(set(n1_list).intersection(set(source_nodes)))
        if verbose:
            print("n1_list", n1_list)
        # count up risk from each source node
        riskd = 0.01
        for n1 in n1_list:
            count = len(g_node_props_dic[n1]['index'])
            #count = g_node_props_dic[n1]['count']
            #print ("n1", n1, "count", count
            riskd += count / lengthd[n1]
        #print ("riskd", riskd
        risk_dic[n0] = riskd
        #print ("Time to update dic", time.time() - t2, "seconds"
            
    print ("Time to compute paths:", time.time() - t1, "seconds")

    names = np.asarray(list(risk_dic.keys()))
    vals = np.asarray([risk_dic[k] for k in names])
    color = color_dic['risk_color']
    alpha = alpha_dic['risk']
    # set sizes
    risk_size, Ac, Bc = lin_scale(vals, global_dic['minS']+size_up,
                                   global_dic['maxS']+size_up)
    vals_print = ['Risk: ' + str(round(v,2)) for v in vals]
    # create cds
    source_risk = bokeh_utils.set_nodes_source(G_, names, size=risk_size, color=color,
                                   fill_alpha=alpha, label=[],
                                    val=vals_print, name=names)
    # # inspect cds                                
    #for key in source_risk.data.keys():
    #    print ("source_risk.data key", key
    #    print ("   type(data)", type(source_risk.data[key])
    #    print ("   len(data)", len(source_risk.data[key])
 
    return source_risk


###############################################################################
def compute_hull_sources(G, sourcenode, source_bbox, skipnodes=set([]), 
                        binsize=0.5, concave=True, concave_alpha=2, 
                        coords='wmp', edge_weight='Travel Time (h)'):
    '''Compute spread of forces from initial node.
    Very similar to compute_travel_time, above
    binsize is in units of hours
    concave_alpha is the alpha parameter for concave hulls
    Unused???'''

    from bokeh.palettes import Spectral6 as pal0
    # stack palette
    palette = pal0 + pal0 + pal0 + pal0
    #t0 = time.time()
    global_dic = global_vars()
    color_dic, alpha_dic = define_colors_alphas()
    #weight = global_dic['edge_weight']
    target_size = 8
    #target_color = color_dic['source_color']

    size_mult = 1
    source_source = bokeh_utils.set_nodes_source(G, [sourcenode], 
                                size=size_mult*global_dic['maxS'], 
                                color=color_dic['target_color'],
                                fill_alpha=alpha_dic['target'], 
                                shape='diamond', label=['Source'], count=[], 
                                val=[], name=['Source'])
    
    # define colors
    #edge_seen_color = color_dic['spread_seen_color']
    #edge_new_color = color_dic['spread_new_color']
    #target_color = edge_new_color
    
    # copy graph
    G2 = G.copy()
    # remove desired nodes
    if len(skipnodes) != 0:
        # ensure targets or goodnote not in skipnodes
        skipnodes = [i for i in skipnodes if i!=sourcenode]
        G2.remove_nodes_from(skipnodes)

    # computing paths from sourcenode, not into sourcenode, so if graph is directed
    # this will cause problems
    t1 = time.time()
    n1 = sourcenode
    lengthd, pathd = nx.single_source_dijkstra(G2, source=n1, 
                                               weight=edge_weight) 
    #print ("tmp0")
    startnodes = np.asarray(list(pathd.keys()))
    #paths = [pathd[k] for k in startnodes]
    lengths = [lengthd[k] for k in startnodes]  
    print ("Time to compute paths:", time.time() - t1, "seconds")
    # # loop: Time to compute paths: 103.538288116 seconds
    # dijkstra: Time to compute paths: 0.0634140968323 seconds

    # Now take paths, lengths, and counts and determine rate of arrival
    # first, bin by time
    # remember bins are in center of time segment
    bins, bin_str, digitized, tmp0, tmp1 = \
            make_histo_arrs(lengths, y=[], binsize=binsize) 
    bin_diff = bins[1] - bins[0]
    #print ("bin_str", bin_str             
    globc = 0
    nlist_hull = []
    nlist_name = []
    ncolor = []
    nlist_tot = []
    xlist_hull = []
    ylist_hull = []
    # initialize histogram counts 
    good_bin, good_cum = np.zeros(len(bins)), np.zeros(len(bins))
    bad_bin, bad_cum = np.zeros(len(bins)), np.zeros(len(bins))
    tot_bin, tot_cum = np.zeros(len(bins)), np.zeros(len(bins))
    #sources_hull_nodes, sources_hull_patch = np.zeros(len(bins)), np.zeros(len(bins))
    # loop through bins
    for i,b in enumerate(bins):
        
        print ("Computing bin", i+1, "of", len(bins)  )    
        # get indices into lenths and paths
        idxs = np.where(digitized==i)[0]        
        # set total counts of nodes reached
        totc = 0    
        
        # if idxs is empty, continue
        if len(idxs) == 0: 
            if i > 0:
                bad_cum[i] = bad_cum[i-1]
                good_cum[i] = good_cum[i-1]
                tot_cum = tot_cum[i-1]
            continue
                
        # plot hulls          
        nlist_new = np.unique(startnodes[idxs])
        # combine nlist_tot with last nlist to make sure hull encompasses
        # all points this means that the final hull will have all the 
        # points, which is much slower but allows for a more aggressive 
        # alpha parameter.  Plotting with nlist_hull is much faster
        # though agressive (high) alpha parameter can split the hull into
        # multiple polygons
        if i > 0:
            #nlist_old = nlist_hull[i-1]
            nlist_tot = np.unique(np.concatenate((nlist_new, nlist_tot)))
        else:
            #nlist_old = []
            nlist_tot = nlist_new
        # plot new hull
        # remember bins are now in center of the time segment
        #hull_name = 'Hull ' + str(round(b-binsize,2)) + '-' + str(round(b,2)) + 'H'
        hull_name = 'Hull ' + str(round(b-0.5*bin_diff,2)) + '-' \
            + str(round(b+0.5*bin_diff,2)) + 'H'     
        
        print ("hull_name", hull_name)
        # much faster to use nlist_plot
        #hulln, hullx, hully = compute_hull(G, nlist_plot, concave=concave, alpha=concave_alpha)
        # more accurate to use nlist_tot
        hulln, hullx, hully = compute_hull(G, nlist_tot, concave=concave, 
                                           concave_alpha=concave_alpha,
                                           coords=coords)
        nlist_hull.append(hulln)  
        ncolor.extend(len(hulln) * [palette[i]])
        nlist_name.append(len(hulln)*[hull_name])
        xlist_hull.append(hullx)  
        ylist_hull.append(hully)  
        
        # get bbox points inside hull                       
        good_indices, good_count, bad_indices, bad_count = \
            bbox_inside_hull_v1(nlist_new, source_bbox, ignore=[]) 
        # extract subset of data
        # update counts
        bad_bin[i] = np.sum(source_bbox.data['count'][bad_indices])
        bad_cum[i] = np.sum(bad_bin)
        good_bin[i] = np.sum(source_bbox.data['count'][good_indices])
        good_cum[i] = np.sum(good_bin) 
        tot_bin[i] = bad_bin[i] + good_bin[i]
        tot_cum[i] = bad_cum[i] + good_cum[i]
        #print 'Adversary Forces Encountered', bad_bin[i]
        #print 'Allied Forces Encountered', good_bin[i]
        # update total count                
        globc += totc        
 
    # return histo sources and node, patch sources
    #print ("nlist_hull", nlist_hull
    nflat = [item for sublist in nlist_hull for item in sublist]
    nflat_name = [item for sublist in nlist_name for item in sublist]
    #print ("nflat_name", nflat_name

    source_hull_node = bokeh_utils.set_nodes_source(G, nflat, size=target_size, 
                color=ncolor,#target_color, 
                fill_alpha=alpha_dic['end_node'], shape='circle', label=[], 
                count=[], val=nflat_name, name=nflat_name)
    #source_hull_patch = bk.ColumnDataSource(dict(xs=xlist_hull, ys=ylist_hull, 
    #                                alpha=len(xlist_hull)*[alpha_dic['hull']], 
    #                                color=len(xlist_hull)*[target_color]))
    source_hull_patch = bk.ColumnDataSource(dict(xs=xlist_hull, ys=ylist_hull,
                            alpha=alpha_dic['hull']*np.ones(len(xlist_hull)),
                            #alpha=alpha_dic['hull']*np.ones(len(xlist_hull)),
                            color=np.asarray(palette[:len(xlist_hull)]) ))
                            #color=np.asarray(len(xlist_hull)*[target_color])))
                                    #alpha=len(xlist_hull)*[alpha_dic['hull']], 
                                    #color=len(xlist_hull)*[target_color])) 
    source_histo_bin = bokeh_utils.get_histo_source(bins, tot_bin, 
                                      color=color_dic['histo_bin_color'], 
                                      alpha=alpha_dic['histo_bin'], width=0.99, 
                                      legend='Binned', x_str_arr=bin_str)   
    source_histo_cum = bokeh_utils.get_histo_source(bins, tot_cum, 
                                      color=color_dic['histo_cum_color'], 
                                      alpha=alpha_dic['histo_cum'], width=0.99, 
                                      legend='Cumulative', x_str_arr=bin_str)

    return source_histo_bin, source_histo_cum, source_hull_node, \
            source_hull_patch, source_source
  
            
###############################################################################
### Contours
###############################################################################
# Compute gaussian process (same as Kriging) and plot contours
#http://stackoverflow.com/questions/18304722/python-find-contour-lines-from-matplotlib-pyplot-contour
#http://scikit-learn.org/stable/auto_examples/gaussian_process/plot_gp_probabilistic_classification_after_regression.html
# need to cut out contour pionts that aren't inside hull???

# https://pythonmatplotlibtips.blogspot.com/2018/11/2d-contour-plot-bokeh-without-colorbar.html
###############################################################################
def get_gauss_contours(source_bbox, auglist, G, remove_empties=True,
                       theta0=0.15, res=500, 
                       mpl_plot=False, use_aug=False, corr='linear',
                       nugget=0.05, smooth_sigma=3, smooth=True, 
                       coords='latlon'):
    '''Instantiate and fit Gaussian Process Model, and return contours
    x = [[x0, y0], [x1, y1], ...]
    z = values at x,y coords
    theta0: since thetaL and thetaU are also specified, theta0 is considered 
    as the starting point for the maximum likelihood estimation of the best 
    set of parameters
    res = number of points along each axis for interpolation
    see gauss_procc.py for tests
    larger nugget smoothes data
    corr=linear or absolute_exponential gives the best fit for gdelt data
    use_aug is extremely slow!
    if remove_empties, remove any shapes whose convex hull encloses no gdelt 
        data points
    http://scikit-learn.org/stable/modules/generated/sklearn.gaussian_process.GaussianProcess.html
    http://scikit-learn.org/stable/modules/gaussian_process.html
    '''

    color_dic, alpha_dic = define_colors_alphas()    
    # conf levels (center, 1sig_lo, 2sig_lo, 1sig_hi, 2sig_hi)
    conf_levels = [0.5, 0.1587, 0.025, 0.8413, 0.975]
    colors = ['black', color_dic['goodnode_aug_color'], 
              color_dic['goodnode_color'], 
              color_dic['badnode_aug_color'],
              color_dic['badnode_color']]
    linestyles = ['solid', 'solid', 'solid', 'solid', 'solid'] 
    
    t0 = time.time()
    
    if coords == 'latlon':
        xname = 'lon'
        yname = 'lat'
    elif coords == 'wmp':
        xname = 'x'
        yname = 'y'
    else:
        print ("oops")
        return
    
    # set data (using dataframe)
    #dftmp = dfgdelt.copy()
    #X = dftmp[['ActionGeo Long', 'ActionGeo Lat']].values
    #z = dftmp['Avg. Goldstein Scale'].values
    gxarr = source_bbox.data[xname]
    gyarr = source_bbox.data[yname]
    X = np.vstack((gxarr, gyarr)).T
    z = np.ones(len(source_bbox.data['val']))
    for i in range(len(source_bbox.data['val'])):
        z[i] = source_bbox.data['val'][i]
    #z = source_bbox.data['val']  # changes to z change source_bbox !!???
    # optional, convert to -1, 1
    z[np.where(z < 0)] = -1
    z[np.where(z >= 0)] = 1
    # could also scale z values by numvber of records
    #z = dftmp['Avg. Goldstein Scale'].values * dftmp['Number of Records'].values  

    # add augmented points to arrays
    if use_aug:
        [ngood, nbad, ngood_aug, nbad_aug, cds_good_aug, cds_bad_aug] = auglist
        nodes = np.concatenate((ngood_aug, nbad_aug))
        #goods = np.append(ngood, ngood_aug)
        #bads = np.append(nbad + nbad_aug)
        #z = np.asarray(len(ngood_aug) * [1] + len(nbad_aug) * [-1])
        z_aug = np.concatenate((1.*np.ones(len(ngood_aug)), -1.*np.ones(len(nbad_aug)) ))
        X_tab = []
        dup_i = []
        for i,n in enumerate(nodes):
            xtmp, ytmp = G.nodes[n][xname], G.nodes[n][yname]
            #lat, lon = G.nodes[n]['lat'], G.nodes[n]['lon']
            if [xtmp, ytmp] in X_tab: 
                print ("Duplicate point: i, x, y", i, xtmp, ytmp)
                dup_i.append(i)
                continue
            X_tab.append([xtmp, ytmp])
        z_aug = np.delete(z_aug, dup_i)
        X_aug = np.asarray(X_tab) 
        # append to X, z
        X = np.concatenate((X, X_aug))
        z = np.concatenate((z, z_aug))
        
    # Standard normal distribution functions
    PHI = stats.distributions.norm().cdf
    gp = GaussianProcessRegressor()
    gp.fit(X, z)
    
    # bounds for lines
    minx, maxx = min(X[:, 0]), max(X[:, 0])
    miny, maxy = min(X[:, 1]), max(X[:, 1])
    dx = maxx - minx
    dy = maxy - miny
    bufferx = 0.1 * dx
    buffery = 0.1 * dy

    # Evaluate real function, the prediction and its MSE on a grid
    x1, x2 = np.meshgrid(np.linspace(minx-bufferx, maxx+bufferx, res),
                         np.linspace(miny-buffery, maxy+buffery, res))
    xx = np.vstack([x1.reshape(x1.size), x2.reshape(x2.size)]).T
    
    #y_true = g(xx)
    #y_true = y_true.reshape((res, res))
    y_pred, MSE = gp.predict(xx, eval_MSE=True)
    sigma = np.sqrt(MSE)
    y_pred = y_pred.reshape((res, res))
    sigma = sigma.reshape((res, res))

    if mpl_plot:    
        # Plot the probabilistic classification iso-values using the Gaussian property
        # of the prediction
        fig = pl.figure(1)
        ax = fig.add_subplot(111)
        ax.axes.set_aspect('equal')
        pl.xticks([])
        pl.yticks([])
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        pl.xlabel('$x_1$')
        pl.ylabel('$x_2$')
        
        cax = pl.imshow(np.flipud(PHI(- y_pred / sigma)), cmap=cm.gray_r, alpha=0.8,
                        extent=(minx-bufferx, maxx+bufferx, miny-buffery, maxy+buffery))
        norm = pl.matplotlib.colors.Normalize(vmin=0., vmax=0.9)
        cb = pl.colorbar(cax, ticks=[0., 0.2, 0.4, 0.6, 0.8, 1.], norm=norm)
        cb.set_label('${\\rm \mathbb{P}}\left[\widehat{G}(\mathbf{x}) \leq 0\\right]$')
        
        # negative points
        pl.plot(X[z <= 0, 0], X[z <= 0, 1], 'r.', markersize=12)
        # postiive points
        pl.plot(X[z > 0, 0], X[z > 0, 1], 'b.', markersize=12)
            
    polys = []
    polyx = []
    polyy = []
    poly_colors = []
    #poly_dash = []
    for (conf, color, lstyle) in zip(conf_levels, colors, linestyles):
        result = c.trace(conf)
        # result is a list of arrays of vertices and path codes
        # (see docs for matplotlib.path.Path)
        nseg = len(result) // 2
        segments, codes = result[:nseg], result[nseg:]
        for seg in segments:
            # test if seg is empty
            if remove_empties:
                # create convex hull from points
                points = seg
                hull = scipy.spatial.ConvexHull(points)
                hullx, hully = points[hull.vertices,0], points[hull.vertices,1]
                # test which gdelt_indices are inside hull
                inside_indices = gdelt_inside_hull_v0(source_bbox, hullx, 
                                                      hully, coords=coords)
                # ignore if no sources inside the hull
                if len(inside_indices) == 0:
                    continue
            # else, add to arrays
            # first, smooth
            # http://gis.stackexchange.com/questions/24827/how-to-smooth-the-polygons-in-a-contour-map
            #http://stackoverflow.com/questions/27642237/smoothing-a-2-d-figure
            if smooth:
                # in general, gaussian_filter1d is better, though doesn't 
                # close the loop for polygons
                from scipy.interpolate import interp1d
                from scipy.ndimage.interpolation import spline_filter1d
                x, y = seg.T   
                #print ("x[0] == x[-1]", x[0] == x[-1]
                #print ("y[0] == y[-1]", y[0] == y[-1]
                # if start = end, wrap second point around end so that
                #     the interpolation gives a smooth line
                orig_len = len(x)
                #if (x[0] == x[-1]) and (y[0] == y[-1]):
                    # appending points to x, y resutls in exraneous features
            
                # smooth
                #print ("len(x)", len(x)
                ###############
                # crude adaptive smoothing
                if len(x) < 10:
                    smooth_sigma = 1.25
                elif len(x) >= 10 and len(x) < 20:
                    smooth_sigma = 1.75
                elif len(x) >= 20 and len(x) < 30:
                    smooth_sigma = 2.25
                elif len(x) >= 30 and len(x) < 40:
                    smooth_sigma = 2.75
                elif len(x) >= 40 and len(x) < 50:
                    smooth_sigma = 3.25
                else:
                    smooth_sigma = 4.
                #################
                x3 = gaussian_filter1d(x, smooth_sigma)
                y3 = gaussian_filter1d(y, smooth_sigma)
                # dirty hack to close loops
                if (x[0] == x[-1]) and (y[0] == y[-1]):
                    #print ("len(x)", len(x)
                    x3 = np.concatenate((x3, [x3[0]]))
                    y3 = np.concatenate((y3, [y3[0]]))

               # # if we want splines, do parametric since we have closed
               # # polygons and not functions
               # # http://docs.scipy.org/doc/scipy/reference/tutorial/interpolate.html#spline-interpolation
               # from scipy import interpolate
               # tck,u = interpolate.splprep([x,y], s=0)
               # unew = np.linspace(np.min(x), np.max(x), res)
               # out = interpolate.splev(unew, tck)
               # x3, y3 = out[0], out[1]
                
                seg = np.vstack((x3, y3)).T   
                #print ("seg.shape", seg.shape
                #seg = gaussian_filter(seg, smooth_sigma)
                #print ("seg.shape", seg.shape

            polys.append(seg)
            polyx.append(seg[:,0])
            polyy.append(seg[:,1])
            poly_colors.append(color)
            # add to plot
            if mpl_plot:
                p = pl.Line2D(seg[:,0], seg[:,1], color=color, linestyle=lstyle)
                #p = pl.Polygon(seg, fill=False, color=color)
                ax.add_artist(p)
                
    #print ("polys[0]", polys [0]   
    #print ("polyx[0]", polyx[0]
                
    print ("Time to compute contours:", time.time() - t0, "seconds")
    return polys, polyx, polyy, poly_colors


###############################################################################
### Complex Computation/Plot funcs
###############################################################################
def show_routes2(G, end_nodes, htmlout, htmlout_centr, auglist, 
                g_node_props_dic, source_bbox=None, ecurve_dic=None,
                show_bbox=True, show_aug=True, target=None, 
                skipnodes=[], map_background='None', compute_subgraph_centrality=True, 
                compute_secondary_routes=False, goodroutes=None, binsize=0.5,
                crit_perc=None, coords='latlon',
                edge_weight='Travel Time (h)'):
    '''Compute routes between end_nodes
    Called in show_routes_all2()'''

    glyph_node_list = []

    global_dic = global_vars()
    plot_width = global_dic['plot_width']
     # expand auglist 
    [ngood, nbad, ngood_aug, nbad_aug, cds_good_aug, cds_bad_aug] = auglist
    print ("len( skipnodes)", len(skipnodes))

    # get columndatasources
    esource, esourcerw, source_target, source_sourcenodes, source_crit, \
    source_missing, source_histo_cum, source_histo_bin, \
    source_subgraph_centrality = get_route_sources(G, end_nodes, auglist,    
                            g_node_props_dic, ecurve_dic=ecurve_dic,
                            target=target,
                            skipnodes=skipnodes, goodroutes=goodroutes, 
                            compute_secondary_routes=compute_secondary_routes,
                            compute_subgraph_centrality=compute_subgraph_centrality, 
                            binsize=binsize, crit_perc=crit_perc,
                            edge_weight=edge_weight)
    print ("source_bbox:", source_bbox)
    print ("esource:", esource)
    print ("esourcerw:", esourcerw)
    print ("source_target:", source_target)
    print ("source_sourcenodes:", source_sourcenodes)
    print ("source_crit:", source_crit)
    print ("source_missing:", source_missing)
    print ("source_histo_cum:", source_histo_cum)
    print ("source_histo_bin:", source_histo_bin)
    print ("source_subgraph_centrality:", source_subgraph_centrality)
    
    # get route glyphs
    paths_seg, paths_seg_sec, paths_line, paths_line_sec,\
            target_shape, target_text, sources_shape,\
            sources_text, crit_shape, crit_text, missing_shape, missing_text, \
            subgraph_centrality_shape, subgraph_centrality_text, diff_shape, diff_text, \
            rect_bin, rect_cum, hull_circ, hull_patch, risk_shape = \
                        get_route_glyphs_all(coords=coords)
    # set routes
    if ecurve_dic is not None:
        print ("Use paths_line")
        paths_plot, paths_plot_sec = paths_line, paths_line_sec
    else:
        print ("Use paths_seg")
        paths_plot, paths_plot_sec = paths_seg, paths_seg_sec
        
 
    # set title
    title, centr_title, histo_title = set_routes_title(end_nodes, target=target, 
                            skipnodes=skipnodes)
    # bbox, aug glyphs
    bbox_quad_shape, bbox_text_shape = bokeh_utils.get_bbox_glyph()
    
    
    # get bbox glyph
    bbox_quad_shape, bbox_text_shape = get_bbox_glyph(coords=coords)
    
    # get aug glyph
    shape_aug_good = get_nodes_glyph(coords=coords)
    shape_aug_bad = get_nodes_glyph(coords=coords)

    # plots
    #####################
    # initialize G plot
    # refined plot
    
    plot_, Gnsource, Gesource, node_glyphs, outdict = G_to_bokeh(G,                                             
                #ecurve_dic=ecurve_dic, 
                global_dic=global_dic,
                htmlout=None, 
                map_background=map_background, show_nodes=True, 
                show_road_labels=False, verbose=False, plot_width=plot_width, 
                title=title, show=False, coords=coords)
    print ("Gnsource:", Gnsource)
    print ("Gesource:", Gesource)

    #bk.output_file(htmlout)
    #bk.save(obj=plot) 
    #bk.show(plot)
    #return
    
     # plot secondary paths
    if compute_secondary_routes:
        glyph_route_sec = plot_.add_glyph(esourcerw, paths_plot_sec) 
        
    # plot paths
    glyph_route = plot_.add_glyph(esource, paths_plot)   
        
    # overlay aug    
    if show_aug:    
        glyph_aug_good = plot_.add_glyph(cds_good_aug, shape_aug_good)
        glyph_aug_bad = plot_.add_glyph(cds_bad_aug, shape_aug_bad)

    # overlay bbox
    if show_bbox:    
        #print ("source_bbox:", source_bbox)
        #print ("  source_bbox.data:", source_bbox.data)
        glyph_bbox_shape = plot_.add_glyph(source_bbox, bbox_quad_shape)
        glyph_bbox_text = plot_.add_glyph(source_bbox, bbox_text_shape)
        # add bbox hover?
        hover_table = \
            [
                ("index", "@index"),
                #("Node Name",  "@name"),
                ("Category", "@Category"),
                ("Prob", "@Prob"),
                ("Val", "@Val"),
                #("Color", "@color"),
                ("(x,y)", "($x, $y)"),
                #("(x,y)", "(@Xmid_wmp, @Ymid_wmp)"),
                ("Nearest OSM Node", "@nearest_osm"),
                ("Distance", "@dist")
            ]
        # add tools 
        hover_bbox = HoverTool(tooltips=hover_table, renderers=[glyph_bbox_shape])#, line_policy=line_policy) 
        plot_.add_tools(hover_bbox) 
    
    # plot missing nodes
    glyph_missing_shape = plot_.add_glyph(source_missing, missing_shape)
    glyph_missing_text = plot_.add_glyph(source_missing, missing_text)    
        
    # plot source nodes
    glyph_sources_shape = plot_.add_glyph(source_sourcenodes, sources_shape)
    glyph_sources_text = plot_.add_glyph(source_sourcenodes, sources_text)    

    # plot critical nodes
    glyph_crit_shape = plot_.add_glyph(source_crit, crit_shape)
    glyph_crit_text = plot_.add_glyph(source_crit, crit_text) 

    # plot target nodes
    glyph_target_shape = plot_.add_glyph(source_target, target_shape)
    glyph_target_text = plot_.add_glyph(source_target, target_text)     
    
    # extend node lists
    if show_aug:
        glyph_node_list.extend([glyph_aug_good])
        glyph_node_list.extend([glyph_aug_bad])
    #if show_bbox:
    #    glyph_node_list.extend([glyph_bbox_shape])
    glyph_node_list.extend([glyph_missing_shape])
    glyph_node_list.extend([glyph_sources_shape])
    glyph_node_list.extend([glyph_crit_shape])
    glyph_node_list.extend([glyph_target_shape])
    
    # add hover
    plot_ = add_hover_save(plot_, htmlout=None, show=False, add_hover=True,
                          renderers=glyph_node_list)
    #plot = add_hover_save(plot, htmlout=htmlout, show=True, add_hover=True,
    #                      renderers=glyph_node_list)
    #####################

    bin_str = source_histo_bin.data['x_str']
    binc_arr = source_histo_bin.data['y'] 
    cumc_arr = source_histo_cum.data['y']
    ploth =  plot_histo(bin_str, binc_arr, cumc_arr, title=histo_title, \
                plot_width=plot_width, plot_height=global_dic['histo_height'], 
                ymax=None)
    # add hover
    hover_table = [("Bin",  "@x"),("Count", "@height"),]
    hoverh = HoverTool(tooltips=hover_table)     
    ploth.add_tools(hoverh)  
    #####################

    #print ("test 02" 
    # stack plots, save
    pz0 = column(plot_, ploth)
    #pz0 = bk.vplot(plot, ploth)
    #print ("test03"
    bk.output_file(htmlout)
    #print ("test04"
    # !! Get an error when saving pz0 !
    #####################
    #https://github.com/bokeh/bokeh/issues/3671
    # shockingly, bokeh balloons file sizes by default
    #bk.reset_output()
    #####################
    bk.save(obj=pz0) 
    #print ("test05"
    bk.show(pz0)    
    

    #####################
    #####################
    if compute_subgraph_centrality:
        glyph_node_list
        # reinitialize G plot
        plot, Gnsource, Gesource, node_glyphs, outdict = G_to_bokeh(G, htmlout=None, 
                    #ecurve_dic=ecurve_dic,
                    global_dic=global_dic,
                    map_background=map_background, show_nodes=True, 
                    show_road_labels=False, verbose=False, plot_width=plot_width, 
                    title=title, show=False, coords=coords)
        # skip secondary paths since these aren't included in subgraph!
        #if compute_secondary_routes:
        #    glyph_route_sec = plot.add_glyph(esourcerw, paths_seg_sec)
            
        # plot paths
        glyph_route = plot.add_glyph(esource, paths_plot)  
        
        # overlay aug    
        if show_aug:    
            glyph_aug_good = plot.add_glyph(cds_good_aug, shape_aug_good)
            glyph_aug_bad = plot.add_glyph(cds_bad_aug, shape_aug_bad)   
    
        # overlay bbox
        if show_bbox:    
            glyph_bbox_shape = plot.add_glyph(source_bbox, bbox_quad_shape)
            glyph_bbox_text = plot.add_glyph(source_bbox, bbox_text_shape)
        
        # plot missing nodes
        glyph_missing_shape = plot.add_glyph(source_missing, missing_shape)
        glyph_missing_text = plot.add_glyph(source_missing, missing_text)    
        
        # plot source nodes
        glyph_sources_shape = plot.add_glyph(source_sourcenodes, sources_shape)
        glyph_sources_text = plot.add_glyph(source_sourcenodes, sources_text)    
    
        # skip critical nodes!
        #glyph_crit_shape = plot.add_glyph(source_crit, crit_shape)
        #glyph_crit_text = plot.add_glyph(source_crit, crit_text)
        
        # plot target nodes
        glyph_target_shape = plot.add_glyph(source_target, target_shape)
        glyph_target_text = plot.add_glyph(source_target, target_text) 

        # plot centrality points
        glyph_centr_shape = plot.add_glyph(source_subgraph_centrality, subgraph_centrality_shape)
        glyph_centr_text = plot.add_glyph(source_subgraph_centrality, subgraph_centrality_text)

        # extend node lists
        if show_aug:
            glyph_node_list.extend([glyph_aug_good])
            glyph_node_list.extend([glyph_aug_bad])
        if show_bbox:
            glyph_node_list.extend([glyph_bbox_shape])
        glyph_node_list.extend([glyph_missing_shape])
        glyph_node_list.extend([glyph_sources_shape])
        #glyph_node_list.extend([glyph_crit_shape])
        glyph_node_list.extend([glyph_target_shape])
        glyph_node_list.extend([glyph_centr_shape])
        
        # add hover
        plot = add_hover_save(plot, htmlout=None, show=False, add_hover=True,
                              renderers=glyph_node_list)
        plot.title.text = title + ' (Centrality)'
        
        # stack histogram and save
        pz1 = column(plot, ploth)
        #pz1 = bk.vplot(plot, ploth)
        bk.output_file(htmlout_centr)
        #####################
        #https://github.com/bokeh/bokeh/issues/3671
        # shockingly, bokeh balloons file sizes by default
        bk.reset_output()
        #####################
        bk.save(obj=pz1) 
        bk.show(pz1)  
    #####################        
        
    # return sources
    return esource, esourcerw, source_target, source_sourcenodes, \
        source_crit, source_missing, source_histo_cum, source_histo_bin, \
        source_subgraph_centrality            
    #return crit_nodes, crit_counts

 
###############################################################################               
def show_routes_all2(outroot, G, auglist, source_bbox, g_node_props_dic,
                        # source_bbox=None,
                        ecurve_dic=None, map_background='None',
                        goodroutes=None, target=None,
                        show_bbox=False, show_aug=True,
                        compute_subgraph_centrality=True,
                        compute_secondary_routes=False, binsize=0.5,
                        coords='wmp', edge_weight='Travel Time (h)',
                        verbose=False):
    '''
    Compute and show minimum evacuation or logistic routes between nodes
    if target: compute routes between node_set and target
    else: compute minimum spanning tree beween all nodes in node_set
    node_set_name is a switch to use goodnodes or badnodes for paths.
    '''                

    # expand auglist 
    [ngood, nbad, ngood_aug, nbad_aug, cds_good_aug, cds_bad_aug] = auglist   
    color_dic, alpha_dic = define_colors_alphas()
    global_dic = global_vars()

    # define path endpoints and skipnodes color
    #if node_set_name == 'goodnodes':
    if goodroutes:        
        end_nodes = ngood
        skipnodes = nbad_aug
        #goodroutes = True
        #sourcenode_color = color_dic['goodnode_color']
        #skipnode_color = color_dic['badnode_aug_color']
        #line_color=color_dic['compute_path_color_good']
    else:
        end_nodes = nbad
        skipnodes = ngood_aug
        #goodroutes=False
        #sourcenode_color = color_dic['badnode_color']
        #skipnode_color = color_dic['goodnode_aug_color']
        #line_color=color_dic['compute_path_color_bad']
    
    if verbose:
        print ("len end_nodes:", len(end_nodes))
        #return

    # define bbox shapes
    bbox_quad_shape, bbox_text_shape = get_bbox_glyph(
                                    text_alpha=0.9, text_font_size='5pt',
                                    coords=coords)
    # define aug shapes
    shape_aug_good = get_nodes_glyph(shape='circle', coords=coords)
    shape_aug_bad = get_nodes_glyph(shape='circle', coords=coords)    
    # get route glyphs
    paths_seg, paths_seg_sec, paths_line, paths_line_sec,\
            target_shape, target_text, sources_shape,\
            sources_text, crit_shape, crit_text, missing_shape, missing_text, \
            subgraph_centrality_shape, subgraph_centrality_text, diff_shape, diff_text, \
            rect_bin, rect_cum, hull_circ, hull_patch, risk_shape = \
                    get_route_glyphs_all(coords=coords)
    # set routes
    if ecurve_dic is not None:
        paths_plot, paths_plot_sec = paths_line, paths_line_sec
    else:
        paths_plot, paths_plot_sec = paths_seg, paths_seg_sec 


    # no skipnodes
    htmlout = outroot + '.html'
    htmlout_centr = outroot + '_centr.html'
    clean_skipnodes=[]
    # cds can only be assigned to one plot object, so need to copy them!
    cds_good_aug2 = ColumnDataSource(copy.deepcopy(cds_good_aug.data))
    cds_bad_aug2 = ColumnDataSource(copy.deepcopy(cds_bad_aug.data))
    auglist2 = [ngood, nbad, ngood_aug, nbad_aug, cds_good_aug2, cds_bad_aug2]
    if verbose:
        print ("cds_good_aug2:", cds_good_aug2)
        print ("cds_bad_aug2:", cds_bad_aug2)
    source_bbox2 = ColumnDataSource(copy.deepcopy(source_bbox.data))
    if verbose:
        print ("source_bbox2:", source_bbox2)
    esource, esourcerw, source_target, source_sourcenodes, source_crit_clean, \
    source_missing, source_histo_cum, source_histo_bin, \
    source_subgraph_centrality = show_routes2(G, end_nodes, htmlout, htmlout_centr, 
                    auglist2, g_node_props_dic, source_bbox=source_bbox2, 
                    ecurve_dic=ecurve_dic,
                    show_bbox=show_bbox, 
                    show_aug=show_aug, target=target, 
                    skipnodes=clean_skipnodes, map_background=map_background, 
                    compute_subgraph_centrality=compute_subgraph_centrality, 
                    compute_secondary_routes=compute_secondary_routes, 
                    goodroutes=goodroutes, binsize=binsize, coords=coords,
                    edge_weight=edge_weight)

    # augmented points (with skipnodes)
    htmlout = outroot + '_aug.html'
    htmlout_centr = outroot + '_aug_centr.html'
    aug_skipnodes=skipnodes
    # cds can only be assigned to one plot object, so need to copy them!
    cds_good_aug2 = ColumnDataSource(copy.deepcopy(cds_good_aug.data))
    cds_bad_aug2 = ColumnDataSource(copy.deepcopy(cds_bad_aug.data))
    auglist2 = [ngood, nbad, ngood_aug, nbad_aug, cds_good_aug2, cds_bad_aug2]
    if verbose:
        print ("cds_good_aug2:", cds_good_aug2)
        print ("cds_bad_aug2:", cds_bad_aug2)
    source_bbox2 = ColumnDataSource(copy.deepcopy(source_bbox.data))
    if verbose:
        print ("source_bbox2:", source_bbox2)
    esource, esourcerw, source_target, source_sourcenodes, source_crit_aug, \
    source_missing, source_histo_cum, source_histo_bin, \
    source_subgraph_centrality = show_routes2(G, end_nodes, htmlout, htmlout_centr, 
                    auglist2, 
                    g_node_props_dic, source_bbox2, ecurve_dic=ecurve_dic,
                    show_bbox=show_bbox, 
                    show_aug=show_aug, target=target, 
                    skipnodes=aug_skipnodes, map_background=map_background, 
                    compute_subgraph_centrality=compute_subgraph_centrality, 
                    compute_secondary_routes=compute_secondary_routes, 
                    goodroutes=goodroutes, binsize=binsize, coords=coords,
                    edge_weight=edge_weight)


    ###############
    # diffs
    # find difference in crit points (augmented - clean)       
    # create dictionaries
    htmlout = outroot + '_diffs.html'
    diff_title = 'Differences in Road Network Critical Points, Skip=' + \
        color_dic['diff_node_color'] + ', Raw=' + \
        color_dic['diff_node_color_sec']
    minS, maxS = global_dic['minS'], global_dic['maxS']
    #end_node_alpha = alpha_dic['end_node']
    diff_alpha = alpha_dic['diff']

    crit_nodes_aug = source_crit_aug.data['name']
    crit_counts_aug = source_crit_aug.data['count']
    crit_nodes_clean = source_crit_clean.data['name']
    crit_counts_clean = source_crit_clean.data['count']  
    
    # if arrays are equal, return
    if np.array_equal(crit_nodes_aug, crit_nodes_clean):
        diff_title = "No Differences in Networks With/Without Skip Nodes!"
        plot_, nsource, esource, node_glyphs, outdict = G_to_bokeh(G, 
                #ecurve_dic=ecurve_dic,
                global_dic=global_dic,
                htmlout=htmlout, 
                map_background=map_background, show_nodes=True, 
                show_road_labels=False, verbose=False, 
                plot_width=global_dic['plot_width'], 
                title=diff_title, show=True, coords=coords)
        return
    
    # else, continue
    #print ("skipnodes", skipnodes
    #print ("crit_counts_aug", crit_counts_aug
    #print ("crit_counts_clean", crit_counts_clean   
    ndic_aug = dict(zip(crit_nodes_aug, crit_counts_aug))
    ndic_clean = dict(zip(crit_nodes_clean, crit_counts_clean))
    ndic_clean_neg = dict(zip(crit_nodes_clean, -1*np.asarray(crit_counts_clean)))
    ndic_tot = ndic_aug.copy()
    ndic_tot.update(ndic_clean_neg)
    nset_aug, nset_clean = set(crit_nodes_aug), set(crit_nodes_clean)
    # below gives the differences in counts between critical nodes in nset_aug
    # and nset_clean
    # update values for intersection 
    for item in nset_aug.intersection(nset_clean):
        ndic_tot[item] = ndic_aug[item] - ndic_clean[item]
    ndiffs = np.asarray(list(ndic_tot.keys()))
    cdiffs = np.asarray(list(ndic_tot.values()))
    #print ("cdiffs:", cdiffs)
    sizediffs, Atmp, Btmp = lin_scale(np.abs(cdiffs), minS=minS, maxS=maxS)
    # positive values
    f1 = np.where(cdiffs >= 0)
    #print ("f1", f1
    #print ("type(ndiffs)", type(ndiffs)
    #print ("cdiffs", cdiffs
    #print ("sizediffs", sizediffs
    ndiffs0, cdiffs0, sizediffs0 = ndiffs[f1], cdiffs[f1], sizediffs[f1]
    # negative values
    f2 = np.where(cdiffs < 0)
    ndiffs1, cdiffs1, sizediffs1 = ndiffs[f2], cdiffs[f2], sizediffs[f2] 
    
    # columndatasources
    source_diff0 = bokeh_utils.set_nodes_source(G, ndiffs0, size=sizediffs0, 
                color=color_dic['diff_node_color'], fill_alpha=diff_alpha,
                shape='square', label=cdiffs0, name=ndiffs0, 
                count=cdiffs0)#, legend='Aug-Clean > 0')
                
    source_diff1 = bokeh_utils.set_nodes_source(G, ndiffs1, size=sizediffs1, 
                color=color_dic['diff_node_color_sec'], fill_alpha=diff_alpha,
                shape='square', label=cdiffs1, name=ndiffs1, 
                count=cdiffs1)#, legend='Aug-Clean < 0')
    
    # plots
    #####################
    # initialize G plot
    glyph_node_list = []
    plot_, nsource, esource, node_glyphs, outdict = G_to_bokeh(G, #ecurve_dic=ecurve_dic, 
                global_dic=global_dic,
                htmlout=None, 
                map_background=map_background, show_nodes=True, 
                show_road_labels=False, verbose=False, 
                plot_width=global_dic['plot_width'], 
                title=diff_title, show=False, coords=coords)

     # plot secondary paths
    #if compute_secondary_routes:
    #    glyph_route_sec = plot_.add_glyph(esourcerw, paths_plot_sec)
        
    # plot paths
    #glyph_route = plot_.add_glyph(esource, paths_plot)  

    # overlay aug    
    if show_aug:    
        # cds can only be assigned to one plot object, so need to copy them!
        cds_good_aug2 = ColumnDataSource(copy.deepcopy(cds_good_aug.data))
        cds_bad_aug2 = ColumnDataSource(copy.deepcopy(cds_bad_aug.data))
        glyph_aug_good = plot_.add_glyph(cds_good_aug2, shape_aug_good)
        glyph_aug_bad = plot_.add_glyph(cds_bad_aug2, shape_aug_bad)   

    # overlay bbox
    if show_bbox:    
        # cds can only be assigned to one plot object, so need to copy them!
        source_bbox2 = ColumnDataSource(copy.deepcopy(source_bbox.data)) 
        glyph_bbox_shape = plot_.add_glyph(source_bbox2, bbox_quad_shape)
        glyph_bbox_text = plot_.add_glyph(source_bbox2, bbox_text_shape)    
    
    # plot source nodes
    source_sourcenodes2 = ColumnDataSource(copy.deepcopy(source_sourcenodes.data)) 
    glyph_sources_shape = plot_.add_glyph(source_sourcenodes2, sources_shape)
    glyph_sources_text = plot_.add_glyph(source_sourcenodes2, sources_text)    
    
    # plot diff nodes
    glyph_diffs0_shape = plot_.add_glyph(source_diff0, diff_shape)
    glyph_diffs0_text = plot_.add_glyph(source_diff0, diff_text)    
    glyph_diffs1_shape = plot_.add_glyph(source_diff1, diff_shape)
    glyph_diffs1_text = plot_.add_glyph(source_diff1, diff_text)         
    
    # extend node lists
    if show_aug:
        glyph_node_list.extend([glyph_aug_good])
        glyph_node_list.extend([glyph_aug_bad])
    if show_bbox:
        glyph_node_list.extend([glyph_bbox_shape])
    glyph_node_list.extend([glyph_sources_shape])
    glyph_node_list.extend([glyph_diffs0_shape])
    glyph_node_list.extend([glyph_diffs1_shape])

    plot_.title.text = diff_title    
    # add hover
    p5 = add_hover_save(plot_, htmlout=htmlout, show=True, add_hover=True,
                              renderers=glyph_node_list)

    
    return


###############################################################################               
def route_overlap(outroot, G, source_bbox, auglist, g_node_props_dic, \
                        ecurve_dic=None, use_skipnodes=False, 
                        show_plots=True, target=None, map_background='None', \
                        show_bbox=False, show_aug=True,
                        compute_secondary_routes=True, binsize=0.5,
                        plot_width=1200, show_text=False, coords='wmp',
                        edge_weight='Travel Time (h)'):
    '''    
    Compute and show overlap between good and bad routes
    Right now if show_plots == True, double computing routes, which is 
    obviously very slow.  Could improve performance by only computing once,
    though this would require editing show_routes2()
    '''                

    # expand auglist 
    [ngood, nbad, ngood_aug, nbad_aug, cds_good_aug, cds_bad_aug] = auglist 
    # update auglist
    cds_good_aug2 = ColumnDataSource(copy.deepcopy(cds_good_aug.data))
    cds_bad_aug2 = ColumnDataSource(copy.deepcopy(cds_bad_aug.data))
    auglist2 = [ngood, nbad, ngood_aug, nbad_aug, cds_good_aug2, cds_bad_aug2] 
    
    color_dic, alpha_dic = define_colors_alphas()
    global_dic = global_vars()

    # good
    end_nodes = ngood
    goodroutes = True
    if use_skipnodes:
        skipnodes=nbad_aug
    else:
        skipnodes=[]
    if show_plots:
        # plot 
        htmlout = outroot + '_allied.html'
        if use_skipnodes:
            htmlout = outroot + '_allied_skip.html'
        source_bbox_tmp = ColumnDataSource(copy.deepcopy(source_bbox.data)) 
        esource, esourcerw, source_target, source_sourcenodes, source_crit_g, \
            source_missing, source_histo_cum, source_histo_bin, \
            source_subgraph_centrality = \
                show_routes2(G, end_nodes, htmlout, None, auglist2, 
                    g_node_props_dic, source_bbox=source_bbox_tmp, 
                    ecurve_dic=ecurve_dic, show_bbox=show_bbox, 
                    show_aug=show_aug, target=target, 
                    skipnodes=skipnodes, map_background=map_background, 
                    compute_subgraph_centrality=False, 
                    compute_secondary_routes=compute_secondary_routes, 
                    goodroutes=goodroutes, binsize=binsize, coords=coords,
                    edge_weight=edge_weight)

        # below only computes primary routes, not secondary!
        crit_nodes_g = source_crit_g.data['name']
        crit_counts_g = source_crit_g.data['count']
    #else:
    crit_nodes_g, crit_counts_g = path_counts(G, end_nodes, auglist, 
                goodroutes=goodroutes, target=target, skipnodes=skipnodes,
                compute_secondary_routes=compute_secondary_routes,
                edge_weight=edge_weight)

    # bad 
    end_nodes = nbad
    goodroutes = False
    if use_skipnodes:
        skipnodes=ngood_aug
    else:
        skipnodes=[]
    if show_plots:
        htmlout = outroot + '_adversary.html'
        if use_skipnodes:
            htmlout = outroot + '_adversary_skip.html'
        cds_good_aug2 = ColumnDataSource(copy.deepcopy(cds_good_aug.data))
        cds_bad_aug2 = ColumnDataSource(copy.deepcopy(cds_bad_aug.data))
        auglist2 = [ngood, nbad, ngood_aug, nbad_aug, cds_good_aug2, cds_bad_aug2] 
        # plot 
        source_bbox_tmp = ColumnDataSource(copy.deepcopy(source_bbox.data)) 
        esource, esourcerw, source_target, source_sourcenodes, source_crit_b, \
            source_missing, source_histo_cum, source_histo_bin, \
            source_subgraph_centrality = \
                show_routes2(G, end_nodes, htmlout, None, auglist2, 
                    g_node_props_dic, source_bbox=source_bbox_tmp, 
                    ecurve_dic=ecurve_dic, show_bbox=show_bbox, 
                    show_aug=show_aug, target=target, 
                    skipnodes=skipnodes, map_background=map_background, 
                    compute_subgraph_centrality=False, 
                    compute_secondary_routes=compute_secondary_routes, 
                    goodroutes=goodroutes, binsize=binsize, coords=coords,
                    edge_weight=edge_weight)

        # below only computes primary routes, not secondary!
        crit_nodes_b = source_crit_b.data['name']
        crit_counts_b = source_crit_b.data['count']  
    #else:
    # bad crit nodes
    crit_nodes_b, crit_counts_b = path_counts(G, end_nodes, auglist, 
                    goodroutes=goodroutes, target=target, skipnodes=skipnodes,
                    compute_secondary_routes=compute_secondary_routes,
                    edge_weight=edge_weight)

    # diffs
    # find overlap in crit points     
    # create dictionaries
    diff_title = 'Overlap of Routes Between Allied and Adversary Nodes'
    if use_skipnodes:
        diff_title = diff_title + ' (Avoid OPFOR)'
    minS, maxS = global_dic['minS'], global_dic['maxS']
    #end_node_alpha = alpha_dic['end_node']
    diff_alpha = alpha_dic['diff']
            
    ndic_g = dict(zip(crit_nodes_g, crit_counts_g))
    ndic_b = dict(zip(crit_nodes_b, crit_counts_b))
    #ndic_b_neg = dict(zip(crit_nodes_b, -1*np.asarray(crit_counts_b)))
    #ndic_tot = ndic_g.copy()
    #ndic_tot.update(ndic_b_neg)
    nset_g, nset_b = set(crit_nodes_g), set(crit_nodes_b)
    # below gives the sum of of counts between critical nodes in nset_g
    # and nset_b
    # update values for intersection 
    ndic_tot = {}
    for item in nset_g.intersection(nset_b):
        ndic_tot[item] = ndic_g[item] + ndic_b[item]
    #print("ndic_tot.keys():", ndic_tot.keys())
    ndiffs = np.asarray(list(ndic_tot.keys()))
    cdiffs = np.asarray(list(ndic_tot.values()))
    sizediffs, Atmp, Btmp = lin_scale(np.abs(cdiffs), minS=minS, maxS=maxS)
    # positive values
    f1 = np.where(cdiffs >= 0)
    ndiffs0, cdiffs0, sizediffs0 = ndiffs[f1], cdiffs[f1], sizediffs[f1]
     
    # columndatasources
    source_diff0 = bokeh_utils.set_nodes_source(G, ndiffs0, size=sizediffs0, 
                color=color_dic['overlap_node_color'], fill_alpha=diff_alpha,
                shape='invertedtriangle', label=cdiffs0, name=ndiffs0, 
                count=cdiffs0)

    if show_plots:    
        # plots
        #####################
        htmlout = outroot + '_overlap.html'
        if use_skipnodes:
            htmlout = outroot + '_overlap_skip.html'
        # initialize G plot
        glyph_node_list=[]
        plot, nsource, esource, node_glyphs, outdict = G_to_bokeh(G, 
                    #ecurve_dic=ecurve_dic, 
                    htmlout=None, 
                    global_dic=global_dic,
                    map_background=map_background, show_nodes=True, 
                    show_road_labels=False, verbose=False, 
                    plot_width=plot_width, 
                    title=diff_title, show=False, coords=coords)
    
        # overlay aug    
        if show_aug:    
            cds_good_aug2 = ColumnDataSource(copy.deepcopy(cds_good_aug.data))
            cds_bad_aug2 = ColumnDataSource(copy.deepcopy(cds_bad_aug.data))
            shape_aug_good = get_nodes_glyph(shape='circle', coords=coords)
            shape_aug_bad = get_nodes_glyph(shape='circle', coords=coords)  
            glyph_aug_good = plot.add_glyph(cds_good_aug2, shape_aug_good)
            glyph_aug_bad = plot.add_glyph(cds_bad_aug2, shape_aug_bad)   
    
        # overlay bbox
        if show_bbox:    
            bbox_quad_shape, bbox_text_shape = get_bbox_glyph(
                                       text_alpha=0.9, text_font_size='5pt',
                                       coords=coords)
            source_bbox_tmp = ColumnDataSource(copy.deepcopy(source_bbox.data)) 
            glyph_bbox_shape = plot.add_glyph(source_bbox_tmp, bbox_quad_shape)
            glyph_bbox_text = plot.add_glyph(source_bbox_tmp, bbox_text_shape)    
    
        # get route glyphs
        paths_seg, paths_seg_sec, paths_line, paths_line_sec,\
                target_shape, target_text, sources_shape,\
                sources_text, crit_shape, crit_text, missing_shape, missing_text, \
                subgraph_centrality_shape, subgraph_centrality_text, diff_shape, diff_text, \
                rect_bin, rect_cum, hull_circ, hull_patch, risk_shape =\
                                get_route_glyphs_all(coords=coords)
        
        # plot source nodes
        #glyph_sources_shape = plot.add_glyph(source_sourcenodes, sources_shape)
        #glyph_sources_text = plot.add_glyph(source_sourcenodes, sources_text)    
        
        # plot diff nodes
        glyph_diffs0_shape = plot.add_glyph(source_diff0, diff_shape)
        if show_text:
            glyph_diffs0_text = plot.add_glyph(source_diff0, diff_text)    
        
        # extend node lists
        if show_aug:
            glyph_node_list.extend([glyph_aug_good])
            glyph_node_list.extend([glyph_aug_bad])
        if show_bbox:
            glyph_node_list.extend([glyph_bbox_shape])
        glyph_node_list.extend([glyph_diffs0_shape])
    
        plot.title.text = diff_title    
        # add hover
        p5 = add_hover_save(plot, htmlout=htmlout, show=True, add_hover=True,
                                  renderers=glyph_node_list)
            
        
    return source_diff0


###############################################################################
def plot_hull(G, p, hulln, hullx, hully, color='red', 
                  fill_alpha=0.15, line_alpha=0.9, line_width=4,
                  hull_name='Hull', coords='latlon',
                  extend_hull_glyph=False):
    '''Plot predefined hull'''

    glyph_node_list = []        
    # define columndatasource    
    hullsource = bk.ColumnDataSource(dict(x=hullx, y=hully))
    # plot hull nodes
    vals = np.asarray(len(hulln)*[hull_name])
    px, hull_glyphs = plot_nodes(G, p, hulln, size=8, color=color, val=vals, coords=coords)
    glyph_node_list.extend(hull_glyphs)
    # plot patch
    # https://github.com/bokeh/bokeh/blob/master/tests/glyphs/Patch.py        
    patch = Patch(x='x', y='y', fill_color=color, \
        fill_alpha=fill_alpha, line_color=color, line_width=line_width, 
        line_alpha=line_alpha)
    glyphy = px.add_glyph(hullsource, patch) 
    if extend_hull_glyph:
        glyph_node_list.extend([glyphy])
    
    #print ("Time to plot hull:", time.time()-t0, "seconds"
    return px, glyph_node_list


############################################################################### 
def risk_plot(G, source_bbox, auglist, g_node_props_dic, outdir, \
                        #ecurve_dic=None, 
                        goodroutes = True, map_background='None',
                        show_bbox=True, show_aug=True, 
                        plot_width=1200, kdtree=None, kd_idx_dic=None, 
                        r_m=100., edge_weight='Travel Time (h)',
                        coords='wmp', verbose=False):
    '''    
    Compute and show inferred risk
    '''                

    # expand auglist 
    [ngood, nbad, ngood_aug, nbad_aug, cds_good_aug, cds_bad_aug] = auglist   
    color_dic, alpha_dic = define_colors_alphas()
    global_dic = global_vars()

    if goodroutes:
        node_set_name = 'alliednodes'
        sourcenodes=nbad
        ignorenodes = nbad_aug
        skipnodes = ngood_aug
        targetnodes=ngood # set to [] for all nodes

    else:
        node_set_name = 'adversarynodes'
        sourcenodes=ngood
        ignorenodes = ngood_aug
        skipnodes=nbad_aug
        targetnodes=nbad # set to [] for all nodes

    # plots
    #####################    
    html_outs = [os.path.join(outdir, 'risk_' + node_set_name + '.html'),
                 os.path.join(outdir, 'risk_' + node_set_name + '_skipnodes.html')]
    titles = ['Inferred Risk - ' + node_set_name, 
              'Inferred Risk - ' + node_set_name + ' - skipnodes']
    skips = [[], skipnodes]
    # make plots
    for htmlout, skipn, title in zip(html_outs, skips, titles):
        glyph_node_list = []

        source_risk = compute_risk(G, sourcenodes,
                    g_node_props_dic, 
                    target_nodes=targetnodes, skipnodes=skipn, 
                    ignore_nodes=ignorenodes, weight=edge_weight,
                    kdtree=kdtree, kd_idx_dic=kd_idx_dic, r_m=r_m,
                    verbose=verbose)
    
        # initialize G plot
        plot, nsource, esource, node_glyphs, outdict = G_to_bokeh(G, 
                    global_dic=global_dic, htmlout=None, 
                    map_background=map_background, show_nodes=True, 
                    show_road_labels=False, verbose=False, 
                    plot_width=plot_width, 
                    title=title, show=False, coords=coords)


        glyph_node_list.extend(node_glyphs)
        print ("glyph_node_list", glyph_node_list)
        # overlay aug    
        if show_aug:    
            # cds can only be assigned to one plot object, so need to copy them!
            cds_good_aug2 = ColumnDataSource(copy.deepcopy(cds_good_aug.data))
            cds_bad_aug2 = ColumnDataSource(copy.deepcopy(cds_bad_aug.data))
            #auglist2 = [ngood, nbad, ngood_aug, nbad_aug, cds_good_aug2, cds_bad_aug2]
            shape_aug_good = get_nodes_glyph(shape='circle', coords=coords)
            shape_aug_bad = get_nodes_glyph(shape='circle', coords=coords)  
            glyph_aug_good = plot.add_glyph(cds_good_aug2, shape_aug_good)
            glyph_aug_bad = plot.add_glyph(cds_bad_aug2, shape_aug_bad)   
            glyph_node_list.extend([glyph_aug_good, glyph_aug_bad])
        # overlay gdelt
        if show_bbox:  
            bbox_quad_shape, bbox_text_shape = get_bbox_glyph(
                                       text_alpha=0.9, text_font_size='5pt',
                                       coords=coords)
            source_bbox2 = ColumnDataSource(copy.deepcopy(source_bbox.data))
            glyph_bbox_shape = plot.add_glyph(source_bbox2, bbox_quad_shape)
            glyph_bbox_text = plot.add_glyph(source_bbox2, bbox_text_shape) 
            #gdelt_circle_shape, gdelt_text_shape = get_gdelt_glyph(
            #                           text_alpha=0.9, text_font_size='5pt',
            #                           coords=coords)
            #glyph_gdelt_shape = plot.add_glyph(source_gdelt, gdelt_circle_shape)
            #glyph_gdelt_text = plot.add_glyph(source_gdelt, gdelt_text_shape) 
            try:
                glyph_node_list.extend(glyph_bbox_shape)
            except:
                glyph_node_list.append(glyph_bbox_shape)
                
        # add risk glyphs   
        shape_risk = get_nodes_glyph(shape='diamond', coords=coords)
        glyph_risk = plot.add_glyph(source_risk, shape_risk) 
        try:
            glyph_node_list.extend(glyph_risk)
        except:
            glyph_node_list.append(glyph_risk)
            
        plot.title.text = title
        p5 = add_hover_save(plot, htmlout, show=True, add_hover=True, 
                            renderers=glyph_node_list)     

    return p5


###############################################################################
def compute_travel_time(G, target, sourcenodes, g_node_props_dic, 
        ecurve_dic=None, skipnodes=set(), htmloutdir=None,
        map_background='None', size_key='count',
        binsize=0.5, plot_width=1200, source_bbox=None, auglist=None,
        node_set_name='alliednodes', 
        edge_weight='Travel Time (h)', coords='wmp'):
            
    '''Compute number of forces that can arrive at target from sourcenodes over
    time slices.  Plot paths and number of forces
    binsize is the time binning (in hours) of slices
    source_node_name is a switch (goodnodes, badnodes) for coloring'''

    glyph_node_list = [] # should be inherited from G!!!

    t0 = time.time()
    btitle = 'Forces arriving at node ' + str(target) + ' over time'
    outroot = 'travel_time'
    color_dic, alpha_dic = define_colors_alphas()
    global_dic = global_vars()
    #weight = global_dic['edge_weight']
    path_width = global_dic['compute_path_width']
    line_alpha = alpha_dic['compute_paths']
    # node sizes
    #size_key = 'count'
    minS, maxS = global_dic['minS'], global_dic['maxS']
    node_size_skip=6

    target_size = 2*global_dic['maxS']
    target_font_size = '9pt'
    target_font_style = 'normal'
    target_label = 'Target'
    #sourcenode_size = 15
    sourcenode_font_size = '8pt'
    sourcenode_font_style = 'normal'

    # define colors
    if node_set_name == 'alliednodes':
    #if source_node_name == 'goodnodes':
        sourcenode_color = color_dic['goodnode_color']
        skipnode_color = color_dic['badnode_aug_color']
        edge_color = color_dic['compute_path_color_good']

    else:
        sourcenode_color = color_dic['badnode_color']
        skipnode_color = color_dic['goodnode_aug_color']
        edge_color = color_dic['compute_path_color_bad']
    missing_node_color=color_dic['missing_node_color']
    target_color = color_dic['target_color']
    
    #TOOLS="pan,wheel_zoom,box_zoom,resize,reset,hover,previewsave"
    TOOLS="pan,wheel_zoom,box_zoom,reset,hover,save"
    #from bokeh.charts import Bar, Area, Step, Histogram

    if not os.path.exists(htmloutdir):
        os.mkdir(htmloutdir)
 
    # copy graph
    G2 = G.copy()
    # remove desired nodes
    if len(skipnodes) != 0:
        # ensure sourcenodes or target not in skipnodes
        skipnodes = [i for i in skipnodes if i!=target and i not in sourcenodes]
        G2.remove_nodes_from(skipnodes)
            
    # alternative: overkill to compute all paths, is actually faster...
    # computing paths from source, not into source, so if graph is directed
    # this will cause problems
    t1 = time.time()
    n1 = target
    lengthd, pathd = nx.single_source_dijkstra(G2, source=n1, 
                                               weight=edge_weight) 
    # not all nodes may be reachable from N0, so find intersection
    startnodes = list(set(pathd.keys()).intersection(set(sourcenodes)))
    missingnodes = list(set(sourcenodes) - set(startnodes))
    paths = [pathd[k] for k in startnodes]
    lengths = [lengthd[k] for k in startnodes] 
    counts = [len(g_node_props_dic[n]['index']) for n in startnodes]
    #counts = [g_node_props_dic[n]['count'] for n in startnodes]
    vals = [str(round(l,2)) + 'H' for l in lengths]
    counts_blocked = [len(g_node_props_dic[n]['index']) for n in missingnodes]
    #counts_blocked = [g_node_props_dic[n]['count'] for n in missingnodes]
    #counts = len(startnodes) * [1]
    print ("Time to compute paths:", time.time() - t1, "seconds")
    # Time to compute paths: 0.0611748695374 seconds
    #print ("n1", n1
    #print ("skipnodes", skipnodes
    #print ("pathd.keys()", pathd.keys()
    #print ("sourcenodes", sourcenodes
    #print ("startnodes", startnodes
    #print ("counts", counts
    if len(counts) == 0:
        print ("Target node is unreachable from all sources!")
        return None
        
    # get constants for size transform 
    sourcesizes, A, B = log_scale(counts, minS, maxS)
    #print ("badsizes", badsizes

    # Take paths, lengths, and counts and determine rate of arrival
    # first, bin by time
    bins = np.arange(binsize, max(lengths)+binsize, binsize)
    digitized = np.digitize(lengths, bins)
    #print ("lengths:", lengths)
    #print ("bins:", bins)
    
    ###############
    # plot initial force distribution
    # roads
    glyph_node_list=[]
    p0, Gnsource, Gesource, node_glyphs, outdict = G_to_bokeh(G, 
                                    #ecurve_dic=ecurve_dic, 
                                    htmlout=None, 
                                    global_dic=global_dic,
                                    map_background=map_background, 
                                    show_nodes=True, plot_width=plot_width,
                                    coords=coords)
    # plot skipnodes
    if len(skipnodes) > 0:
        p0, skipglyphs = plot_nodes(G, p0, skipnodes, 
                size=node_size_skip, color=skipnode_color, 
                fill_alpha=alpha_dic['end_node'], 
                val=len(skipnodes)*['skip'], coords=coords)
    # target
    p0, targetglyphs = plot_nodes(G, p0, [target], shape='invertedtriangle',
            size=target_size, text_font_size=target_font_size, \
            text_font_style=target_font_style, color=target_color, \
            text_alpha=1.0, label=[target_label], val=[target_label], 
            coords=coords)
    # sourcenodes
    p0, sourceglyphs = plot_nodes(G, p0, startnodes, 
                size=sourcesizes, color=sourcenode_color, label=counts, 
                count=counts, val=vals, 
                fill_alpha=alpha_dic['end_node'], 
                text_alpha=alpha_dic['label_bold'], 
                text_font_size=sourcenode_font_size, 
                text_font_style=sourcenode_font_style, coords=coords)
    # plot missing nodes
    missingnodes_count = [len(g_node_props_dic[n]['index']) for n in missingnodes]
    #missingnodes_count = [g_node_props_dic[n]['count'] for n in missingnodes] 
    p0, missingglyphs = plot_nodes(G, p0, missingnodes, shape='square',
        size=maxS, color=color_dic['missing_node_color'], 
        fill_alpha=alpha_dic['end_node'], 
        label=len(missingnodes)*['Obstructed!'], \
        val=len(missingnodes)*['Obstructed!'], name=missingnodes, \
        count=missingnodes_count, coords=coords) 
    # add augmented points
    if auglist is not None:
        [ngood, nbad, ngood_aug, nbad_aug, cds_good_aug, cds_bad_aug] = auglist
        # cds can only be assigned to one plot object, so need to copy them!
        cds_good_aug2 = ColumnDataSource(copy.deepcopy(cds_good_aug.data))
        cds_bad_aug2 = ColumnDataSource(copy.deepcopy(cds_bad_aug.data))
        shape_aug_good = get_nodes_glyph(shape='circle', coords=coords)
        shape_aug_bad = get_nodes_glyph(shape='circle', coords=coords)
        glyph_aug_good = p0.add_glyph(cds_good_aug2, shape_aug_good)
        glyph_aug_bad = p0.add_glyph(cds_bad_aug2, shape_aug_bad)         
    # if desired, plot bbox
    if source_bbox is not None:
        source_bbox2 = ColumnDataSource(copy.deepcopy(source_bbox.data))
        bbox_quad_shape, bbox_text_shape = get_bbox_glyph(coords=coords)
        glyph_bbox_shape = p0.add_glyph(source_bbox2, bbox_quad_shape)
        glyph_bbox_text = p0.add_glyph(source_bbox2, bbox_text_shape) 
        
    # set plot title
    p0.title.text = 'Force Distribution Surrounding ' + target_label
    htmlout = os.path.join(htmloutdir, outroot + '00.html')

    # extend node lists
    if auglist is not None:
        glyph_node_list.extend([glyph_aug_good])
        glyph_node_list.extend([glyph_aug_bad])
    if source_bbox is not None:
        glyph_node_list.extend([glyph_bbox_shape])
    if len(skipnodes) > 0:
        glyph_node_list.extend(skipglyphs)
    glyph_node_list.extend(targetglyphs)
    glyph_node_list.extend(sourceglyphs)
    glyph_node_list.extend(missingglyphs)

    # add hover
    p0 = add_hover_save(p0, htmlout=htmlout, show=True, add_hover=True,
                              renderers=glyph_node_list)
    #p0 = add_hover_save(p0, htmlout, show=True)           

    ###############    
    print ("Looping through bins...")
    globc_arr, totc_arr = len(bins)*[0], len(bins)*[0]
    bin_str = [str(round(b-binsize,2)) + '-' + str(round(b,2)) + 'H' for b in bins]

    globc = 0
    for i,b in enumerate(bins):
        
        print (i, "bin:", b)
        #htmlout = os.path.join(htmloutdir, outroot + str(i+10) + '.html')
             
        glyph_node_list=[]
        p, Gnsource, Gesource, node_glyphs, outdict = G_to_bokeh(G, 
                                                        #ecurve_dic=ecurve_dic,
                                                        htmlout=None, 
                                                        global_dic=global_dic,
                                                        map_background=map_background, 
                                        show_nodes=True, plot_width=plot_width,
                                        coords=coords)

        # plot skipnodes
        if len(skipnodes) > 0:
            p, skipglyphs = plot_nodes(G, p, skipnodes, 
                    size=node_size_skip, color=skipnode_color, 
                    fill_alpha=alpha_dic['end_node'], 
                    val=len(skipnodes)*['skip'], coords=coords)
            glyph_node_list.extend(skipglyphs)

        # target
        p, targetglyphs = plot_nodes(G, p, [target], shape='invertedtriangle',
            size=target_size, text_font_size=target_font_size, \
            text_font_style=target_font_style, color=target_color, \
            text_alpha=1.0, label=[target_label], val=[target_label], 
            coords=coords)
        glyph_node_list.extend(targetglyphs)

        # add augmented points
        if auglist is not None:
            cds_good_aug2 = ColumnDataSource(copy.deepcopy(cds_good_aug.data))
            cds_bad_aug2 = ColumnDataSource(copy.deepcopy(cds_bad_aug.data))
            shape_aug_good = get_nodes_glyph(shape='circle', coords=coords)
            shape_aug_bad = get_nodes_glyph(shape='circle', coords=coords)
            #print ("cds_good_aug2:", cds_good_aug2)
            #print ("cds_bad_aug2:", cds_bad_aug2)
            glyph_aug_good = p.add_glyph(cds_good_aug2, shape_aug_good)
            glyph_aug_bad = p.add_glyph(cds_bad_aug2, shape_aug_bad)         
            glyph_node_list.extend([glyph_aug_good])
            glyph_node_list.extend([glyph_aug_bad])
        # if desired, plot bbox
        if source_bbox is not None:
            source_bbox2 = ColumnDataSource(copy.deepcopy(source_bbox.data))
            source_bbox3 = ColumnDataSource(copy.deepcopy(source_bbox.data))
            #print ("source_bbox2:", source_bbox2)
            #print ("source_bbox3:", source_bbox3)
            bbox_quad_shape, bbox_text_shape = get_bbox_glyph(coords=coords)
            glyph_bbox_shape = p.add_glyph(source_bbox2, bbox_quad_shape)
            glyph_bbox_text = p.add_glyph(source_bbox3, bbox_text_shape) 
            glyph_node_list.extend([glyph_bbox_shape])
                    
        htmlout = os.path.join(htmloutdir, outroot + str(i+10) + '.html')
        # get indices into lenths and paths
        idxs = np.where(digitized==i)[0]
        # if idxs is empty, continue
        if len(idxs) == 0: 
            globc_arr[i] = globc
            continue
        #add_hover_save(p, htmlout=htmlout, show=True)
        #return

        # set total counts arriving
        totc = 0            
        # iterate through individual nodes in the bin
        #print ("idxs:", idxs)
        for idx in idxs:
            len0 = lengths[idx]
            path0 = paths[idx]
            count0 = counts[idx]
            node0 = startnodes[idx]
            totc += count0
            # get size:
            size0 = log_transform(count0, A, B)
            
            #############
            # plot data
            text = 'T='+str(round(len0,2))+' C='+str(count0)
            # plot point
            p, pointglyph = plot_nodes(G, p, [node0], size=6, color=sourcenode_color,
                    fill_alpha=alpha_dic['end_node'], shape='circle', 
                    label=[text], count=[count0], val=[str(round(len0,2))+'H'],
                    name=[node0], text_alpha=alpha_dic['label_bold'], 
                    text_font_size=sourcenode_font_size, 
                    text_font_style=sourcenode_font_style,
                    coords=coords)
            glyph_node_list.extend(pointglyph)

            # paths
            #ex0, ey0, ex1, ey1, emx, emy, elen, edgelist = \
            ex0, ey0, ex1, ey1, elat0, elon0, elat1, elon1, emx, emy, elen, edgelist = \
                    get_path_pos(G, [path0], [len0], skipset=set())    
            # create columndatasource
            esource = bk.ColumnDataSource(
                data=dict(
                    ex0=ex0, ey0=ey0,
                    ex1=ex1, ey1=ey1,
                    elat0=elat0, elon0=elon0,
                    elat1=elat1, elon1=elon1,
                    emx=emx, emy=emy,
                    ealpha=line_alpha*np.ones(len(ex0)),#len(ex0)*[line_alpha]
                    ewidth=np.asarray(len(ex0)*[path_width]),
                    ecolor=np.asarray(len(ex0)*[edge_color]),
                    #add stuff here 
                )
            )    
                                
            # set routes
            if ecurve_dic is not None:
                seg = get_paths_glyph_line(coords=coords)
                elx0, ely0, ellat0, ellon0 = get_ecurves(edgelist, ecurve_dic)  
                # add to esource
                esource.data['elx0'] = elx0
                esource.data['ely0'] = ely0
                esource.data['ellon0'] = ellon0
                esource.data['ellat0'] = ellat0              
        
            else:
                seg = get_paths_glyph_seg(coords=coords)            
                    
            # add to plot
            p.add_glyph(esource, seg)
            #p = plot_paths(p, esource, gmap_background=gmap_background, \
            #        show_labels=False, line_alpha=line_alpha, legend='Paths')

        # update total count                
        globc += totc        
        # set plot title, count arrays
        globc_arr[i] = globc
        totc_arr[i] = totc
        title = str(totc) + ' Units Arrive In ' + bin_str[i] + \
                ' (Total Arrived=' + str(globc) + ')'
        p.title.text = title       
        
        #add_hover_save(p, htmlout=htmlout, show=True)
        #return
    
        ######
        # create bar figure
        pb = plot_histo(bin_str, totc_arr, globc_arr, title=btitle, \
              plot_width=plot_width, plot_height=300, ymax=1.1*np.sum(counts))       
        
        
        # add hover
        # p = add_hover_save(p, htmlout=None)           
        add_hover_save(p, htmlout=None, show=False, add_hover=True,
                              renderers=glyph_node_list)

        # stack plots
        pz = column(p, pb)
        #pz = bk.vplot(p, pb)
        # save
        bk.output_file(htmlout)
        #####################
        #https://github.com/bokeh/bokeh/issues/3671
        # shockingly, bokeh balloons file sizes by default
        #bk.reset_output()
        #####################
        bk.save(obj=pz)    
        bk.show(pz)    

    # make one more plot with all paths, add to p0
    htmlout = os.path.join(htmloutdir, outroot + str(10+len(bins)) + '_routes.html') 

    #ex0, ey0, ex1, ey1, emx, emy, elen, edgelist = \
    ex0, ey0, ex1, ey1, elat0, elon0, elat1, elon1, emx, emy, elen, edgelist =\
        get_path_pos(G, paths, lengths)    
    # create columndatasource
    esource = bk.ColumnDataSource(
        data=dict(
            ex0=ex0, ey0=ey0,
            ex1=ex1, ey1=ey1,
            elat0=elat0, elon0=elon0,
            elat1=elat1, elon1=elon1,
            emx=emx, emy=emy,
            ealpha=line_alpha*np.ones(len(ex0)),#len(ex0)*[line_alpha]
            ewidth=np.asarray(len(ex0)*[path_width]),
            ecolor=np.asarray(len(ex0)*[edge_color]) 
        )
    )   
    # set routes
    if ecurve_dic is not None:
        seg = get_paths_glyph_line(coords=coords)
        elx0, ely0, ellat0, ellon0 = get_ecurves(edgelist, ecurve_dic)  
        # add to esource
        esource.data['elx0'] = elx0
        esource.data['ely0'] = ely0
        esource.data['ellon0'] = ellon0
        esource.data['ellat0'] = ellat0

    else:
        seg = get_paths_glyph_seg(coords=coords)    
    p0.add_glyph(esource, seg)
    #p0 = add_hover_save(p0, htmlout=htmlout, show=True)
    #return
    
    #p0 = plot_paths(p0, esource, gmap_background=gmap_background, \
    #                show_labels=False, line_alpha=line_alpha, legend='Paths') 
    
    ## add sources that cannot reach the target (already done above)
    ## plot missing nodes
    #missingnodes_count = [g_node_props_dic[n]['count'] for n in missingnodes] 
    #p0, sourceB = plot_nodes(G, p0, missingnodes, gmap_background=gmap_background, \
    #    size=maxS, color=color_dic['missing_node_color'], fill_alpha=0.5, \
    #    labels=len(missingnodes)*['Obstructed!'], \
    #    val=len(missingnodes)*['Obstructed!'], nid=missingnodes, \
    #    count=missingnodes_count)                     
    p0.title.text = 'All Routes to ' + target_label

    #p0 = add_hover_save(p0, htmlout=htmlout, show=True)
    #return

    # create bar figure
    pbf = plot_histo(bin_str, totc_arr, globc_arr, title=btitle, \
              plot_width=plot_width, plot_height=300, ymax=1.1*np.sum(counts))
    #print ("pbf:", pbf)
        
    # stack plots?
    pzf = p0 # this works
    #pzf = column(p0, pbf)  # this fails!
    # save
    #####################
    #https://github.com/bokeh/bokeh/issues/3671
    # shockingly, bokeh balloons file sizes by default
    bk.reset_output()
    #####################
    bk.output_file(htmlout)
    bk.save(obj=pzf) 
    bk.show(pzf)    
    
    # optional: standalone bar chart 
    #   since we can't stack the charts for some dumb reason 
    #   (see 13 lines above), save it separately
    htmlout = os.path.join(htmloutdir, outroot + str(10+len(bins)) + '_histo.html') 
    bk.reset_output()
    bk.output_file(htmlout)
    bk.save(obj=pbf) 
    bk.show(pbf)    
    ## create bar figure
    #pbf = plot_histo(bin_str, totc_arr, globc_arr, title=btitle, \
    #          plot_width=plot_width, plot_height=300, ymax=1.1*np.sum(counts))
    #print ("pbf:", pbf)
    #
    #bar_data = {'Binned':totc_arr, 'Cumulative':globc_arr}               
    #pb2 = Bar(bar_data, cat=bin_str, stacked=False, \
    #        title=btitle, xlabel='Time Bins', ylabel='Counts', \
    #        legend=True, width=plot_width, height=400)#, \
    #        #y_range=[0, 1.1*np.sum(counts)])
    #bk.output_file(htmlout)
    #bk.save(obj=pb2)           

    return 
    

###############################################################################
def compute_spread(G, sourcenode, ecurve_dic=None, skipnodes=set([]), 
        binsize=0.5, htmloutdir=None, map_background='None', plot_width=1200, 
        use_hull=True, source_bbox=None, auglist=None, 
        node_set_name='alliednodes', \
        concave=True, concave_alpha=2, coords='latlon', 
        edge_weight='Travel Time (h)'):
    '''Compute spread of forces from initial node.
    Very similar to compute_travel_time, above
    binsize is in units of hours
    node_set_name is a switch (goodnodes, badnodes) for coloring
    concave_alpha is the alpha parameter for concave hulls
    !! Need to only plot new nodes and edges at each time step !!'''

    t0 = time.time()
    os.makedirs(htmloutdir, exist_ok=True)
    #print ("coords for compute_spread():", coords)
    from bokeh.palettes import Spectral10 as pal0
    # stack palette
    palette = pal0 + pal0 + pal0 + pal0
    outroot = 'spread'
    color_dic, alpha_dic = define_colors_alphas()
    global_dic = global_vars()
    #weight = global_dic['edge_weight']
    path_width = global_dic['compute_path_width']
    line_alpha = alpha_dic['compute_paths']

    color_dic, alpha_dic = define_colors_alphas()
    size_mult2 = 1.0  # 1.5
    source_size = size_mult2*global_dic['maxS']
    source_font_size = '8pt'
    source_font_style = 'bold' #normal
    source_color = color_dic['target_color']
    source_label = 'Source'
    target_size = 10
    target_color = color_dic['source_color']
    target_font_size = '7pt'
    target_font_style = 'normal'
    node_size_skip=6
    
    # define colors
    if node_set_name == 'alliednodes':#'goodnodes':
        #source_color = color_dic['goodnode_color']
        skipnode_color = color_dic['badnode_aug_color']
    else:
        #source_color = color_dic['badnode_color']
        skipnode_color = color_dic['goodnode_aug_color']
    edge_seen_color = color_dic['spread_seen_color']
    edge_new_color = color_dic['spread_new_color']
    target_color = edge_new_color
    
    if not os.path.exists(htmloutdir):
        os.mkdir(htmloutdir)

    # copy graph
    G2 = G.copy()
    # remove desired nodes
    if len(skipnodes) != 0:
        # ensure targets or goodnote not in skipnodes
        skipnodes = [i for i in skipnodes if i!=sourcenode]
        G2.remove_nodes_from(skipnodes)

    # alternative: much faster...
    # computing paths from sourcenode, not into sourcenode, so if graph is directed
    # this will cause problems
    t1 = time.time()
    n1 = sourcenode
    lengthd, pathd = nx.single_source_dijkstra(G2, source=n1, 
                                               weight=edge_weight) 
    startnodes = np.asarray(list(pathd.keys()))
    paths = [pathd[k] for k in startnodes]
    lengths = [lengthd[k] for k in startnodes]  
    print ("Time to compute paths:", time.time() - t1, "seconds")
    # # loop: Time to compute paths: 103.538288116 seconds
    # dijkstra: Time to compute paths: 0.0634140968323 seconds

    # Now take paths, lengths, and counts and determine rate of arrival
    # first, bin by time
    #bins = np.arange(binsize, max(lengths)+binsize, binsize)
    #digitized = np.digitize(lengths, bins)
    bins, bin_str, digitized, tmp0, tmp1 = \
            make_histo_arrs(lengths, y=[], binsize=binsize)
    bin_diff = bins[1] - bins[0]
    
    ###############
    # plot initial force distribution
    ###############
    # roads
    glyph_node_list=[]
    p0, Gnsource, Gesource, node_glyphs, outdict = G_to_bokeh(G, htmlout=None, 
                                                     #ecurve_dic=ecurve_dic,
                                                     global_dic=global_dic,
                                    map_background=map_background, 
                                    show_nodes=True, plot_width=plot_width,
                                    coords=coords)
    # plot skipnodes
    if len(skipnodes) > 0:
        p0, skipglyph = plot_nodes(G, p0, skipnodes, 
                size=node_size_skip, color=skipnode_color, 
                fill_alpha=alpha_dic['end_node'], 
                val=len(skipnodes)*['skip'], coords=coords)
        glyph_node_list.extend(skipglyph)        
     
    # source
    p0, sourceglyph = plot_nodes(G, p0, [sourcenode], shape='diamond',
            size=source_size, text_font_size=source_font_size, \
            text_font_style=source_font_style, color=source_color, \
            text_alpha=1.0, label=[source_label], val=[source_label], 
            coords=coords)
    glyph_node_list.extend(sourceglyph)
    print ("p0")

    # add augmented points
    if auglist is not None:
        [ngood, nbad, ngood_aug, nbad_aug, cds_good_aug, cds_bad_aug] = auglist
        # cds can only be assigned to one plot object, so need to copy them!
        cds_good_aug2 = ColumnDataSource(copy.deepcopy(cds_good_aug.data))
        cds_bad_aug2 = ColumnDataSource(copy.deepcopy(cds_bad_aug.data))
        shape_aug_good = get_nodes_glyph(shape='circle', coords=coords)
        shape_aug_bad = get_nodes_glyph(shape='circle', coords=coords)
        glyph_aug_good = p0.add_glyph(cds_good_aug2, shape_aug_good)
        glyph_aug_bad = p0.add_glyph(cds_bad_aug2, shape_aug_bad)         
    # if desired, plot bbox
    if source_bbox is not None:
        source_bbox2 = ColumnDataSource(copy.deepcopy(source_bbox.data))
        bbox_quad_shape, bbox_text_shape = get_bbox_glyph(coords=coords)
        glyph_bbox_shape = p0.add_glyph(source_bbox2, bbox_quad_shape)
        glyph_bbox_text = p0.add_glyph(source_bbox2, bbox_text_shape) 

    # set plot title
    p0.title.text = 'Road network surrounding ' + source_label
    htmlout = os.path.join(htmloutdir, outroot + '_00.html')
    #p0 = add_hover_save(p0, htmlout, show=True)           
    # add hover
    p0 = add_hover_save(p0, htmlout=htmlout, show=True, add_hover=True,
                              renderers=glyph_node_list) 
    
    print ("p1")
    
    # make one more plot with all paths (append to plot p0)
    #ex0, ey0, ex1, ey1, emx, emy, elen, edgelist =\
    ex0, ey0, ex1, ey1, elat0, elon0, elat1, elon1, emx, emy, elen, edgelist =\
            get_path_pos(G, paths, lengths)    
    esource = bk.ColumnDataSource(data=dict(
            ex0=ex0, ey0=ey0,
            ex1=ex1, ey1=ey1,
            elat0=elat0, elon0=elon0,
            elat1=elat1, elon1=elon1,
            emx=emx, emy=emy,
            ealpha = line_alpha*np.ones(len(ex0)), #len(ex0)*[line_alpha]))
            ewidth=np.asarray(len(ex0)*[path_width]),
            ecolor=np.asarray(len(ex0)*[edge_new_color])
        )
    )
    # set routes
    if ecurve_dic is not None:
        seg = get_paths_glyph_line(coords=coords)
        elx0, ely0, ellat0, ellon0 = get_ecurves(edgelist, ecurve_dic)  
        # add to esource
        esource.data['elx0'] = elx0
        esource.data['ely0'] = ely0
        esource.data['ellon0'] = ellon0
        esource.data['ellat0'] = ellat0

    else:
        seg = get_paths_glyph_seg(coords=coords)
    p0.add_glyph(esource, seg)
    htmlout = os.path.join(htmloutdir, outroot + '_01.html')
    p0 = add_hover_save(p0, htmlout, renderers=glyph_node_list, add_hover=True)           
    ###############    

    ###############    
    # Initialize plot if use_hull=False
    if not use_hull:
        # Show growth over time, so don't reinitilize the plot within loop
        plot_node_list=[]
        p, Gnsource, Gesource, node_glyphs, outdict = G_to_bokeh(G, htmlout=None, 
                                                        #ecurve_dic=ecurve_dic,
                                                        global_dic=global_dic,
                                        map_background=map_background, 
                                        show_nodes=True, plot_width=plot_width,
                                        coords=coords)
        # plot skipnodes
        if len(skipnodes) > 0:
            p, skipglyph = plot_nodes(G, p, skipnodes, 
                    size=node_size_skip, color=skipnode_color, 
                    fill_alpha=alpha_dic['end_node'], 
                    val=len(skipnodes)*['skip'], coords=coords)
            plot_node_list.extend(skipglyph)
        # source
        p, sourceglyph = plot_nodes(G, p, [sourcenode], shape='diamond',
                size=source_size, text_font_size=source_font_size, \
                text_font_style=source_font_style, color=source_color, \
                text_alpha=1.0, label=[source_label], val=[source_label],
                coords=coords)
        plot_node_list.extend(sourceglyph)
        # add augmented points
        if auglist is not None:
            [ngood, nbad, ngood_aug, nbad_aug, cds_good_aug, cds_bad_aug] = auglist
            # cds can only be assigned to one plot object, so need to copy them!
            cds_good_aug2 = ColumnDataSource(copy.deepcopy(cds_good_aug.data))
            cds_bad_aug2 = ColumnDataSource(copy.deepcopy(cds_bad_aug.data))
            shape_aug_good = get_nodes_glyph(shape='circle', coords=coords)
            shape_aug_bad = get_nodes_glyph(shape='circle', coords=coords)
            glyph_aug_good = p.add_glyph(cds_good_aug2, shape_aug_good)
            glyph_aug_bad = p.add_glyph(cds_bad_aug2, shape_aug_bad)         
            plot_node_list.extend([glyph_aug_good])
            plot_node_list.extend([glyph_aug_bad])
        # if desired, plot bbox
        if source_bbox is not None:
            source_bbox2 = ColumnDataSource(copy.deepcopy(source_bbox.data))
            bbox_quad_shape, bbox_text_shape = get_bbox_glyph(coords=coords)
            glyph_bbox_shape = p.add_glyph(source_bbox2, bbox_quad_shape)
            glyph_bbox_text = p.add_glyph(source_bbox2, bbox_text_shape) 
            plot_node_list.extend([glyph_bbox_shape])
    ############### 
                    
    # loop through bins
    globc = 0
    nodes_seen_set = set([])
    nodes_seen_list = []
    node_text_list = []
    edges_seen_set = set([])
    edges_seen_list = []
    nlist_hull = []
    nlist_tot = []
    xlist_hull = []
    ylist_hull = []
    # initialize histogram counts (only if source_bbox is not not)
    if source_bbox is not None:
        good_bin, good_cum = np.zeros(len(bins)), np.zeros(len(bins))
        bad_bin, bad_cum = np.zeros(len(bins)), np.zeros(len(bins))
        # set ymaxs for histos
        bad_ymax = 1.1 * np.sum(source_bbox.data['count'][np.where(source_bbox.data['status'] == 'bad')])
        good_ymax = 1.1 * np.sum(source_bbox.data['count'][np.where(source_bbox.data['status'] == 'good')])
        #print ("bad_ymax", bad_ymax
        #print ("good_ymax", good_ymax

    for i,b in enumerate(bins):
        
        print ("\nComputing bin", i+1, "of", len(bins) )     
        htmlout = os.path.join(htmloutdir, outroot + '_' + str(i+10) + '.html')
        new_nodes_set = set([])
        new_nodes_list = []
        new_edges_set = set([])
        new_edges_list = []
        # get indices into lenths and paths
        idxs = np.where(digitized==i)[0]        
        # set total counts of nodes reached
        totc = 0    
        
        # if idxs is empty, continue
        if len(idxs) == 0: 
            if i > 0:
                bad_cum[i] = bad_cum[i-1]
                good_cum[i] = good_cum[i-1]
            continue
                
        if use_hull:
            # reinitizlize plot
            glyph_node_list=[]
            p, Gnsource, Gesource, node_glyphs, outdict = G_to_bokeh(G, htmlout=None, 
                                                global_dic=global_dic,
                                                #ecurve_dic=ecurve_dic,
                                            map_background=map_background, 
                                            show_nodes=True, 
                                            plot_width=plot_width,
                                            coords=coords)
            # plot skipnodes
            if len(skipnodes) > 0:
                p, skipglyph = plot_nodes(G, p, skipnodes, 
                        size=node_size_skip, color=skipnode_color, 
                        fill_alpha=alpha_dic['end_node'], 
                        val=len(skipnodes)*['skip'], coords=coords)
                glyph_node_list.extend(skipglyph)
            # source
            p, sourceglyph = plot_nodes(G, p, [sourcenode], shape='diamond',
                    size=source_size, text_font_size=source_font_size, \
                    text_font_style=source_font_style, color=source_color, \
                    text_alpha=1.0, label=[source_label], val=[source_label],
                    coords=coords)
            glyph_node_list.extend(sourceglyph)

            # add augmented points
            if auglist is not None:
                # cds can only be assigned to one plot object, so need to copy them!
                cds_good_aug2 = ColumnDataSource(copy.deepcopy(cds_good_aug.data))
                cds_bad_aug2 = ColumnDataSource(copy.deepcopy(cds_bad_aug.data))
                shape_aug_good = get_nodes_glyph(shape='circle', coords=coords)
                shape_aug_bad = get_nodes_glyph(shape='circle', coords=coords)
                glyph_aug_good = p.add_glyph(cds_good_aug2, shape_aug_good)
                glyph_aug_bad = p.add_glyph(cds_bad_aug2, shape_aug_bad)         
                glyph_node_list.extend([glyph_aug_bad])
                glyph_node_list.extend([glyph_aug_good])

            # plot hulls          
            nlist_new = np.unique(startnodes[idxs])
            # combine nlist_tot with last nlist to make sure hull encompasses
            # all points this means that the final hull will have all the 
            # points, which is much slower but allows for a more aggressive 
            # alpha parameter.  Plotting with nlist_hull is much faster
            # though agressive (high) alpha parameter can split the hull into
            # multiple polygons
            if i > 0:
                nlist_old = nlist_hull[i-1]
                nlist_plot = np.unique(np.concatenate((nlist_new, nlist_old)))
                nlist_tot = np.unique(np.concatenate((nlist_new, nlist_tot)))
            else:
                nlist_old = []
                nlist_plot = nlist_new
                nlist_tot = nlist_new
            # plot new hull
            #hull_name = 'Hull ' + str(round(b-binsize,2)) + '-' + str(round(b,2)) + 'H'
            hull_name = 'Hull ' + str(round(b-0.5*bin_diff,2)) + '-' \
                + str(round(b+0.5*bin_diff,2)) + 'H'
            
            # much faster to use nlist_plot
            #hulln, hullx, hully = compute_hull(G, nlist_plot, concave=concave, alpha=concave_alpha)
            # more accurate to use nlist_tot
            hulln, hullx, hully = compute_hull(G, nlist_tot, concave=concave, 
                                               concave_alpha=concave_alpha, 
                                               coords=coords)

            nlist_hull.append(hulln)  
            xlist_hull.append(hullx)  
            ylist_hull.append(hully)  
            px, hullglyph = plot_hull(G, p, hulln, hullx, hully, 
                    color=palette[i],#target_color, \
                    fill_alpha=alpha_dic['hull'], \
                    hull_name=hull_name, coords=coords)
            glyph_node_list.extend(hullglyph)

            # get bbox points inside hull                       
            if source_bbox is not None:
                good_indices, good_count, bad_indices, bad_count = \
                    bbox_inside_hull_v1(nlist_new, source_bbox, ignore=[]) 
                # extract subset of data
                print ("good_indices", good_indices)
                source_bbox_good_tmp = keep_bbox_indices(source_bbox, good_indices)
                source_bbox_bad_tmp = keep_bbox_indices(source_bbox, bad_indices)
                # plot good
                bbox_quad_shape, bbox_text_shape = get_bbox_glyph(coords=coords)
                bbox_good_glyph=px.add_glyph(source_bbox_good_tmp, bbox_quad_shape)
                px.add_glyph(source_bbox_good_tmp, bbox_text_shape)
                # plot bad
                bbox_quad_shape, bbox_text_shape = get_bbox_glyph(coords=coords)
                bbox_bad_glyph=px.add_glyph(source_bbox_bad_tmp, bbox_quad_shape)  
                px.add_glyph(source_bbox_bad_tmp, bbox_text_shape) 
                # extend lists
                glyph_node_list.extend([bbox_good_glyph])
                glyph_node_list.extend([bbox_bad_glyph])
 
                #add_hover_save(px, htmlout, show=True)       
                #return
               
                # add histograms
                # update counts
                bad_bin[i] = np.sum(source_bbox.data['count'][bad_indices])
                bad_cum[i] = np.sum(bad_bin)
                good_bin[i] = np.sum(source_bbox.data['count'][good_indices])
                good_cum[i] = np.sum(good_bin)    
                # make plots
                pb0 = plot_histo(bin_str, bad_bin, bad_cum, 
                    title='Adversary Forces Encountered', 
                    plot_width=plot_width, 
                    plot_height=global_dic['histo_height'], 
                    ymax=bad_ymax)
                pb1 = plot_histo(bin_str, good_bin, good_cum, 
                    title='Allied Forces Encountered', 
                    plot_width=plot_width, 
                    plot_height=global_dic['histo_height'],
                    ymax=good_ymax)
                print ("Adversary Forces Encountered", bad_bin[i])
                print ("Allied Forces Encountered", good_bin[i])
            # now add old hull to plot
            if i > 0:
                #nlist_old = np.unique(startnodes[np.where(digitized==i-1)[0]])
                hull_name = 'Hull ' + str(round(b-2*binsize,2)) + '-' + str(round(b-binsize,2)) + 'H'
                px, hullglyphold = plot_hull(G, p, nlist_hull[i-1], xlist_hull[i-1], \
                        ylist_hull[i-1], 
                        color=palette[i-1],#edge_seen_color, \
                        fill_alpha=alpha_dic['hull'],
                        hull_name=hull_name, coords=coords)
                glyph_node_list.extend(hullglyphold)
                
                    
        else:
            
            # plot routes (very slow and not very useful, also probably 
            # not working right now)
            print ("")
            
        # update total count                
        globc += totc        
        # set plot title
        title = 'Timestep ' + bin_str[i] 
                #\str(round(b-binsize,2)) + '-' + str(round(b,2)) + 'H'

        p.title.text = title
        
        if use_hull:
            add_hover=True
        else:
            if i == 0:
                add_hover=True
            else:
                add_hover=False
        if source_bbox is not None:
            px = add_hover_save(px, htmlout=None, add_hover=add_hover,
                                renderers=glyph_node_list) 
            pz = column(px, pb0, pb1)
            #pz = bk.vplot(px, pb0, pb1)
            bk.output_file(htmlout)
            #####################
            #https://github.com/bokeh/bokeh/issues/3671
            # shockingly, bokeh balloons file sizes by default
            bk.reset_output()
            bk.output_file(htmlout)
           #####################            
            bk.save(obj=pz) 
            bk.show(pz) 
            print ("htmlout:", htmlout)
            
        else:
            px = add_hover_save(px, htmlout, add_hover=add_hover, show=True,
                                renderers=glyph_node_list)           

    # one final plot showing all hulls
    if use_hull:
        if concave:
            title = "All Expansion Concave Hulls"
        else:
            title = "All Expansion Convex Hulls"
        htmlout = os.path.join(htmloutdir, outroot + str(20+len(bins)) + '.html')
        # initialize plot
        glyph_node_list = []
        p, Gnsource, Gesource, node_glyphs, outdict = G_to_bokeh(G, htmlout=None, 
                                        #ecurve_dic=ecurve_dic,
                                        global_dic=global_dic,
                                        map_background=map_background, 
                                        show_nodes=True, plot_width=plot_width,
                                        coords=coords)
        # plot skipnodes
        if len(skipnodes) > 0:
            p, gg0 = plot_nodes(G, p, skipnodes, 
                    size=node_size_skip, color=skipnode_color, 
                    fill_alpha=alpha_dic['end_node'], 
                    val=len(skipnodes)*['skip'], coords=coords)
            glyph_node_list.extend(gg0)  
           
        # source
        p, gg1 = plot_nodes(G, p, [sourcenode], shape='diamond',
            size=source_size, text_font_size=source_font_size, \
            text_font_style=source_font_style, color=source_color, \
            text_alpha=1.0, label=[source_label], val=[source_label],
            coords=coords) 
        glyph_node_list.extend(gg1)  
    
        # add augmented points
        if auglist is not None:
            [ngood, nbad, ngood_aug, nbad_aug, cds_good_aug, cds_bad_aug] = auglist
            # cds can only be assigned to one plot object, so need to copy them!
            cds_good_aug2 = ColumnDataSource(copy.deepcopy(cds_good_aug.data))
            cds_bad_aug2 = ColumnDataSource(copy.deepcopy(cds_bad_aug.data))
            shape_aug_good = get_nodes_glyph(shape='circle', coords=coords)
            shape_aug_bad = get_nodes_glyph(shape='circle', coords=coords)
            glyph_aug_good = p.add_glyph(cds_good_aug2, shape_aug_good)
            glyph_aug_bad = p.add_glyph(cds_bad_aug2, shape_aug_bad)  
            glyph_node_list.extend([glyph_aug_good])  
            glyph_node_list.extend([glyph_aug_bad])             
        # if desired, plot bbox
        if source_bbox is not None:
            source_bbox2 = ColumnDataSource(copy.deepcopy(source_bbox.data))
            bbox_quad_shape, bbox_text_shape = get_bbox_glyph(coords=coords)
            glyph_bbox_shape = p.add_glyph(source_bbox2, bbox_quad_shape)
            glyph_bbox_text = p.add_glyph(source_bbox2, bbox_text_shape) 
            glyph_node_list.extend([glyph_bbox_shape])  
            glyph_node_list.extend([glyph_bbox_text])  
            
        # plot new hulls
        for i, (nlisti, xlisti, ylisti) in \
                            enumerate(zip(nlist_hull, xlist_hull, ylist_hull)):
            hull_name = 'Hull ' + str(round(bins[i]-binsize,2)) + '-' + str(round(bins[i],2)) + 'H'
            # plot new hull
            p, gg6 = plot_hull(G, p, nlisti, xlisti, ylisti, 
                        color=palette[i],#target_color, \
                        fill_alpha=alpha_dic['hull']/3, \
                        hull_name=hull_name, coords=coords)
            glyph_node_list.extend(gg6)  

        # get bbox points inside hull, and histograms                       
        if source_bbox is not None:
#            # plot bbox nodes
#            p = plot_bbox(G, p, source_bbox, \
#                gmap_background=gmap_background, htmlout=None) 
            # add histograms
            pb0 = plot_histo(bin_str, bad_bin, bad_cum, \
                title='Adversary Forces Encountered', \
                plot_width=plot_width, plot_height=global_dic['histo_height'], 
                ymax=bad_ymax)
            pb1 = plot_histo(bin_str, good_bin, good_cum, \
                title='Allied Forces Encountered', \
                plot_width=plot_width, plot_height=global_dic['histo_height'],
                 ymax=good_ymax)
            p.title.text = title
            p = add_hover_save(p, htmlout=None, add_hover=True) 
            pzf = column(p, pb0, pb1)
            #pzf = bk.vplot(p, pb0, pb1)
            bk.output_file(htmlout)
            #####################
            #https://github.com/bokeh/bokeh/issues/3671
            # shockingly, bokeh balloons file sizes by default
            #bk.reset_output()
            #####################
            bk.save(obj=pzf) 
            bk.show(pzf)
        else: 
            p.title.text = title    
            px = add_hover_save(p, htmlout, show=True, 
                                renderers=glyph_node_list)   

    print ("Time to compute spread from", sourcenode, time.time()-t0, "seconds")
    
    return p           


###############################################################################
### main
###############################################################################
def main():
    
    # Instead, use networkx_osm_run.py!!
    
    #For querie use: http://www.the-di-lab.com/polygon/ 
    webbrowser.open('http://www.the-di-lab.com/polygon/')
    
    indir = '/Users/avanetten/Documents/osm/'

    # Set Variables
    # smallest road type   
    min_road_type = 'tertiary' 
        #min_road_type = 'secondary'   
    # output root
    root = 'osm_' + min_road_type + '_demo'   
    # blank or gmap background
    map_background = 'None' #OSM'
    # max distance a datapoint can be from osm node
    max_node_dist_km = 5.0    
    # plot size
    plot_width = 1200    
    # switch to download a new map
    download_new_map = False

    # Files
    # input files
    bbox_infile = indir + 'bbox_feb2015_global.xlsx' 
    # output files
    outdir = indir + root + '_bg=' + map_background.lower()
    #if gmap_background:
    #    outdir =  indir + root + '_gmap/'
    #else:
    #    outdir = indir + root + '_no_gmap/'  
    if not os.path.exists(outdir):
            os.mkdir(outdir)
    queryfile =  outdir + root + '_poly.query'
    osmfile = outdir + root + '.osm'
    filenames = [bbox_infile, queryfile, osmfile, outdir]
    graph_out = outdir + root + '.graphml'
    csvout =  outdir + root + '.csv'
    htmlout0 = outdir + root  + '_raw.html'
    htmlout = outdir + root  + '.html'
    
    ###################
    construct_poly_query(poly, queryfile)
    ###################    
     
    ################### 
    # download osmfile with query
    t0 = time.time()
    download_osm_query(queryfile, osmfile)
    t1 = time.time()
    print ("Time to download graph =", t1-t0, "seconds")

    # create osm, dictionaries
    osm = graph_init.OSM(osmfile)  # extremely fast
    node_histogram, ntype_dic, X_dic = graph_init.intersect_dic(osm) 
    
    ###################
    # full graph
    t1 = time.time()
    graph_out = outdir + root + '_full.graphml'
    csvout =  outdir + root + '_full.csv'
    G0 = graph_init.read_osm(osm)
    t2 = time.time()
    print ("Time to create graph =", t2-t1, "seconds")
    # print to csv
    # G_to_csv(G0, csvout)
    ###################

    ###################
    # refined graph
    #test_refine_osm(osmfile)
    t2 = time.time()
    graph_out = outdir + root + '.graphml'
    csvout =  outdir + root + '.csv'
    G, ecurve_dic = graph_init.refine_osm(osm, node_histogram, ntype_dic, X_dic)#(osmfile)
    t3 = time.time()
    print ("Time to create refined graph =", t3-t2, "seconds")
    # print to csv
    # G_to_csv(G, csvout)
    ###################
    
    ################### 
    # crate kdtree of nodes
    kd_idx_dic, kdtree = G_to_kdtree(G)   
    # Time to create k-d tree: 0.569053888321 seconds
    ################### 

    ###################
    # can skip everything between ##^## above with:
    osm, node_histogram, ntype_dic, X_dic, G0, G, ecurve_dic, kd_idx_dic, kdtree = \
                get_osm_data(poly, filenames, \
                map_background, download_osm=download_new_map ) 

    # load bbox:
    dfbbox, g_node_props_dic, source_bbox = \
                load_bbox(bbox_infile, kdtree, kd_idx_dic, \
                max_dist_km=max_node_dist_km)

    ###################
    # initialize plot
    p0, source0, esource0, node_glyphs, outdict = G_to_bokeh(G0, htmlout=htmlout0, 
                            global_dic=global_dic,
                            map_background=map_background, \
            show_nodes=False, plot_width=plot_width, coords=coords)
        
    p, source, esource, node_glyphs, outdict = G_to_bokeh(G, htmlout=htmlout, 
            global_dic=global_dic,
            map_background=map_background, \
            show_nodes=True, show_road_labels=False, verbose=True, \
            plot_width=plot_width, coords=coords)
    ###################

    ###################
    # tests

    ## test0: points
    #test0_html = outdir+root+'_test0.html'
    #test_points = ['3098693716', '3175464525']
    #pt = plot_nodes(G, p, test_points, gmap_background=gmap_background, \
    #        size=15, text_font_size='10pt', color='green', \
    #        text_alpha=0.9, labels=test_points)
    #add_hover_save(pt, test0_html, gmap_background=gmap_background, show=True) 

    # test1
    ## plot nodes of interest:
    ##NetworkXNoPath: node 3490396197 not reachable from 3065079347
    #p = plot_nodes(G0, p, ['3490396197', '3065079347'], gmap_background=gmap_background, \
    #            size=20, color='red', fill_alpha=0.3)  
    #add_hover_save(p, htmlout, gmap_background=gmap_background, show=True)  

    # test2
    #t0 = time.time()
    #source = '2387911068'
    #lengths,paths = nx.single_source_dijkstra(G, source, weight='Travel Time (h)') 
    #print ("Time to compute all paths from", source, time.time()-t0, "seconds"
    # add_hover_save(p, htmlout, gmap_background=gmap_background, show=True)        
    ###################

    ###################
    # plot bbox
    pg, source, esource, node_glyphs, outdict = G_to_bokeh(G, htmlout=None, 
            global_dic=global_dic,
            map_background=map_background, \
            show_nodes=True, show_road_labels=False, verbose=False, \
            plot_width=plot_width, coords=coords)
    htmlbbox = outdir + root + '_bbox.html' # None
    pg = plot_bbox(G, pg, source_bbox, \
        map_background=map_background, htmlout=htmlbbox)          
    ###################

    ###################
    # run analytics
    ####################
    
    #######
    # test: remove 1000 random nodes from the graph, and then compute paths
    #idxs = np.random.random_integers(low=0, high=len(G.nodes())-1, size=1000)
    #ngood = np.asarray(G.nodes())[idxs]
    #######

    ####################
    # compute and plot augmented points within r_m of goodnodes and badnodes
    r_m = 300     # distance to extrapolate control values
    plot_aug_html = outdir + root + '_aug.html' #None
    # get augmented points
    auglist = get_aug(G, dfbbox, kdtree, kd_idx_dic, r_m=r_m) 
    # make plots
    p0, source, esource, node_glyphs, outdict = G_to_bokeh(G, htmlout=None, 
            global_dic=global_dic,
            map_background=map_background, \
            show_nodes=True, show_road_labels=False, verbose=False, \
            plot_width=plot_width, coords=coords)
    p1 = plot_bbox(G, p0, source_bbox, \
        map_background=map_background, htmlout=None)  
    p2 = plot_aug(G, p1, auglist, htmlout=plot_aug_html, map_background=map_background)
    ####################   

    ####################
    # compute minimum spanning tree between nodes 
    node_set_name = 'goodnodes'  # 'badnodes'
    r_m = 300.     # distance to extrapolate control values
    outroot = outdir + root + '_' + node_set_name + '_mst_paths'
    # get augmented points
    auglist = get_aug(G, dfbbox, kdtree, kd_idx_dic, r_m=r_m) 
    # plots
    show_routes_all(outroot, G, dfbbox, source_bbox, auglist, \
                        map_background=map_background, \
                        plot_width=plot_width, node_set_name=node_set_name, \
                        target=None)                
    ####################
                        
    ####################
    # compute evacuation routes between nodes and target
    evac_direction = 'east'
    node_set_name = 'goodnodes'  
    r_m = 300.     # distance to extrapolate control values
    outroot = outdir + root + '_' + node_set_name + '_evac_paths_' + evac_direction
    # get augmented points
    auglist = get_aug(G, dfbbox, kdtree, kd_idx_dic, r_m=r_m) 
    # expand auglist 
    [ngood, nbad, ngood_aug, nbad_aug, ngood_aug_nid, nbad_aug_nid, \
                ngood_aug_count, nbad_aug_count, ngood_aug_val, nbad_aug_val] = \
                auglist    
    # define which nodes to skip for target selection
    target_skiplists = [ngood, nbad, ngood_aug, nbad_aug]
    target_evac = choose_target(G, skiplists=target_skiplists, \
                                            direction=evac_direction)
    print ("Target:", target_evac)
    # compute evac paths
    show_routes_all(outroot, G, dfbbox, source_bbox, auglist, \
                        map_background=map_background, \
                        plot_width=plot_width, node_set_name=node_set_name, \
                        target=target_evac)
    ####################
    
    
    ####################
    # compute travel_time
    reload(networkx_osm_ops)
    target = '2387911068'
    tmp_dfbbox, tmp_source_bbox = dfbbox, source_bbox  #None, None 
    ######
    # optional: choose random target
    #target_idx = np.random.randint(0, len(G.nodes()))
    #target = G.nodes()[target_idx]
    ######
    # get augmented points, omit target from augmenting
    auglist = get_aug(G, dfbbox, kdtree, kd_idx_dic, r_m=r_m, \
        special_nodes=set([target])) 
    # expand auglist 
    [ngood, nbad, ngood_aug, nbad_aug, ngood_aug_nid, nbad_aug_nid, \
                ngood_aug_count, nbad_aug_count, ngood_aug_val, nbad_aug_val] = \
                auglist    
    print ("Compute Travel time to target:", target)
    htmloutdir = os.path.join(outdir, target + '_travel_time/')
    sources = nbad
    skipnodes = ngood_aug
    p = compute_travel_time(G, target, sources, \
        g_node_props_dic, skipnodes=skipnodes, htmloutdir=htmloutdir, \
        map_background=map_background, binsize=0.25, plot_width=plot_width, \
        dfbbox=tmp_dfbbox, source_bbox=tmp_source_bbox)
    ####################
    
    ####################
    
    #################### 
    # compute spread
    reload(networkx_osm_ops)
    source = '2387911068'
    tmp_dfbbox, tmp_source_bbox = dfbbox, source_bbox  #None, None 
    ######
    # optional: choose random source
    #source_idx = np.random.randint(0, len(G.nodes()))
    #source = G.nodes()[source_idx]
    ######
    print ("Compute spread from source:", source)
    htmloutdir = os.path.join(outdir,  source + '_spread/')
    p = compute_spread(G, source, htmloutdir=htmloutdir, \
        map_background=map_background, plot_width=plot_width, \
        dfbbox=tmp_dfbbox, source_bbox=tmp_source_bbox)
    #Time to compute paths: 0.0421640872955 seconds
    #Time to compute spread from 2387911068 26.1249129772 seconds
    ####################
    
    #t4 = time.time()
    #nx.write_graphml(G, outfile)#, encoding='unicode')
    #print ("query:", queryfile, "osmfile:", osmfile, "outfile:", outfile
    #print ("Time to write graph =", time.time() - t4, "seconds"

