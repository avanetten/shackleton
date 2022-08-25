#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 21 20:39:34 2017

@author: avanetten
"""

import bokeh.plotting as bk
import numpy as np
import time
import sys
import os

from bokeh.plotting import figure
from bokeh.models import (
    Plot, GMapPlot, GMapOptions, Range1d, LinearAxis, Grid, DataRange1d,
    PanTool, WheelZoomTool, BoxZoomTool, ResetTool, BoxSelectTool, \
    HoverTool, #ResizeTool, 
    SaveTool, Legend, Axis, CategoricalAxis, 
    UndoTool, RedoTool,
    CategoricalTicker, FactorRange,
    WMTSTileSource, BBoxTileSource)#, GeoJSOptions, GeoJSPlot)
from bokeh.models.glyphs import (Square, Circle, Triangle, InvertedTriangle, 
                    Segment, Patch, Diamond, Text, Rect, Patches, MultiLine,
                    Quad)
from bokeh.tile_providers import STAMEN_TONER, STAMEN_TERRAIN, CARTODBPOSITRON_RETINA

# local files
from utils import (distance, distance_euclid, latlon_to_wmp, \
                   lin_scale, log_scale, \
                   log_transform, value_by_key_prefix, query_kd, \
                   query_kd_ball, global_vars)


###############################################################################
### core
###############################################################################
def G_to_nsource(G, nsize_key='deg', constant_node_size=None,
                 ncolor_key='ntype', verbose=False):
    """Get columndatasource for nodes from G"""

    color_dic, alpha_dic = define_colors_alphas()

    # node proprties
    size_mult = 1.0  # 1.5               # node size multiplier

    numN = len(G.nodes())
    # initialize node arrays
    nodex = np.zeros(numN)
    nodey = np.zeros(numN)
    nodelat = np.zeros(numN)
    nodelon = np.zeros(numN)
    nxwmp = np.zeros(numN)
    nywmp = np.zeros(numN)
    nsize = np.zeros(numN) 
    ndeg = np.zeros(numN) 
    nalpha = np.zeros(numN)
    ncolor = numN*['']  # np.chararray(numN)#numN*['']
    nid = numN*['']#np.chararray(numN)#numN*[''] #[str(i) for i in np.arange(numN)]
    nname = numN*['']#np.chararray(numN)#numN*[''] #[str(i) for i in np.arange(numN)]
    ncount = numN*[''] # np.chararray(numN)#
    nval = numN*[''] # np.chararray(numN)#
    
    # populate node arrays
    t1 = time.time()
    nlist = np.sort(list(G.nodes()))
    for i, n in enumerate(nlist):
        n_props = G.nodes[n]
        # set degree
        deg = G.degree[n]
        n_props['deg'] = deg
        # set type
        if deg == 1:
            node_type = 'endpoint'
        elif deg == 2:
            node_type = 'midpoint'
        else:
            node_type = 'intersection'
        n_props[ncolor_key] = node_type
        lat, lon = n_props['lat'], n_props['lon']

        # add web mercator projection coords (should already be in n_props!)
        if 'x' not in n_props.keys():
            x_wmp, y_wmp = latlon_to_wmp(lat, lon)
            nxwmp[i] = x_wmp
            nywmp[i] = y_wmp
            # at x_wmp and y_wmp to G props
            G.nodes[n]['x'] = x_wmp
            G.nodes[n]['y'] = y_wmp

        # ntype = n_props['ntype']
        # deg = n_props['deg']
        nid[i] = n
        nname[i] = str(n)

        # set positions
        nodex[i] = n_props['x']  # lon
        nodey[i] = n_props['y']  # lat

        # set size, color
        if constant_node_size is not None:
            nsize[i] = constant_node_size
        else:
            ndeg[i] = n_props['deg']
            size_val = n_props[nsize_key]
            nsize[i] = size_mult * float(size_val)
        # print ("nsize[i]:", nsize[i])
        cdic_key = n_props[ncolor_key]
        if type(cdic_key) == list:
            cdic_key = cdic_key[0]
        ncolor[i] = color_dic[cdic_key]
        # print ("i:", i, "n_props", n_props
        # print ("size_val", size_val, "cdic_key", cdic_key
        # print ("ncolor_dic[cdic_key]", ncolor_dic[cdic_key]
        nalpha[i] = alpha_dic['osm_node']
        if (i % 1000) == 0 and verbose:
            print(i, "node:", n, "lat, lon:", lat, lon)

    if verbose:
        print("Time to create node arrays:", time.time() - t1, "seconds")

    nsource = set_nodes_source(G, nid, size=nsize, color=np.asarray(ncolor),
                               fill_alpha=nalpha, shape='circle', label=[],
                               count=np.asarray(ncount), val=np.asarray(nval),
                               name=nname)
    return nsource


###############################################################################
def G_to_esource(G, use_multiline=True,
                 ewidth_key='inferred_speed_mph',  # 'Num Lanes'
                 ecolor_key='inferred_speed_mph',
                 ecolor_key2='speed2',
                 cong_key='congestion',
                 width_mult=1.0/8,  # line width multiplier
                 verbose=False):
    """
    Create edge columndatasource from graph G
    """

    color_dic, alpha_dic = define_colors_alphas()
    t1 = time.time()

    if use_multiline:
        numLines = len(G.edges())
        ex0 = np.zeros(numLines)
        ex1 = np.zeros(numLines)
        ey0 = np.zeros(numLines)
        ey1 = np.zeros(numLines)
        elat0 = np.zeros(numLines)
        elat1 = np.zeros(numLines)
        elon0 = np.zeros(numLines)
        elon1 = np.zeros(numLines)
        emx = np.zeros(numLines)
        emy = np.zeros(numLines)
        ewidth = np.zeros(numLines)
        ealpha = np.zeros(numLines)
        eid = numLines*['']
        elx0 = []
        ely0 = []
        lengths = np.zeros(numLines)
        numLanes = np.zeros(numLines)
        speeds = np.zeros(numLines)
        speeds2 = np.zeros(numLines)
        congestion = np.zeros(numLines)
        cong_color = numLines*['']
        names = numLines*['']
        uvs = numLines*['']
        raw_color = numLines*['']
        ecolor = numLines*['']
        ecolor2 = numLines*['']
        plot_color = numLines*['']

        for j, (s, t, data) in enumerate(G.edges(data=True)):
            # print ("s,t,data:", s,t,data)
            uvs[j] = (s, t)
            if 'geometry' in data.keys():
                geom = data['geometry']
                npoints = len(list(geom.coords))
            else:
                print("missing geom")
                return

            # this just recreates graph_utils.make_ecurve_dic()
            geom = data['geometry']
            line_coords = np.array(list(geom.coords))
            elx0.append(line_coords[:, 0])
            ely0.append(line_coords[:, 1])

            # set width
            # print("data[ewidth_key]:", data[ewidth_key])
            if ewidth_key not in data.keys():
                data[ewidth_key] = 1
            if type(data[ewidth_key]) == list:
                data[ewidth_key] = int(max(data[ewidth_key]))
            # print ("e_props:", data)
            width_val = data[ewidth_key]
            # print ("  width_val:", width_val)
            ewidth[j] = width_mult * float(width_val)
            # print("  ewidth[j]:", ewidth[j])

            lengths[j] = data['length']
            # print (data)
            if 'lanes' in data.keys():
                ztmp = data['lanes']
                if type(ztmp) == list:
                    numLanes[j] = ztmp[0]
                else:
                    numLanes[j] = ztmp
            else:
                numLanes[j] = 1
            speeds[j] = data['inferred_speed_mph']
            if 'speed2' in data.keys():
                speeds2[j] = data['speed2']
            else:
                speeds2[j] = data['inferred_speed_mph']
            if 'name' in data.keys():
                names[j] = data['name']
            else:
                names[j] = str(j)

            # test if it's a bridge or not
            if 'Bridge' in data.keys():
                ecolor[j] = color_dic['bridge']  # bridge_color
                ewidth[j] = 2*width_val
            else:
                cdic_key = data[ecolor_key]
                # print ("type cdic_key:", type(cdic_key))
                if type(cdic_key) == list:
                    cdic_key = cdic_key[0]
                elif isinstance(cdic_key, (int, float, np.int64, np.float)):
                    # print ("cdic_key:",cdic_key)
                    # round to neareast 5
                    cdic_key_round = 5 * round(cdic_key / 5)
                    # print ("color_dic[cdic_key_round]:",
                    #   color_dic[cdic_key_round])
                    ecolor[j] = color_dic[cdic_key_round]
                else:
                    ecolor[j] = value_by_key_prefix(color_dic, cdic_key)

            # assume speed2 is just 'speed2'
            speed2_round = 5 * round(speeds2[j] / 5)
            ecolor2[j] = color_dic[speed2_round]

            # set alpha
            ealpha[j] = alpha_dic['osm_edge']

            # node properties
            s_props = G.nodes[s]
            t_props = G.nodes[t]
            slat, slon = s_props['lat'], s_props['lon']
            tlat, tlon = t_props['lat'], t_props['lon']
            # sx, sy = latlon_to_wmp(slat, slon)
            # tx, ty = latlon_to_wmp(tlat, tlon)
            sx, sy = s_props['x'], s_props['y']
            tx, ty = t_props['x'], t_props['y']
            ex0[j], ey0[j] = sx, sy
            ex1[j], ey1[j] = tx, ty
            emx[j], emy[j] = 0.5*(sx + tx), 0.5*(sy+ty)
            elon0[j], elat0[j] = slon, slat
            elon1[j], elat1[j] = tlon, tlat

            if type(data['osmid']) == list:
                eid[j] = data['osmid']  # + [ktmp]
            else:
                eid[j] = data['osmid']  # + ktmp/10

            # congestion initally
            if cong_key in data.keys():
                congestion[j] = data[cong_key]
            else:
                congestion[j] = 1
            # congestion color
            # round to nearest 0.2
            cong_round = 0.2 * round(congestion[j] / 0.2)
            cong_round_str = ("{:.1f}".format(cong_round))
            cong_color[j] = color_dic[cong_round_str]
            raw_color[j] = color_dic['raw_edge']
            # plot_color[j] = 

    # This is the older version with segments.  Multiline is much better, as
    #  it supports hover, and doesn't flicker when the plot is panned!
    else:
        # determine number of segments
        numSegs = 0
        for u, v, data in G.edges(data=True):
            # print ("u,v,data:", u,v,data)
            if 'geometry' in data.keys():
                geom = data['geometry']
                npoints = len(list(geom.coords))
                numSegs += npoints - 1
            else:
                numSegs += 1
                print("missing geom")
                break
        if verbose:
            print("G numSegs:", numSegs)

        ex0 = np.zeros(numSegs)
        ex1 = np.zeros(numSegs)
        ey0 = np.zeros(numSegs)
        ey1 = np.zeros(numSegs)
        elat0 = np.zeros(numSegs)
        elat1 = np.zeros(numSegs)
        elon0 = np.zeros(numSegs)
        elon1 = np.zeros(numSegs)
        emx = np.zeros(numSegs)
        emy = np.zeros(numSegs)
        ewidth = np.zeros(numSegs)
        ealpha = np.zeros(numSegs)
        ecolor = numSegs*['']
        eid = numSegs*['']
        # elx0 = []
        # ely0 = []
        elx0 = [[] for i in range(numSegs)]
        ely0 = [[] for i in range(numSegs)]
        # ellat0 = [[] for i in range(numSegs)] #(numE)]
        # ellon0 = [[] for i in range(numSegs)] #(numE)]
        lengths = np.zeros(numSegs)
        numLanes = np.zeros(numSegs)
        speeds = np.zeros(numSegs)
        speeds2 = np.zeros(numSegs)
        names = numSegs*['']
        uvs = numSegs*['']
        congestion = np.zeros(numLines)
        cong_color = numLines*['']
        raw_color = numLines*['']
        plot_color = numLines*['']
        j = 0
        for s, t, data in G.edges(data=True):
            # print ("u,v,data:", u,v,data)
            if 'geometry' in data.keys():
                geom = data['geometry']
                npoints = len(list(geom.coords))
            else:
                print("missing geom")
                break

            # # this just recreates graph_utils.make_ecurve_dic()
            # geom = data['geometry']
            # line_coords = np.array(list(geom.coords))
            # elx0.append([line_coords[:,0]])
            # ely0.append([line_coords[:,1]])

            # set width
            if ewidth_key not in data.keys():
                data[ewidth_key] = 1
            if type(data[ewidth_key]) == list:
                data[ewidth_key] = int(max(data[ewidth_key]))
            # print ("e_props:", data)

            # node properties
            s_props = G.nodes[s]
            t_props = G.nodes[t]
            slat, slon = s_props['lat'], s_props['lon']
            tlat, tlon = t_props['lat'], t_props['lon']
            # sx, sy = latlon_to_wmp(slat, slon)
            # tx, ty = latlon_to_wmp(tlat, tlon)
            sx, sy = s_props['x'], s_props['y']
            tx, ty = t_props['x'], t_props['y']

            # populate arrays
            # if the edge has a geometry tag, get all the points
            if 'geometry' in list(data.keys()):
                geom = data['geometry']
                points = list(geom.coords)
                # print ("points:", points)

                for ktmp in range(0, len(points)-1):
                    (p0_x, p0_y) = points[ktmp]
                    (p1_x, p1_y) = points[ktmp+1]
                    # add to lists
                    ex0[j], ey0[j] = p0_x, p0_y
                    ex1[j], ey1[j] = p1_x, p1_y
                    emx[j], emy[j] = 0.5*(p0_x + p1_x), 0.5*(p0_y + p1_y)
                    if type(data['osmid']) == list:
                        eid[j] = data['osmid'] + [ktmp]
                    else:
                        eid[j] = data['osmid'] + ktmp/10
                    # set width, color
                    width_val = data[ewidth_key]
                    # print ("width_val:", width_val)
                    ewidth[j] = width_mult * float(width_val)
                    # test if it's a bridge or not
                    if 'Bridge' in data.keys():
                        ecolor[j] = color_dic['bridge']  # bridge_color
                        ewidth[j] = 3*width_val
                    else:
                        cdic_key = data[ecolor_key]
                        if type(cdic_key) == list:
                            cdic_key = cdic_key[0]
                        # print ("cdic_key:",cdic_key)
                        ecolor[j] = value_by_key_prefix(color_dic, cdic_key)
                    # set alpha
                    ealpha[j] = alpha_dic['osm_edge']
                    j += 1

            else:
                ex0[j], ey0[j] = sx, sy
                ex1[j], ey1[j] = tx, ty
                emx[j], emy[j] = 0.5*(sx + tx), 0.5*(sy+ty)
                elon0[j], elat0[j] = slon, slat
                elon1[j], elat1[j] = tlon, tlat
                eid[j] = data['osmid']
                # set width, color
                width_val = data[ewidth_key]
                # print ("width_val:", width_val)
                ewidth[j] = width_mult * float(width_val)
                # test if it's a bridge or not!
                if 'Bridge' in data.keys():
                    ecolor[j] = color_dic['bridge']  # bridge_color
                    ewidth[j] = 3*width_val
                else:
                    cdic_key = data[ecolor_key]
                    ecolor[j] = value_by_key_prefix(color_dic, cdic_key)
                # set alpha
                ealpha[j] = alpha_dic['osm_edge']
                j += 1

    if verbose:
        print("Time to create edge arrays:", time.time() - t1, "seconds")

    esource = bk.ColumnDataSource(
        data=dict(
            uv=uvs,
            elat0=elat0, elon0=elon0,
            elat1=elat1, elon1=elon1,
            ex0=ex0, ey0=ey0,
            ex1=ex1, ey1=ey1,
            emx=emx, emy=emy,
            ewidth=ewidth,
            ealpha=ealpha,
            eid=eid,
            ecolor=np.asarray(ecolor),
            elabel=eid,
            elx0=elx0,
            ely0=ely0,
            name=names,
            length=lengths,
            lanes=numLanes,
            speed_mph=speeds,
            speed2=speeds2,
            congestion=congestion,
            cong_color=cong_color,
            raw_color=raw_color,
            plot_color=plot_color,
            ecolor2=np.asarray(ecolor2),
        )
    )

    return esource


###############################################################################
def update_esource(esource, update_dict, speed_key1='inferred_speed_mph',
                   speed_key2='speed2', edge_id_key='uv',
                   congestion_key='congestion',
                   # ecolor_key='speed2',
                   ecolor_key='ecolor2',
                   travel_time_key='Travel Time (h)',
                   verbose=False):
    """Update esource"""

    color_dic, alpha_dic = define_colors_alphas()
    update_keys = set(list(update_dict.keys()))

    # update edge cds
    keys = list(esource.data.keys())
    N = len(esource.data[keys[0]])
    for j in range(N):
        (u, v) = esource.data[edge_id_key][j]
        print("u, v:", u, v)
        if (u, v) in update_keys or (v, u) in update_keys:
            try:
                frac = update_dict[(u, v)]
            except KeyError:
                frac = update_dict[(v, u)]

            if verbose:
                print(j, "u, v, frac:", u, v, frac)

            # set congestion
            esource.data[congestion_key] = frac
            # set congestion color

            # update speed
            speed1 = esource.data[speed_key1][j]
            speed2 = speed1 * frac
            esource.data[speed_key2][j] = speed2
            # set speed color
            cdic_key = speed2
            cdic_key_round = 5 * round(cdic_key / 5)
            color_out = color_dic[cdic_key_round]
            esource.data[ecolor_key][j] = color_out

    return esource


###############################################################################        
def plot_nodes(G, plot, nodes, size=6, color='red', \
                    fill_alpha=0.6, shape='circle', label=[], count=[], val=[],
                    name=[], text_alpha=0.9, text_font_size='5pt', 
                    text_font_style='normal', text_color='black',
                    coords='latlon'):
                
    '''Combine a few functions to add nodes to plot'''


    # get columndatasource
    source = set_nodes_source(G, nodes, size=size, color=color, 
                fill_alpha=fill_alpha, shape=shape, label=label, count=count, 
                val=val, name=name)
                
    # get nodes
    marker = get_nodes_glyph(shape=shape, coords=coords)
    
    # get labels
    text = get_nodes_labels_glyph(text_alpha=text_alpha, 
                                  text_font_size=text_font_size, 
                                  text_font_style=text_font_style, 
                                  text_color=text_color, coords=coords) 
    
    # add to plot
    glyph_shape = plot.add_glyph(source, marker)
    glyph_text = plot.add_glyph(source, text) 
    glyph_node_list = [glyph_shape]
    
    return plot, glyph_node_list


###############################################################################
def cds_to_bokeh(nsource, esource, plot_in=None, htmlout=None,
               map_background='None',
               ecolor_key='inferred_speed_mph',
               show_nodes=True, show_road_labels=False,
               show=False, plot_width=1200,  title='None', add_glyphs=True,
               coords='wmp', use_multiline=True, add_hover=True,
               lod_threshold=None, # http://bokeh.pydata.org/en/latest/docs/user_guide/tools.html#controlling-level-of-detail
               show_line_hover=True, verbose=False):
    '''
    Create a bokeh plot of the network G.
    if plot_in: update existing plot
    Performance could certainly be improved.
    Try splitting edge arrays based on color, then plotting each color
    individually and adding that color to the legend
    coords can be 'latlon' or 'wmp', assume coords are in wmp
    if lod_threshold = None, displacy all lines at all times
    !! This function could be sped up marketly by using ecurve_dic and
       removing some arrays that are extraneous !!
    '''

    out_dict = {}
    glyph_node_list = []
    plot_glyph_list = []
    color_dic, alpha_dic = define_colors_alphas()
    t0 = time.time()

    # node properties
    nodex = nsource.data['x']
    nodey = nsource.data['y']
    # find x, y extent of plot (lat,lon)
    x_mid = 0.5*(max(nodex) + min(nodex))
    y_mid = 0.5*(max(nodey) + min(nodey))
    if verbose:
        print("x_mid:", x_mid, "y_mid:", y_mid)

    # get min, max for plotting
    # xmintmp, ymintmp = latlon_to_wmp(minlat, minlon)
    # xmaxtmp, ymaxtmp = latlon_to_wmp(maxlat, maxlon)
    if coords == 'latlon':
        minlat, maxlon = min(nodey), max(nodey)
        minlon, maxlat = min(nodex), max(nodex)
        xmin, xmax, ymin, ymax = minlon, maxlon, minlat, maxlat
    elif coords == 'wmp':
        xmin, xmax, ymin, ymax = min(nodex), max(nodex), min(nodey), max(nodey)
    else:
        print("Unknown coords in G_to_bokeh()")
        return
    if verbose:
        print("Plotting limits:", xmin, xmax, ymin, ymax)

    # plot
    ########################
    if map_background.upper() == 'GMAP':
        print("Create GMAP...")

        # Now need an api key
        # see: https://github.com/bokeh/bokeh/blob/0.12.0/examples/models/maps.py#L19-L27
        # also see: https://developers.google.com/maps/documentation/javascript/get-api-key
        api_key = '???'

        # try importing a gmap
        # https://github.com/bokeh/bokeh/blob/master/examples/glyphs/maps.py
        # https://github.com/bokeh/bokeh/blob/master/examples/glyphs/trail.py

        # set map height, options
        plot_height = int(0.75 * plot_width)
        map_options = GMapOptions(lat=y_mid, lng=x_mid, zoom=9)
        # map_options = GMapOptions(lat=y_mid, lng=x_mid, zoom=mapzoom)
        #geojs_map_options = GeoJSOptions(lat=y_mid, lng=x_mid, zoom=mapzoom)
        x_range = Range1d()
        y_range = Range1d()
        plot = GMapPlot(x_range=x_range, y_range=y_range, 
                        map_options=map_options, 
                        plot_height=plot_height, plot_width=plot_width, 
                        #title=title,
                        api_key=api_key)  # set title later 
        plot.title.text = title
        plot.map_options.map_type='hybrid'
#        plot = GeoJSPlot(x_range=x_range, y_range=y_range, 
#                        map_options=geojs_map_options, 
#                        plot_height=plot_height, plot_width=plot_width, 
#                        title=title)  # set title later 
        
        # !! Reset Tool goes to lat, lon = 0,0 !! 
        plot.add_tools(PanTool(), WheelZoomTool(), BoxZoomTool(), \
                SaveTool(), )#, ResetTool())   

    #################
    #if not gmap_background:
    else:
        # list of map servers: https://leaflet-extras.github.io/leaflet-providers/preview/
        # plot (no background)    
        # set plot width and height such that lat lon degrees are equivalent
        # compute distances along x and y midpoints
        if coords == 'wmp':
            distx = distance_euclid(y_mid, min(nodex), y_mid, max(nodex))
            disty = distance_euclid(min(nodey), x_mid, max(nodey), x_mid)
        elif coords == 'latlon':
            distx = distance(y_mid, min(nodex), y_mid, max(nodex))
            disty = distance(min(nodey), x_mid, max(nodey), x_mid)
        plot_height = int(float(plot_width * disty/distx)) 
        if verbose:
            print ("distx:", distx, "disty:", disty)            
        # old, crummy version
        #x_extent = max(nodex) - min(nodex)
        #y_extent = max(nodey) - min(nodey)
        #plot_height = int(float(plot_width) * (y_extent/x_extent))
        # keep it square>
        #plot_height = plot_width

        #x_range = DataRange1d()#sources=[source.columns("nodex")])
        #y_range = DataRange1d()#sources=[source.columns("nodey")])
        # allow some buffer around plot
        buffbuff = 0.02
        dx = max(nodex) - min(nodex)
        dy = max(nodey) - min(nodey)
        x0, x1 = min(nodex) - buffbuff*dx, max(nodex) + buffbuff*dx
        y0, y1 = min(nodey) - buffbuff*dy, max(nodey) + buffbuff*dy
        x_range = (x0, x1)
        y_range = (y0, y1)
        # x_range = (min(nodex), max(nodex))
        # y_range = (min(nodey), max(nodey))
        if verbose:
            print ("x_range:", x_range)
            print ("y_range:", y_range)
    
        #for key in nsource.data.keys():
        #    print ("key", key, "len data:", len(nsource.data[key]))
        #TOOLS="crosshair,pan,wheel_zoom,box_zoom,undo,redo,reset,tap,save,box_select,poly_select,lasso_select"
        #TOOLS="crosshair,pan,wheel_zoom,box_zoom,undo,redo,reset,save"
        if plot_in:
            # update plot properties
            if verbose:
                print ("Updating plot properties...")
                print ("x_range:", x_range, "y_range:", y_range)
                print ("plot width, height:", plot_width, plot_height)
            #x_range = DataRange1d(start=min(nodex), end=max(nodex))
            #y_range = DataRange1d(start=min(nodey), end=max(nodey))
            out_dict['x_range'] =  x_range
            out_dict['y_range'] =  y_range
            out_dict['plot_height'] = plot_height
            #plot.x_range = DataRange1d(start=min(nodex), end=max(nodex))
            #plot.y_range = DataRange1d(start=min(nodey), end=max(nodey))
            #plot.plot_height = plot_height
        else:
            if verbose:
                print ("Creating new plot...")
            plot = figure(x_range=x_range, y_range=y_range,
                 plot_width=plot_width, plot_height=plot_height, 
                 x_axis_type="mercator", y_axis_type="mercator", tools='',
                 lod_threshold=lod_threshold)
                 #title=title, tools=TOOLS, active_scroll="wheel_zoom")#, active_tap="pan", )
            #plot = Plot(x_range=x_range, y_range=y_range,
            #         plot_width=plot_width, plot_height=plot_height)#,
            #         #x_axis_type="mercator", y_axis_type="mercator")#, title=title)
            plot.title.text = title
            # set wheelzoomtool to active
            wheezeetool = WheelZoomTool()
            plot.add_tools(PanTool(), wheezeetool, BoxZoomTool(), 
                           SaveTool(), ResetTool(), UndoTool(), RedoTool()) 
            plot.toolbar.active_scroll = wheezeetool 

        if map_background.upper() != 'NONE':      
            # https://wiki.openstreetmap.org/wiki/Tile_servers
            #https://github.com/bokeh/bokeh/issues/4770
            if map_background.upper() == 'STAMEN_TERRAIN':
                #ts = plot.add_tile(STAMEN_TERRAIN)
                tile = WMTSTileSource(
                url='http://tile.stamen.com/terrain/{Z}/{X}/{Y}.png',
                attribution=(
                    'Map tiles by <a href="http://stamen.com">Stamen Design</a>, '
                    'under <a href="http://creativecommons.org/licenses/by/3.0">CC BY 3.0</a>.'
                    'Data by <a href="http://openstreetmap.org">OpenStreetMap</a>, '
                    'under <a href="http://www.openstreetmap.org/copyright">ODbL</a>'
                    )
                )
                #ts = plot.add_tile(stamen_terrain)
            elif map_background.upper() == 'OSM':
                tile = WMTSTileSource(
                url='https://a.tile.openstreetmap.org/${z}/${x}/${y}.png ',
                attribution=('OpenStreeMap')
                )
            elif map_background.upper() == 'STAMEN_TONER':
                #ts = plot.add_tile(STAMEN_TONER)
                tile = WMTSTileSource(
                url='http://tile.stamen.com/toner/{Z}/{X}/{Y}.png',
                attribution=(
                    'Map tiles by <a href="http://stamen.com">Stamen Design</a>, '
                    'under <a href="http://creativecommons.org/licenses/by/3.0">CC BY 3.0</a>.'
                    'Data by <a href="http://openstreetmap.org">OpenStreetMap</a>, '
                    'under <a href="http://www.openstreetmap.org/copyright">ODbL</a>'
                    )
                )
                #ts = plot.add_tile(stamen_toner)
            #elif map_background.upper() == 'CARTODBPOSITRON_RETINA':
            #    ts = plot.add_tile(CARTODBPOSITRON_RETINA)
            # ADDRESS CUSTOM TILES
            else:
                osm_tile_options = {}
                if map_background.upper() == 'OSM':
                    osm_tile_options['url'] = 'http://c.tile.openstreetmap.org/{Z}/{X}/{Y}.png'
                elif map_background.upper() == 'OSM_TOPO':
                    osm_tile_options['url'] = 'http://c.tile.opentopomap.org/{z}/{x}/{y}.png'
                elif map_background.upper() == 'OPENMAPSURFER_ROADS':
                    osm_tile_options['url'] = 'http://korona.geog.uni-heidelberg.de/tiles/roads/x={x}&y={y}&z={z}'
                elif map_background.upper() == 'ESRI_STREET':
                    osm_tile_options['url'] = 'http://server.arcgisonline.com/ArcGIS/rest/services/World_Street_Map/MapServer/tile/{z}/{y}/{x}'
                elif map_background.upper() == 'ESRI_DELORME':
                    osm_tile_options['url'] = 'http://server.arcgisonline.com/ArcGIS/rest/services/Specialty/DeLorme_World_Base_Map/MapServer/tile/{z}/{y}/{x}'
                elif map_background.upper() == 'ESRI_TOPO':
                    osm_tile_options['url'] = 'http://server.arcgisonline.com/ArcGIS/rest/services/World_Topo_Map/MapServer/tile/{z}/{y}/{x}'
                elif map_background.upper() == 'ESRI_IMAGERY':
                    osm_tile_options['url'] = 'http://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}'
                elif map_background.upper() == 'ESRI_NATGEO':
                    osm_tile_options['url'] = 'http://server.arcgisonline.com/ArcGIS/rest/services/NatGeo_World_Map/MapServer/tile/{z}/{y}/{x}'
    
                ## labels
                #var Stamen_TonerHybrid = L.tileLayer('http://stamen-tiles-{s}.a.ssl.fastly.net/toner-hybrid/{z}/{x}/{y}.{ext}', {
                                          
                #osm_tile_options['use_latlon'] = True
                #osm_tile_source = BBoxTileSource(**osm_tile_options)
                tile = WMTSTileSource(**osm_tile_options)
                print ("osm_tile_source:", tile)#osm_tile_source)
                
            if not plot_in:
                ts = plot.add_tile(tile)  
                print ("  WMTSTileSource:", ts)
            else:
                out_dict['tile_source'] = tile
                                
    # add to plot 
    if add_glyphs:
        
        # edges
        if use_multiline:
            line = get_paths_glyph_line(coords=coords, color_key=ecolor_key)
            if not plot_in:
                road_glyph = plot.add_glyph(esource, line)
                if show_line_hover:
                    _ = add_line_hover(plot, renderers=[road_glyph])
            #else:
            #    out_dict['line'] = line
        else:
            # segments shudder when the graph is panned...
            seg = get_paths_glyph_seg(coords=coords)
            if not plot_in:
                road_glyph = plot.add_glyph(esource, seg)
        
        #if straight_lines:
        #    # below is unreliable!
        #    road_glyph = plot.add_glyph(esource, seg)
        #    ## Since Bokeh is busted and add_glyph frequently skips sements, try 
        #    ## splitting up and adding each segment individually
        #    #esource_list = split_columndatasource(esource)
        #    ##print ("esource_list", esource_list
        #    ##print ("len esource_list", len(esource_list)
        #    #for cds in esource_list:
        #    #    seg_glyph = plot.add_glyph(cds, seg)   
        #else:
        #    #print ("esource.data.keys()", esource.data.keys()
        #    road_glyph = plot.add_glyph(esource, line)

        if show_road_labels:
            seg_labels = get_paths_labels_glyph(esource, coords=coords)
            if not plot_in:
                seg_labels_glyph = plot.add_glyph(esource, seg_labels)

    # nodes (just to be cautius, split as well?)
    circ = get_nodes_glyph(coords=coords, shape='circle')
    #print ("G nodes glyph", circ
    #print ("nsource.data", nsource.data
    #circ = Circle(x="lon", y="lat", size='size', fill_color="color", \
    #        fill_alpha='alpha', line_color=None)#, legend='Intersections/Endpoints'
    if show_nodes and add_glyphs:
        if not plot_in:
            circ_glyph = plot.add_glyph(nsource, circ)
            glyph_node_list.extend([circ_glyph])

    # add axes          
    xaxis = LinearAxis(axis_label='Longitude')
    yaxis = LinearAxis(axis_label='Latitude')
    if not plot_in:
        plot.add_layout(xaxis, 'below')
        plot.add_layout(yaxis, 'left')
        # add grid
        xgrid = Grid(plot=plot, dimension=0, ticker=xaxis.ticker, \
                grid_line_dash="dashed", grid_line_color="gray")
        ygrid = Grid(plot=plot, dimension=1, ticker=yaxis.ticker, \
                grid_line_dash="dashed", grid_line_color="gray")
        plot.renderers.extend([xgrid, ygrid])  
                  
    # test, add legend
    if show_nodes and add_glyphs and not plot_in:
        #legends=[("Intersections/Endpoints", [circ_glyph])] 
        legends=[("Intersections/Endpoints", [circ_glyph]),
                             ("Road", [road_glyph])] 
        plot.add_layout(
                Legend(orientation="vertical", items=legends))
            #Legend(orientation="bottom_right", legends=legends))

    #bk.output_notebook()   # if within IPython notebook
    if not plot_in:
        add_hover_save(plot, htmlout=htmlout, show=show, add_hover=add_hover,
                          renderers=glyph_node_list)
    #bk.output_file(htmlout)
    #bk.save(obj=p)  #bk.save(p, filename='output.html')
    #if show:
    #    bk.show(p)
    
    print ("Time to plot G with Bokeh:", time.time() - t0, "seconds")
    if plot_in:
        return None, glyph_node_list, out_dict
    else:
        return plot, glyph_node_list, out_dict


###############################################################################
def G_to_bokeh(G, plot_in=None, global_dic=None, htmlout=None,
               map_background='None',
               nsize_key='deg', ncolor_key='ntype',
               ewidth_key='inferred_speed_mph',  # 'Num Lanes'
               width_mult=1.0/8,  # line width multiplier
               ecolor_key='inferred_speed_mph',
               show_nodes=True, show_road_labels=False,
               show=False, plot_width=1200,  title='None', add_glyphs=True,
               coords='wmp', use_multiline=True, add_hover=True,
               lod_threshold=None, # http://bokeh.pydata.org/en/latest/docs/user_guide/tools.html#controlling-level-of-detail
               show_line_hover=True, verbose=False):
    '''
    Create a bokeh plot of the network G.
    if plot_in: update existing plot
    Performance could be improved.
    Try splitting edge arrays based on color, then plotting each color
    individually and adding that color to the legend
    coords can be 'latlon' or 'wmp', assume coords are in wmp
    if lod_threshold = None, displacy all lines at all times
    !! This function could be sped up marketly by using ecurve_dic and
       removing some arrays that are extraneous !!
    '''

    out_dict = {}
    glyph_node_list = []
    plot_glyph_list = []
    color_dic, alpha_dic = define_colors_alphas()
    mapzoom = global_dic['mapzoom']
    t0 = time.time()

    # node properties
    constant_node_size = global_dic['gnode_size']  #None   # set to None if we want dynamic node sizes
    nsource = G_to_nsource(G, nsize_key=nsize_key,
                           constant_node_size=constant_node_size,
                           ncolor_key=ncolor_key,
                           verbose=verbose)
    nodex = nsource.data['x']
    nodey = nsource.data['y']
    if verbose:
        print("Time to create node arrays:", time.time() - t0, "seconds")

    # edge arrays
    # numE = len(G.edges())
    t1 = time.time()
    esource = G_to_esource(G, use_multiline=use_multiline,
                           ewidth_key=ewidth_key, ecolor_key=ecolor_key,
                           width_mult=width_mult, verbose=verbose)
    if verbose:
        print("Time to create edge arrays:", time.time() - t1, "seconds")

    # find x, y extent of plot (lat,lon)
    x_mid = 0.5*(max(nodex) + min(nodex))
    y_mid = 0.5*(max(nodey) + min(nodey))
    if verbose:
        print("x_mid:", x_mid, "y_mid:", y_mid)

    # get min, max for plotting
    # xmintmp, ymintmp = latlon_to_wmp(minlat, minlon)
    # xmaxtmp, ymaxtmp = latlon_to_wmp(maxlat, maxlon)
    if coords == 'latlon':
        minlat, maxlon = min(nodey), max(nodey)
        minlon, maxlat = min(nodex), max(nodex)
        xmin, xmax, ymin, ymax = minlon, maxlon, minlat, maxlat
    elif coords == 'wmp':
        xmin, xmax, ymin, ymax = min(nodex), max(nodex), min(nodey), max(nodey)
    else:
        print("Unknown coords in G_to_bokeh()")
        return
    if verbose:
        print("Plotting limits:", xmin, xmax, ymin, ymax)

    # plot
    ########################
    if map_background.upper() == 'GMAP':
        print("Create GMAP...")

        # Now need an api key
        # see: https://github.com/bokeh/bokeh/blob/0.12.0/examples/models/maps.py#L19-L27
        # also see: https://developers.google.com/maps/documentation/javascript/get-api-key
        api_key = '???'

        # try importing a gmap
        # https://github.com/bokeh/bokeh/blob/master/examples/glyphs/maps.py
        # https://github.com/bokeh/bokeh/blob/master/examples/glyphs/trail.py

        # set map height, options
        plot_height = int(0.75*plot_width)   
        map_options = GMapOptions(lat=y_mid, lng=x_mid, zoom=mapzoom)   
        #geojs_map_options = GeoJSOptions(lat=y_mid, lng=x_mid, zoom=mapzoom)   
        x_range = Range1d()
        y_range = Range1d()           
        plot = GMapPlot(x_range=x_range, y_range=y_range, 
                        map_options=map_options, 
                        plot_height=plot_height, plot_width=plot_width, 
                        #title=title,
                        api_key=api_key)  # set title later 
        plot.title.text = title
        plot.map_options.map_type='hybrid'
#        plot = GeoJSPlot(x_range=x_range, y_range=y_range, 
#                        map_options=geojs_map_options, 
#                        plot_height=plot_height, plot_width=plot_width, 
#                        title=title)  # set title later 
        
        # !! Reset Tool goes to lat, lon = 0,0 !! 
        plot.add_tools(PanTool(), WheelZoomTool(), BoxZoomTool(), \
                SaveTool(), )#, ResetTool())   

    #################
    #if not gmap_background:
    else:
        # list of map servers: https://leaflet-extras.github.io/leaflet-providers/preview/
        # plot (no background)    
        # set plot width and height such that lat lon degrees are equivalent
        # compute distances along x and y midpoints
        if coords == 'wmp':
            distx = distance_euclid(y_mid, min(nodex), y_mid, max(nodex))
            disty = distance_euclid(min(nodey), x_mid, max(nodey), x_mid)
        elif coords == 'latlon':
            distx = distance(y_mid, min(nodex), y_mid, max(nodex))
            disty = distance(min(nodey), x_mid, max(nodey), x_mid)
        plot_height = int(float(plot_width * disty/distx)) 
        if verbose:
            print ("distx:", distx, "disty:", disty)            
        # old, crummy version
        #x_extent = max(nodex) - min(nodex)
        #y_extent = max(nodey) - min(nodey)
        #plot_height = int(float(plot_width) * (y_extent/x_extent))
        # keep it square>
        #plot_height = plot_width

        #x_range = DataRange1d()#sources=[source.columns("nodex")])
        #y_range = DataRange1d()#sources=[source.columns("nodey")])
        x_range = (min(nodex), max(nodex))
        y_range = (min(nodey), max(nodey))
        if verbose:
            print ("x_range:", x_range)
            print ("y_range:", y_range)
    
        #for key in nsource.data.keys():
        #    print ("key", key, "len data:", len(nsource.data[key]))
        #TOOLS="crosshair,pan,wheel_zoom,box_zoom,undo,redo,reset,tap,save,box_select,poly_select,lasso_select"
        #TOOLS="crosshair,pan,wheel_zoom,box_zoom,undo,redo,reset,save"
        if plot_in:
            # update plot properties
            if verbose:
                print ("Updating plot properties...")
                print ("x_range:", x_range, "y_range:", y_range)
                print ("plot width, height:", plot_width, plot_height)
            #x_range = DataRange1d(start=min(nodex), end=max(nodex))
            #y_range = DataRange1d(start=min(nodey), end=max(nodey))
            out_dict['x_range'] =  x_range
            out_dict['y_range'] =  y_range
            out_dict['plot_height'] = plot_height
            #plot.x_range = DataRange1d(start=min(nodex), end=max(nodex))
            #plot.y_range = DataRange1d(start=min(nodey), end=max(nodey))
            #plot.plot_height = plot_height
        else:
            if verbose:
                print ("Creating new plot...")
            plot = figure(x_range=x_range, y_range=y_range,
                 plot_width=plot_width, plot_height=plot_height, 
                 x_axis_type="mercator", y_axis_type="mercator", tools='',
                 lod_threshold=lod_threshold)
                 #title=title, tools=TOOLS, active_scroll="wheel_zoom")#, active_tap="pan", )
            #plot = Plot(x_range=x_range, y_range=y_range,
            #         plot_width=plot_width, plot_height=plot_height)#,
            #         #x_axis_type="mercator", y_axis_type="mercator")#, title=title)
            plot.title.text = title
            # set wheelzoomtool to active
            wheezeetool = WheelZoomTool()
            plot.add_tools(PanTool(), wheezeetool, BoxZoomTool(), 
                           SaveTool(), ResetTool(), UndoTool(), RedoTool()) 
            plot.toolbar.active_scroll = wheezeetool 

        if map_background.upper() != 'NONE':      
            # https://wiki.openstreetmap.org/wiki/Tile_servers
            #https://github.com/bokeh/bokeh/issues/4770
            if map_background.upper() == 'STAMEN_TERRAIN':
                #ts = plot.add_tile(STAMEN_TERRAIN)
                tile = WMTSTileSource(
                url='http://tile.stamen.com/terrain/{Z}/{X}/{Y}.png',
                attribution=(
                    'Map tiles by <a href="http://stamen.com">Stamen Design</a>, '
                    'under <a href="http://creativecommons.org/licenses/by/3.0">CC BY 3.0</a>.'
                    'Data by <a href="http://openstreetmap.org">OpenStreetMap</a>, '
                    'under <a href="http://www.openstreetmap.org/copyright">ODbL</a>'
                    )
                )
                #ts = plot.add_tile(stamen_terrain)
            elif map_background.upper() == 'OSM':
                tile = WMTSTileSource(
                url='https://a.tile.openstreetmap.org/${z}/${x}/${y}.png ',
                attribution=('OpenStreeMap')
                )
            elif map_background.upper() == 'STAMEN_TONER':
                #ts = plot.add_tile(STAMEN_TONER)
                tile = WMTSTileSource(
                url='http://tile.stamen.com/toner/{Z}/{X}/{Y}.png',
                attribution=(
                    'Map tiles by <a href="http://stamen.com">Stamen Design</a>, '
                    'under <a href="http://creativecommons.org/licenses/by/3.0">CC BY 3.0</a>.'
                    'Data by <a href="http://openstreetmap.org">OpenStreetMap</a>, '
                    'under <a href="http://www.openstreetmap.org/copyright">ODbL</a>'
                    )
                )
                #ts = plot.add_tile(stamen_toner)
            #elif map_background.upper() == 'CARTODBPOSITRON_RETINA':
            #    ts = plot.add_tile(CARTODBPOSITRON_RETINA)
            # ADDRESS CUSTOM TILES
            else:
                osm_tile_options = {}
                if map_background.upper() == 'OSM':
                    osm_tile_options['url'] = 'http://c.tile.openstreetmap.org/{Z}/{X}/{Y}.png'
                elif map_background.upper() == 'OSM_TOPO':
                    osm_tile_options['url'] = 'http://c.tile.opentopomap.org/{z}/{x}/{y}.png'
                elif map_background.upper() == 'OPENMAPSURFER_ROADS':
                    osm_tile_options['url'] = 'http://korona.geog.uni-heidelberg.de/tiles/roads/x={x}&y={y}&z={z}'
                elif map_background.upper() == 'ESRI_STREET':
                    osm_tile_options['url'] = 'http://server.arcgisonline.com/ArcGIS/rest/services/World_Street_Map/MapServer/tile/{z}/{y}/{x}'
                elif map_background.upper() == 'ESRI_DELORME':
                    osm_tile_options['url'] = 'http://server.arcgisonline.com/ArcGIS/rest/services/Specialty/DeLorme_World_Base_Map/MapServer/tile/{z}/{y}/{x}'
                elif map_background.upper() == 'ESRI_TOPO':
                    osm_tile_options['url'] = 'http://server.arcgisonline.com/ArcGIS/rest/services/World_Topo_Map/MapServer/tile/{z}/{y}/{x}'
                elif map_background.upper() == 'ESRI_IMAGERY':
                    osm_tile_options['url'] = 'http://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}'
                elif map_background.upper() == 'ESRI_NATGEO':
                    osm_tile_options['url'] = 'http://server.arcgisonline.com/ArcGIS/rest/services/NatGeo_World_Map/MapServer/tile/{z}/{y}/{x}'
    
                ## labels
                #var Stamen_TonerHybrid = L.tileLayer('http://stamen-tiles-{s}.a.ssl.fastly.net/toner-hybrid/{z}/{x}/{y}.{ext}', {
                                          
                #osm_tile_options['use_latlon'] = True
                #osm_tile_source = BBoxTileSource(**osm_tile_options)
                tile = WMTSTileSource(**osm_tile_options)
                print ("osm_tile_source:", tile)#osm_tile_source)
                
            if not plot_in:
                ts = plot.add_tile(tile)  
                print ("  WMTSTileSource:", ts)
            else:
                out_dict['tile_source'] = tile
                                
    # add to plot 
    if add_glyphs:
        
        # edges
        if use_multiline:
            line = get_paths_glyph_line(coords=coords)
            if not plot_in:
                road_glyph = plot.add_glyph(esource, line)
                if show_line_hover:
                    _ = add_line_hover(plot, renderers=[road_glyph])
            #else:
            #    out_dict['line'] = line
        else:
            # segments shudder when the graph is panned...
            seg = get_paths_glyph_seg(coords=coords)
            if not plot_in:
                road_glyph = plot.add_glyph(esource, seg)
        
        #if straight_lines:
        #    # below is unreliable!
        #    road_glyph = plot.add_glyph(esource, seg)
        #    ## Since Bokeh is busted and add_glyph frequently skips sements, try 
        #    ## splitting up and adding each segment individually
        #    #esource_list = split_columndatasource(esource)
        #    ##print ("esource_list", esource_list
        #    ##print ("len esource_list", len(esource_list)
        #    #for cds in esource_list:
        #    #    seg_glyph = plot.add_glyph(cds, seg)   
        #else:
        #    #print ("esource.data.keys()", esource.data.keys()
        #    road_glyph = plot.add_glyph(esource, line)

        if show_road_labels:
            seg_labels = get_paths_labels_glyph(esource, coords=coords)
            if not plot_in:
                seg_labels_glyph = plot.add_glyph(esource, seg_labels)

    # nodes (just to be cautius, split as well?)
    circ = get_nodes_glyph(coords=coords, shape='circle')
    #print ("G nodes glyph", circ
    #print ("nsource.data", nsource.data
    #circ = Circle(x="lon", y="lat", size='size', fill_color="color", \
    #        fill_alpha='alpha', line_color=None)#, legend='Intersections/Endpoints'
    if show_nodes and add_glyphs:
        if not plot_in:
            circ_glyph = plot.add_glyph(nsource, circ)
            glyph_node_list.extend([circ_glyph])

    # add axes          
    xaxis = LinearAxis(axis_label='Longitude')
    yaxis = LinearAxis(axis_label='Latitude')
    if not plot_in:
        plot.add_layout(xaxis, 'below')
        plot.add_layout(yaxis, 'left')
        # add grid
        xgrid = Grid(plot=plot, dimension=0, ticker=xaxis.ticker, \
                grid_line_dash="dashed", grid_line_color="gray")
        ygrid = Grid(plot=plot, dimension=1, ticker=yaxis.ticker, \
                grid_line_dash="dashed", grid_line_color="gray")
        plot.renderers.extend([xgrid, ygrid])  
                  
    # test, add legend
    if show_nodes and add_glyphs and not plot_in:
        #legends=[("Intersections/Endpoints", [circ_glyph])] 
        legends=[("Intersections/Endpoints", [circ_glyph]),
                             ("Road", [road_glyph])] 
        plot.add_layout(
                Legend(orientation="vertical", items=legends))
            #Legend(orientation="bottom_right", legends=legends))

    #bk.output_notebook()   # if within IPython notebook
    if not plot_in:
        add_hover_save(plot, htmlout=htmlout, show=show, add_hover=add_hover,
                          renderers=glyph_node_list)
    #bk.output_file(htmlout)
    #bk.save(obj=p)  #bk.save(p, filename='output.html')
    #if show:
    #    bk.show(p)
    
    print ("Time to plot G with Bokeh:", time.time() - t0, "seconds")
    if plot_in:
        return None, nsource, esource, glyph_node_list, out_dict
    else:
        return plot, nsource, esource, glyph_node_list, out_dict



###############################################################################
### Columndatasource funcs
###############################################################################

###############################################################################
def set_bbox_source(bboxes, color='red', \
                fill_alpha=0.35, line_alpha=0.8, shape='circle', 
                label=[], count=[], val=[], name=[]):

    '''
    DEPRECATED AND UNUSED
    Assume bbox is [[x0, y0, x1, y1], ...]
    
    Make cds for get_bbox_glyph()...
    
    def get_bbox_glyph(text_alpha=0.9, text_font_size='5pt'):            
        quad_glyph = Quad(left='left', right='right', top='top', bottom='bottom',
                          fill_color='color', fill_alpha='alpha', 
                          line_color='color', line_alpha='alpha_line')
        text_glyph = Text(x='left', y='top', text="num", text_alpha=text_alpha,
                          text_font_size=text_font_size, text_baseline="middle", 
                          text_align="center")

    '''
    
    left = bboxes[:, 0]
    bottom = bboxes[:, 1]
    right = bboxes[:, 2]
    top = bboxes[:, 3]
    
    N = len(bboxes)
    # convert variables to list
    if type(color) == str:
        color = np.asarray(N * [color])
    # lists of length 0 are now full of emptry strings 
    if len(label) == 0:
        label = np.asarray(N * [''])
    if len(count) == 0:
        count = np.asarray(N * [''])
    if len(val) == 0:
        val = np.asarray(N * [''])
    if len(name) == 0:
        name = np.asarray(N * [''])
    fill_alpha = fill_alpha * np.ones(N)#N * [fill_alpha]
    line_alpha = line_alpha * np.ones(N)#N * [fill_alpha]
    
    
    
    # create a columndatasource
    source = bk.ColumnDataSource(data=dict(
            left=left,
            bottom=bottom,
            right=right,
            top=top,
            color=color,
            fill_alpha=fill_alpha,
            line_alpha=line_alpha
            #label=np.asarray(label),
            #name=np.asarray(name),
            #count=np.asarray(count),
            #val=np.asarray(val),
            
            #text_font_size=tsize,
            #text_font_style=tstyle,
            #text_color=tcolor,
            #text_alpha=talpha
            ))        
    
    return source


###############################################################################
def set_nodes_source(G, nodes, size=6, color='red', \
                fill_alpha=0.6, shape='circle', label=[], count=[], val=[],
                name=[]):
                #text_alpha=0.9, text_font_size='5pt', text_font_style='normal', 
                #text_color='black'):
    '''Set columndatasource for nodes
    nodes should be the ids of osm nodes'''

    N = len(nodes)
    # convert variables to list
    nodes = np.asarray(nodes)
    if type(size) == int or type(size) == float:
        size = size * np.ones(N)#N * [size]
    if type(color) == str:
        color = np.asarray(N * [color])
    # lists of length 0 are now full of emptry strings 
    if len(label) == 0:
        label = np.asarray(N * [''])
    if len(count) == 0:
        count = np.asarray(N * [''])
    if len(val) == 0:
        val = np.asarray(N * [''])
    if len(name) == 0:
        name = np.asarray(N * [''])
    shape_list = np.asarray(N * [shape])
    alpha_list = fill_alpha * np.ones(N)#N * [fill_alpha]
    nearest = nodes
    #tsizet = N * [text_font_size]
    #tstyle = N * [text_font_style]
    #tcolor = N * [text_color]
    #talpha = N * [talpha]

    # extract locations
    #x, y = [], []
    #for n in nodes:
    #    x.append(G.nodes[n]['lon'])
    #    y.append(G.nodes[n]['lat'])
    # np arrays should be faster   
    
    lat, lon = np.zeros(N), np.zeros(N)
    x_wmp, y_wmp = np.zeros(N), np.zeros(N)
    for i,n in enumerate(nodes):
        lon[i] = G.nodes[n]['lon']
        lat[i] = G.nodes[n]['lat']
        x_wmp[i] = G.nodes[n]['x']
        y_wmp[i] = G.nodes[n]['y']

    # convert to web_mercator_projection (should already be in node properties)
    #x_wmp,y_wmp = latlon_to_wmp(lat, lon)

    # create a columndatasource
    source = bk.ColumnDataSource(data=dict(
            lon=lon, 
            lat=lat, 
            x_wmp=x_wmp,
            y_wmp=y_wmp,
            x=x_wmp,
            y=y_wmp,
            label=np.asarray(label),
            nid=nodes,
            name=np.asarray(name),
            count=np.asarray(count),
            val=np.asarray(val),
            size=np.asarray(size),
            color=np.asarray(color),
            shape=shape_list,
            alpha=alpha_list,
            nearest=nearest
            #text_font_size=tsize,
            #text_font_style=tstyle,
            #text_color=tcolor,
            #text_alpha=talpha
            ))          

    return source


###############################################################################
def set_nodes_source_empty(val=0):
    '''create a empty columndatasource'''
    
    if val == None:
        v = np.array([])
    else:
        v = np.array([val])
        
    source = bk.ColumnDataSource(data=dict(
            lon=v, 
            lat=v, 
            x=v,
            y=v,
            x_wmp=v,
            y_wmp=v,
            label=v,
            nid=v,
            name=v,
            count=v,
            val=v,
            size=v,
            color=v,
            shape=v,
            alpha=v,
            nearest=v
            #text_font_size=tsize,
            #text_font_style=tstyle,
            #text_color=tcolor,
            #text_alpha=talpha
            ))          
    return source


###############################################################################
def set_bbox_source_empty(val=0):
    '''create a empty columndatasource
    source_bbox.data.keys(): dict_keys(['level_0', 'Unnamed: 0', 'Loc_Tmp', 
    'Prob', 'Xmin', 'Ymin', 'Xmax', 'Ymax', 'Category', 'Image_Root_Plus_XY', 
    'Image_Root', 'Slice_XY', 'Upper', 'Left', 'Height', 'Width', 'Pad', 
    'Im_Width', 'Im_Height', 'Image_Path', 'Xmin_Glob', 'Xmax_Glob', 
    'Ymin_Glob', 'Ymax_Glob', 
    'Xmin_wmp', 'Xmax_wmp', 'Ymin_wmp', 'Ymax_wmp', 
    'Val', 'count', 'num', 'line_alpha', 'fill_alpha', 'nearest_osm', 'dist', 
    'status', 'color', 'Xmid_wmp', 'Ymid_wmp', 'index'])

    '''
    
    if val == None:
        v = np.array([])
    else:
        v = np.array([val])

   # create a columndatasource
    source = bk.ColumnDataSource(data=dict(
            Xmin_wmp=v,
            Xmax_wmp=v,
            Ymin_wmp=v,
            Ymax_wmp=v,
            Val=v,
            val=v,
            count=v,
            num=v,
            fill_alpha=v,
            line_alpha=v,
            nearest_osm=v,
            dist=v,
            status=v,
            color=v,
            Xmid_wmp=v,
            Ymid_wmp=v,
            index=v ,           
            #left=v,
            #bottom=v,
            #right=v,
            #top=v,
            text_font_size=v,
            text_font_style=v,
            text_color=v,
            text_alpha=v
            #label=np.asarray(label),
            #name=np.asarray(name),
            ))        
    
    return source


###############################################################################
def set_hull_patch_source_empty(val=0):
    '''create a empty columndatasource
    source_hull_patch = bk.ColumnDataSource(dict(xs=xlist_hull, ys=ylist_hull,
                        alpha=alpha_dic['hull']*np.ones(len(xlist_hull)),
                        color=np.asarray(palette[:len(xlist_hull)]) ))
    '''
    
    if val == None:
        v = np.array([])
    else:
        v = np.array([val])
        
    source = bk.ColumnDataSource(data=dict(
            xs=v, 
            ys=v, 
            alpha=v,
            color=v,
            fill_alpha=v,
            fill_color=v,
            ))          
    return source


###############################################################################
def set_rect_source_empty(val=0):  #(val=None):
    '''create a empty columndatasource
    rect_bin = Rect(x='x', y='y_mid',  height='y',  width='width', 
                    fill_color='color', fill_alpha='alpha') #get_histo_glyph()
    '''

    if val == None:
        v = np.array([])
    else:
        v = np.array([val])
        
    source = bk.ColumnDataSource(data=dict(
            x=v, 
            y=v, 
            y_mid=v,
            width=v,
            color=v,
            alpha=v
            ))          
    return source


###############################################################################
def set_edge_source_empty():
    # create columndatasource
    v = np.array([0])
    esource = bk.ColumnDataSource(
        data=dict(
            elat0=v, elon0=v,
            elat1=v, elon1=v,
            ex0=v, ey0=v,
            ex1=v, ey1=v,
            emx=v, emy=v,
            ewidth=v,
            ecolor=v,
            elabel=v,
            ealpha=v,
            elx0=v, ely0=v,
            ellat0=v, ellon0=v,
            uv=v,
            cong_color=v,
            raw_color=v,
            congestion=v,
            speed2=v,
            speed_mph=v,
            length=v,
            lanes=v,
            name=v,
            plot_color=v
        )
    )
    return esource


###############################################################################
def make_histo_arrs(x, y=[], binsize=0.5):
    '''Create binned and cumulative histograms
    If y=None, just use number of counts per xbin'''

    # first, bin by path length
    if len(x) == 0:
        return [], [], [], [], []
    bins = np.arange(binsize, max(x)+binsize, binsize)
    # reformate bins_out to reflect the midpoint of each bin
    bins_out = bins - 0.5*binsize
    # bin_str must be list since bokeh range setting cannot handle np.array!
    bin_str = np.asarray([str(round(b-binsize,2)) + '-' + str(round(b,2)) + 'H' for b in bins])
    #print ("bin_str", bin_str
    digitized = np.digitize(x, bins)
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
def get_histo_source(x_arr, y_arr, color='green', alpha=0.6, 
                   width=0.99,
                   legend='', x_str_arr=None):
                       
    # numpy arrays are faster
    x_arr = np.asarray(x_arr)
    y_arr = np.asarray(y_arr)
    x_str_arr = np.asarray(x_str_arr)

    # for categorical plots, we want width of 0.99, else we want widt of 
    # 0.99 * bin width
    if len(x_arr) > 1:
        width = 0.99 * (x_arr[1] - x_arr[0])
    
    # create a columndatasource
    source = bk.ColumnDataSource(data=dict(
            x=x_arr,
            x_str=x_str_arr,
            y=y_arr,
            y_mid=y_arr/2.,#[y/2 for y in y_arr],
            color=np.asarray(len(x_arr)*[color]),
            alpha=np.asarray(len(x_arr)*[alpha]),
            width=np.asarray(len(x_arr)*[width])
            #legend=legend
            ))   
    
    return source

###############################################################################
def split_columndatasource(source, block_size=5000):
    '''add_glyph fails for segments if the columndatasource is too large, so
    take a columndatasource (CDS) and split it into a number of smaller CDS
    return a list of CDS
    !! Assume all arrays are the same length !!'''

    print ("Splitting ColumnDataSource for plotting purposes...")
    keys = source.data.keys()
    N = len(source.data[keys[0]])
    indices = range(0, N, block_size)
    source_list = []
    #print ("keys", keys
    #print ("data length", N
    #print ("indices", indices
    numI = len(indices)
    
    # iterate through source indices    
    for iteri in range(numI):
        # i is the beginning index, set the end index
        i = indices[iteri]
        if iteri == numI - 1:
            j = N + block_size
        else:
            j = indices[iteri + 1]
        print ("chunkinging columndatasource from", i, "to", j)
        
        # populate data dict
        dic_tmp = {}
        for key in keys:
            dic_tmp[key] = source.data[key][i:j]
        
        source_tmp = bk.ColumnDataSource(data=dic_tmp)
        source_list.append(source_tmp)
       
    return source_list   
    

###############################################################################
def keep_bbox_indices(source_gdelt, indices):
    '''indices to keep'''

    # keep only desired indices
    # need to make a copy since edits here are apparently global

    if len(indices) == 0:
        return set_nodes_source_empty()

    else:
        new_dic = {}
        #print ("indices", indices
        for key in source_gdelt.data.keys():
            #print ("source_gdelt.data[key]", source_gdelt.data[key]
        
            # python list version
            #new_dic[key] = [source_gdelt.data[key][i] for i in indices]
            # !! numpy array version, should be faster !!
            new_dic[key] = source_gdelt.data[key][indices]
          
        return bk.ColumnDataSource(data=new_dic)
              

###############################################################################
### glyphs
###############################################################################
def get_paths_glyph_seg(coords='wmp'):
    '''Plot paths, assume a columndatasource holds all the data
    columns should include:
        elat0=elat0, elon0=elon0,
        elat1=elat1, elon1=elon1,
        ex0, ey0,
        ex1, ey1,
        emx, emy,
        ewidth, 
        ecolor, 
        ealpha,
        elabel (optional)'''

#    # reset color, if desired
#    if ecolor:
#        esource.data['ecolor'] = len(esource.data['ecolor'])*[ecolor]

    if coords == 'latlon':
        x0tmp = 'elon0'
        y0tmp = 'elat0'
        x1tmp = 'elon1'
        y1tmp = 'elat1'
    elif coords == 'wmp':
        x0tmp = 'ex0'
        y0tmp = 'ey0'
        x1tmp = 'ex1'
        y1tmp = 'ey1'
    else:
        print ("oops")
        return

    #print ("get_paths_glyph_seg: x0tmp:", x0tmp)
    seg = Segment(x0=x0tmp, y0=y0tmp, x1=x1tmp, y1=y1tmp, \
            line_width="ewidth", line_color="ecolor", \
            line_alpha="ealpha") 

    #p.add_glyph(esource, seg)
    return seg


###############################################################################
def get_paths_glyph_line(coords='wmp', color_key='ecolor'):
    '''Plot paths, assume a columndatasource holds all the data
    columns should include:
        elat0=elat0, elon0=elon0,
        elat1=elat1, elon1=elon1,      
        ex0, ey0,
        ex1, ey1,
        emx, emy,
        ewidth, 
        ecolor, 
        ealpha,
        elabel (optional)
        elx0,
        ely0'''

#    # reset color, if desired
#    if ecolor:
#        esource.data['ecolor'] = len(esource.data['ecolor'])*[ecolor]
    if coords == 'latlon':
        xtmp = 'ellon0'
        ytmp = 'ellat0'
    elif coords == 'wmp':
        xtmp = 'elx0'
        ytmp = 'ely0'
    else:
        print ("oops")
        return

    #print ("get_paths_glyph_line: xtmp:", xtmp)
    l = MultiLine(xs=xtmp, ys=ytmp,
            line_width="ewidth", line_color=color_key, \
            line_alpha="ealpha") 
    return l


###############################################################################   
def get_paths_labels_glyph(text_alpha=0.7, text_font_size='5pt', 
                           coords='latlon'):
    '''segment labels
    Still need to address non latlon coords!!!!'''
    
    text = Text(x="emx", y="emy", text="elabel", text_alpha=text_alpha, \
               text_font_size=text_font_size, \
               text_baseline="middle", text_align="center")
    #p.add_glyph(esource, text)  
    return text

###############################################################################        
def get_bbox_glyph(text_alpha=0.9, coords='wmp', text_font_size='5pt'):            
    '''Get bbox glyphs
    https://bokeh.pydata.org/en/latest/docs/reference/plotting.html#bokeh.plotting.figure.Figure.quad
    '''
       
#    quad_glyph = Quad(left='x0_wmp', right='x1_wmp', 
#                      top='y0_wmp', bottom='y1_wmp',
#                      fill_color='color', fill_alpha='fill_alpha', 
#                      line_color='color', line_alpha='line_alpha')
#    text_glyph = Text(x='x0_wmp', y='y0_wmp', text="num", 
#                      text_alpha=text_alpha,
#                      text_font_size=text_font_size, text_baseline="middle", 
#                      text_align="center")
    quad_glyph = Quad(left='Xmin_wmp', right='Xmax_wmp', 
                      top='Ymax_wmp', bottom='Ymin_wmp',
                      fill_color='color', fill_alpha='fill_alpha', 
                      line_color='color', line_alpha='line_alpha')
    text_glyph = Text(x='Xmin_wmp', y='Ymax_wmp', text="num", 
                      text_alpha=text_alpha,
                      text_font_size=text_font_size, text_baseline="middle", 
                      text_align="center")
                        
    return quad_glyph, text_glyph


###############################################################################        
def get_bbox_glyph_v0(text_alpha=0.9, text_font_size='5pt'):            
    '''Get bbox glyphs
    https://bokeh.pydata.org/en/latest/docs/reference/plotting.html#bokeh.plotting.figure.Figure.quad
    '''
       
    quad_glyph = Quad(left='left', right='right', top='top', bottom='bottom',
                      fill_color='color', fill_alpha='fill_alpha', 
                      line_color='color', line_alpha='line_alpha')
    text_glyph = Text(x='left', y='top', text="num", text_alpha=text_alpha,
                      text_font_size=text_font_size, text_baseline="middle", 
                      text_align="center")
                        
    return quad_glyph, text_glyph


###############################################################################        
def get_gdelt_glyph(text_alpha=0.9, text_font_size='5pt', coords='wmp'):            
    '''Get gdelt glyphs'''
    
    if coords == 'latlon':
        xname = 'lon'
        yname = 'lat'
    elif coords == 'wmp':
        xname = 'x_wmp'
        yname = 'y_wmp'
    else:
        print ("oops")
        return
    
    circle_glyph = Circle(x=xname, y=yname, size='size', 
                fill_color='color', fill_alpha='alpha', line_color=None)
    text_glyph = Text(x=xname, y=yname, text="num", text_alpha=text_alpha,
                text_font_size=text_font_size, text_baseline="middle", 
                text_align="center")
                        
    return circle_glyph, text_glyph


###############################################################################
def get_nodes_glyph(shape='circle', coords='wmp'):
    '''source should have the form:
        # create a columndatasource
        source = bk.ColumnDataSource(data=dict(
            lon=x, 
            lat=y, 
            label=label,
            nid=nodes,
            count=count,
            val=val,
            size=size,
            color=color,
            shape=shape_list,
            alpha=alpha_list,
            nearest=nearest
            ))    
    label (optional) are node labels
    nid, count, val (each optional) are for hover text
    shape options: square, circle, asterisk, circlecross, diamond, 
            inverted triangle, triangle
    http://bokeh.pydata.org/en/latest/docs/gallery/scatter.html'''
        
    # could infer shape from source, but easier just to pass in a variable
    #shape=source.data['shape'][0]

    if coords == 'latlon':
        xname = 'lon'
        yname = 'lat'
    elif coords == 'wmp':
        xname = 'x'
        yname = 'y'
    else:
        print ("oops")
        return
            
    if shape.lower() == 'square':
        s = Square(x=xname, y=yname, size='size', 
            fill_color='color', fill_alpha='alpha', line_color=None)
    elif shape.lower() == 'circle':
        s = Circle(x=xname, y=yname, 
            #size='size',    # screen units
            radius='size',   # dataspace units
            fill_color='color', fill_alpha='alpha', line_color=None)
    elif shape.lower() == 'triangle':
        s = Triangle(x=xname, y=yname, size='size', 
            fill_color='color', fill_alpha='alpha', line_color=None)
    elif shape.lower() == 'invertedtriangle':
        s = InvertedTriangle(x=xname, y=yname, size='size', 
            fill_color='color', fill_alpha='alpha', line_color=None)
    elif shape.lower() == 'diamond':
        s = Diamond(x=xname, y=yname, size='size', 
            fill_color='color', fill_alpha='alpha', line_color=None)
    #p.add_glyph(source, s)
    return s


###############################################################################
def get_nodes_labels_glyph(text_alpha=0.2, text_font_size='5pt', 
                             text_font_style='normal', text_color='black',
                             coords='wmp'):
    '''Get text label glyph'''

    if coords == 'latlon':
        xname = 'lon'
        yname = 'lat'
    elif coords == 'wmp':
        xname = 'x_wmp'
        yname = 'y_wmp'
    else:
        print ("oops")
        return
      
    
    text = Text(x=xname, y=yname, text='label', text_alpha=text_alpha, \
                   text_font_size=text_font_size, \
                   text_font_style=text_font_style, \
                   text_baseline="middle", \
                   text_align="center", text_color=text_color)
    #p.add_glyph(source, text)
    return text


###############################################################################
def get_route_glyphs_all(coords='wmp'):
    '''Define shapes for route plotting'''
    
    paths_seg = get_paths_glyph_seg(coords=coords)
    paths_seg_sec = get_paths_glyph_seg(coords=coords)
    
    paths_line = get_paths_glyph_line(coords=coords)
    paths_line_sec = get_paths_glyph_line(coords=coords)

    target_shape = get_nodes_glyph(shape='invertedtriangle', coords=coords)
    target_text = get_nodes_labels_glyph(text_alpha=0.9, text_font_size='8pt', 
                             text_font_style='bold', text_color='black',
                             coords=coords)

    sources_shape = get_nodes_glyph(shape='square', coords=coords)
    sources_text = get_nodes_labels_glyph(text_alpha=0.9, text_font_size='7pt', 
                             text_font_style='bold', text_color='black',
                             coords=coords)

    crit_shape = get_nodes_glyph(shape='triangle', coords=coords)
    crit_text = get_nodes_labels_glyph(text_alpha=0.1, text_font_size='5pt', 
                             text_font_style='normal', text_color='black',
                             coords=coords)           
 
    missing_shape = get_nodes_glyph(shape='square', coords=coords)
    missing_text = get_nodes_labels_glyph(text_alpha=0.5, text_font_size='5pt', 
                             text_font_style='normal', text_color='black',
                             coords=coords)           

    subgraph_shape = get_nodes_glyph(shape='diamond', coords=coords)
    subgraph_text = get_nodes_labels_glyph(text_alpha=0.15, text_font_size='5pt', 
                             text_font_style='normal', text_color='black',
                             coords=coords)

    diff_shape = get_nodes_glyph(shape='square', coords=coords)
    diff_text = get_nodes_labels_glyph(text_alpha=0.15, text_font_size='5pt', 
                             text_font_style='normal', text_color='black',
                             coords=coords)           

    rect_bin = Rect(x='x', y='y_mid',  height='y',  width='width', 
                    fill_color='color', fill_alpha='alpha') #get_histo_glyph()
    rect_cum = Rect(x='x', y='y_mid',  height='y',  width='width', 
                    fill_color='color', fill_alpha='alpha') #get_histo_glyph()

    risk_shape = get_nodes_glyph(shape='diamond', coords=coords)

    hull_circ = get_nodes_glyph(shape='circle', coords=coords)
    # custom patch
    #hull_patch = Patch(x='lon', y='lat', fill_color='color', 
    #                fill_alpha='alpha', line_color='color')  
    hull_patch = Patches(xs='xs', ys='ys', fill_color='color', 
                    fill_alpha='alpha', line_color='color', line_alpha=0.75,
                    line_width=4)   

    return paths_seg, paths_seg_sec, paths_line, paths_line_sec, \
            target_shape, target_text, sources_shape, \
            sources_text, crit_shape, crit_text, missing_shape, missing_text, \
            subgraph_shape, subgraph_text, diff_shape, diff_text, \
            rect_bin, rect_cum, hull_circ, hull_patch, risk_shape
            

###############################################################################
### histograms
###############################################################################
def init_histo(plot_width=1200, plot_height=300, x_range=None, y_range=None,
               logo=None, title='Movement Histogram'):
    '''Initialize histogram plot'''
    
    if x_range == None:
        x_range = DataRange1d()#start=0.0, rangepadding=0.1)#FactorRange()#DataRange1d()
    if y_range == None:
        y_range = DataRange1d()#start=-0.1, rangepadding=0.05)

    plot = Plot(x_range=x_range, y_range=y_range,
                 plot_width=plot_width, plot_height=plot_height)#, logo=None)#,
                 #title=title)#,
    plot.title.text=title
                 #h_symmetry=False, v_symmetry=False)#, min_border=0)
    #plot = bk.figure(plot_width=plot_width, plot_height=plot_height, 
    #        y_range=y_range, x_range=x_range)
    plot.add_tools( #PanTool(), WheelZoomTool(), BoxZoomTool(), 
                 SaveTool(), ResetTool())  
                 
    xaxis = LinearAxis(axis_label='Time (H)')
    yaxis = LinearAxis()
    plot.add_layout(xaxis, 'below')
    plot.add_layout(yaxis, 'left')
    plot.add_layout(Grid(dimension=0, ticker=xaxis.ticker))
    plot.add_layout(Grid(dimension=1, ticker=yaxis.ticker))

    # add hover
    hover_tableh = [("Bin",  "@x_str"),("Count", "@y"),]
    hoverh = HoverTool(tooltips=hover_tableh)    
    plot.add_tools(hoverh)
    if not logo:
        plot.toolbar.logo = None

    return plot
    
    
###############################################################################
def bar_chart(pb, x_arr, y_arr, legend=None, \
                color='green', alpha=0.6, width=0.99):

    x=np.asarray(x_arr)
    y=np.asarray(y_arr)
    # create a columndatasource
    source = bk.ColumnDataSource(data=dict(
            x=x,
            y=y,
            y_mid=y/2.#[y/2 for y in y_arr]
            ))   
    
    pb.rect(x=source.data['x'], y=source.data['y_mid'],  \
        height=source.data['y'], \
        width=width, color=color, alpha=alpha, legend=legend)  
    
    return pb


###############################################################################
def plot_histo(bin_str, binc_arr, cumc_arr, title='Histogram', \
                plot_width=1200, plot_height=300, ymax=None):
    '''Create binned and cumulative histograms
    If y=None, just use number of counts per xbin'''
    
    TOOLS="pan,wheel_zoom,box_zoom,reset,undo,redo,hover,save"
    #from bokeh.charts import Bar, Area, Step, Histogram     
    color_dic, alpha_dic = define_colors_alphas()
    
     # set y_max
    if ymax:
        plot_y_max = ymax
    else:
        if len(cumc_arr) > 0:
            plot_y_max = 1.1*np.max(cumc_arr)
        else:
            plot_y_max = 1.
            
    # create figure
    # make sure bin_str is a list!  (bokeh can't handle np arrays for range)
    pb = bk.figure(plot_width=plot_width, plot_height=plot_height, tools=TOOLS, \
            y_range=[0, plot_y_max], x_range=list(bin_str))#title=title, 
    pb.title.text = title
    # add cumulative plot
    # make sure bin_str is a list!
    pb = bar_chart(pb, list(bin_str), cumc_arr, color=color_dic['histo_cum_color'],\
            legend='Cumulative', alpha=0.5, width=0.99)  
    # add bin plot              
    pb = bar_chart(pb, list(bin_str), binc_arr, color=color_dic['histo_bin_color'], \
            legend='Binned', alpha=0.6, width=0.99)    
    pb.legend.orientation = 'vertical'#'top_left'
    # optional: rotate axis labels
    #bk.xaxis().major_label_orientation = np.pi/4   # radians, "horizontal", "vertical", "normal"
    # add hover
    hover_table = [("Bin",  "@x"),("Count", "@height"),]
    hover = pb.select(dict(type=HoverTool))
    hover.tooltips = hover_table 
    
    return pb


###############################################################################
### misc
###############################################################################
def add_legend():
    '''Add legend in separate pane'''
    pass

###############################################################################        
def set_routes_title(endnodes, target=None, skipnodes=[], use_hull=False):
    '''set title'''
    if target:
        histo_title = 'Number of Units Encountered by Node ' + str(target) + ' Over Time'
        # histo_title = 'Number of Forces Encountered by Node ' + str(target) + ' Over Time'
        if len(skipnodes) > 0:
            title = 'Paths Between Target and ' + str(len(endnodes)) + \
                ' Nodes of Interest, Skip Augmented Nodes'    
        else:
            title = 'Paths Between Target and ' + str(len(endnodes)) + \
                ' Nodes of Interest, Include All Nodes' 
    else:
        histo_title = 'Median Path Length Histogram Between Nodes'
        if len(skipnodes) > 0:
            title = 'Paths Between ' + str(len(endnodes)) + \
                ' Nodes of Interest, Skip Augmented Nodes'         
        else:
            title = 'Paths Between ' + str(len(endnodes)) + \
            ' Nodes of Interest, Include All Nodes'   
    if use_hull:
        histo_title = 'Number of Units Encountered by Node ' + str(target) + ' Over Time'
        # histo_title = 'Number of Forces Encountered by Node ' + str(target) + ' Over Time'
        if len(skipnodes) > 0:
            title = 'Area Encompassed by Source Node Over Time'#, Skip Augmented Nodes'         
        else:
            title = 'Area Encompassed by Source Node Over Time'#, Include All Nodes'         
        
    # centrality     
    centr_title = title + ' (Subgraph Centrality)'

    return title, centr_title, histo_title


###############################################################################
def add_line_hover(p, renderers=[]):
    '''Add hover to line glyph
    https://stackoverflow.com/questions/38304753/multi-line-hover-in-bokeh
    esource.data.keys()                                                                                                           
       dict_keys(['elat0', 'elon0', 'elat1', 'elon1', 'ex0', 'ey0', 'ex1', 
       'ey1', 'emx', 'emy', 'ewidth', 'ealpha', 'eid', 'ecolor', 'elabel', 
       'elx0', 'ely0', 'name', 'length', 'lanes', 'speed_mph'])
    '''
    
    hover_table = [
            ("Road ID", "@name"),
            ("Length (m)", "@length{0.0}"),
            #("ID", "@eid"),
            ("Speed Limit (mph)", "@speed_mph{0.0}"),
            ("Traffic Speed (mph)", "@speed2{0.0}")
            ]
    #print ("hover_table:", hover_table)
    # add tools 
    if len(renderers) > 0:
        hover = HoverTool(tooltips=hover_table, renderers=renderers)#, line_policy=line_policy) 
        p.add_tools(hover)
    return p


###############################################################################
def add_hover_save(p, htmlout=None, show=False, add_hover=True, 
                   renderers=[], verbose=False):#line_policy='none'):
    '''Add generic hover_table, and save output'''

    t0 = time.time()
    if add_hover:
        # add hover
        hover_table = \
            [
                #("index", "$index"),
                #("(x,y)", "($x, $y)"),
                #("degree", "@ndeg"),
                #("(lat,lon)", "($y, $x)"),
                ("Node Name",  "@name"),
                ("(lat, lon)", "(@lat, @lon)"),
                #("Count", "@count"),
                #("Value", "@val"),
                #("Nearest OSM Node", "@nearest"),
            ]
        # add tools 
        if len(renderers) > 0:
            hover = HoverTool(tooltips=hover_table, renderers=renderers)#, line_policy=line_policy) 
        else:
            hover = HoverTool(tooltips=hover_table)
        p.add_tools(hover)   
    t1 = time.time()
    if verbose:
        print ("Time to add hover:", t1 - t0, "seconds")
    
    #bk.output_notebook()   # if within IPython notebook
    if htmlout:
        #####################
        #https://github.com/bokeh/bokeh/issues/3671
        # shockingly, bokeh balloons file sizes by default
        bk.reset_output()        
        #####################
        
        #https://groups.google.com/a/continuum.io/forum/#!topic/bokeh/abjay8M22Q0
        #curstate().autoadd = False 
        #doc = curdoc()#set_curdoc(curdoc())
        #doc.clear()
        #doc.add_root(p)
        #mydoc = Document()
        #mydoc.add_root(p)
        
        bk.output_file(htmlout)
        bk.save(obj=p)
        
    t2 = time.time()
    if verbose:
        print ("Time to save:", t2 - t1, "seconds")

    if show:
        bk.show(p)       
    t3 = time.time()
    if verbose:
        print ("Time to show :", t3 - t2, "seconds")

    if verbose:
        print ("Time to add hover and save:", time.time() - t0, "seconds")
    return p#, hover
 

###############################################################################
def define_colors_alphas():
    '''
    Define colors for plotting
    https://www.w3schools.com/colors/colors_names.asp
    '''
    # colors
    color_dic = \
    {
     
        # categories of objects
        'Bus':                      'blue',
        'Truck':                    'firebrick',
        'Small_Vehicle':            'mediumseagreen',
        'Car':                      'mediumseagreen',
        'building':                 'blue',
        'Private_Boat':             'gray',
        'Medium_Ship':               'gray',
        # computed colors
        'goodnode_color':           'blue',
        'goodnode_aug_color':       'dodgerblue',#'cornflowerblue',
        'badnode_color':            'firebrick',#'crimson',
        'badnode_aug_color':        'red',
        'diff_node_color':          'purple',
        'diff_node_color_sec':      'lightgreen',
        'missing_node_color':       'chartreuse',#'fuchsia',
        'compute_path_color_good':  'aqua', #'mediumseagreen'
        'compute_path_color_bad':   'magenta',#'darkmagenta',#'aqua',#'mediumseagreen'
        'compute_path_color':       'lawngreen',#'aqua',#'chartreuse',#'darkmagenta',#'aqua',#'mediumseagreen',#'mediumvioletred',
        'crit_node_color':          'orange',#'darkorchid','mediumseagreen',
        'source_color':             'mediumvioletred',
        'target_color':             'green',
        'spread_seen_color':        'maroon',
        'spread_new_color':         'mediumvioletred',
        'histo_bin_color':          'teal',#'purple',
        'histo_cum_color':          'slateblue',#'orange'
        'node_centrality_color':    'lime',#'teal'#'lightgreen'
        'overlap_node_color':       'darkorchid',
        'risk_color':               'springgreen',

        # osm edges
        'motorway':                 'darkred',
        'trunk':                    'tomato',
        'primary':                  'orange',
        'secondary':                'gold',
        'tertiary':                 'yellow',
        'bridge':                   'pink',

        # speed colors
        0:                          'lightyellow',
        5:                          'lightyellow',
        10:                         'lightyellow',
        15:                         'yellow',
        20:                         'yellow',
        25:                         'gold',
        30:                         'gold',
        35:                         'orange',
        40:                         'orange',
        45:                         'tomato',
        50:                         'tomato',
        55:                         'firebrick',
        60:                         'firebrick',
        65:                         'darkred',
        70:                         'darkred',

        # traffic colors (1 means no traffic, 0 means all the traffic)
        # https://www.rapidtables.com/web/color/purple-color.html
        '0.0':                      'darkslategray',
        '0.2':                      'indigo',
        '0.4':                      'purple',
        '0.6':                      'blueviolet',
        '0.8':                      'mediumorchid',
        '1.0':                      'lavender',

        # raw color
        'raw_edge':                 'darkgray',

        # osm nodes
        'intersection':             'gray',
        'endpoint':                 'gray',
        'midpoint':                 'gray',  # 'black'
        'start':                    'green',
        'end':                      'red'
    }

    # opacity
    alpha_dic = {
        'osm_edge':                 0.4,
        'osm_node':                 0.15,
        'gdelt':                    0.6,
        'aug':                      0.35,
        'end_node':                 0.6,
        'crit_node':                0.7,
        'missing_node':             0.6,
        'target':                   0.7,
        'compute_paths':            0.6,
        'compute_paths_sec':        0.3,
        'centrality':               0.5,
        'histo_bin':                0.5,
        'histo_cum':                0.6,
        'label_slight':             0.5,
        'label_general':            0.7,
        'label_bold':               0.9,
        'hull':                     0.025,  # #4.5, #0.025,
        'diff':                     0.5,
        'risk':                     0.65,
        'contours':                 0.65,
        'force_proj':               0.2
    }

    return color_dic, alpha_dic
