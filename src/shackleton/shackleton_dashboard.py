# -*- coding: utf-8 -*-
"""
Created on Tue Jul  7 17:01:31 2015

@author: avanetten

To add new features, search for _risk, and duplicate code

To create the environment:
cd /Users/ave/projects/geodesic/shackleton
# conda env remove -n shackleton
conda env create --file environment.yml
conda activate shackleton

To run:
conda activate shackleton
bokeh serve --show src/shackleton/shackleton_dashboard.py --args \
    /Users/ave/projects/geodesic/shackleton/data/test_imagery/input/AOI_10_Dar_Es_Salaam_PS-RGB_COG_clip_final.tif \
    /Users/ave/projects/geodesic/shackleton/results/test0/yoltv5/AOI_10_Dar_Es_Salaam_PS-MS_COG_clip_3857.geojson \
    /Users/ave/projects/geodesic/shackleton/results/test0/cresi/AOI_10_Dar_Es_Salaam_PS-MS_COG_clip_3857.gpickle

Notes:
buses are good (+1) trucks are bad (-1), cars are neutral (0)

"""

# import shackleton code
from utils import (distance, latlon_to_wmp, wmp_to_latlon)
import bokeh_utils
import graph_funcs
import graph_utils
import utils

# for tile server
from localtileserver import TileClient
from bokeh.tile_providers import CARTODBPOSITRON, get_provider

import os
import sys
import time
import pickle
import random
import networkx as nx
import numpy as np
import pandas as pd
import bokeh.plotting as bk
from bokeh.plotting import figure, curdoc, show
from bokeh.layouts import widgetbox, column, row
from bokeh.models.glyphs import (Square, Circle, InvertedTriangle,
                                 Segment, Patch, Diamond, Text, Rect, 
                                 Patches, MultiLine)
from bokeh.models import (ColumnDataSource, DataRange1d,
                          TapTool,  CheckboxGroup, Button,
                          RadioButtonGroup, RadioGroup, Dropdown,
                          Legend, Toggle, CheckboxButtonGroup, Paragraph,
                          Range1d, LinearAxis, HoverTool, PanTool,
                          PolySelectTool, LassoSelectTool, WheelZoomTool,
                          Plot, GMapPlot, GMapOptions, BoxZoomTool,
                          BoxSelectTool, Grid, Panel, Tabs,
                          SaveTool, ResetTool, UndoTool, RedoTool,
                          OpenURL, TextInput, Select, Slider,
                          Panel, Tabs, PolyDrawTool,
                          PreText, Div,
                          WMTSTileSource, BBoxTileSource,
                          )

#############
src_path = os.path.dirname(os.path.realpath(__file__))
root_path = os.path.dirname(src_path)
sys.path.extend([src_path])

#############
# input arguments
IMAGE_PATH = sys.argv[1]
YOLT_PATH = sys.argv[2]
CRESI_PATH = sys.argv[3]
#############

# SETUP DEFAULTS
#############
title = 'Shackleton Dashboard' + ' - ' + os.path.basename(IMAGE_PATH)
global_dic = utils.global_vars()
color_dic, alpha_dic = bokeh_utils.define_colors_alphas()

# webgl supposedly speeds things up, but in this example it drastically slows down the server
output_backend = 'canvas'  # default
# output_backend = 'webgl'   # uses GPU, on my macbook pro this clobbers performance

# http://bokeh.pydata.org/en/latest/docs/user_guide/tools.html#controlling-level-of-detail
# lod_threshold = None displays all points when panning and zooming (sloooowwwww)
lod_threshold =  None
# lod_threshold = 100

projection = 'EPSG:3857'
coords = 'wmp'  # 'latlon'
edge_weight = 'Travel Time (h)'  # 'distance'  # 'Travel Time (h)'
map_background = 'spacenet'
plot_width = 900
histo_height = 240
panel_width_gui = 220
panel_width = 320

# notional bbox settings
df_bbox, source_bbox = None, None
max_node_dist_m = 50  

download_new_osm = False
network_type = 'drive'
to_crs = {'init': 'epsg:3857'}  # None
travel_time_s_key = 'travel_time'
travel_time_key = 'Travel Time (h)'
travel_time_key2 = 'Travel Time (h) Traffic'
osm_road_type_key = 'highway'
node_set_name = 'alliednodes'
# set plot bounds
# set bounds? https://bokeh.pydata.org/en/0.12.1/docs/reference/models/ranges.html
# plot_xrange_bounds = 'auto'  # None
# plot_yrange_bounds = 'auto'  # None
plot_xrange_min = 100 # 200
plot_yrange_min = 100 # 200

### YOLT PROPERTIES
min_YOLT_prob = 0.075
max_dist_m = 5   # max distance vehicle can be from road
dist_buff = 80  # radius to look for nodes when finding nearest edge
concave_alpha = 0.01
line_width_mult = 0.15

r_m = 5   # distance to extrapolate control values
binsize = 0.01   # time binning in hours
toolbar_location = "right" # None  # turn off toolbars?
init_page_status_para = Paragraph(text='', width=1200, height=60)


####################
# Create the plot

# # First, create a tile server from local raster file
client = TileClient(IMAGE_PATH)
raster_provider = WMTSTileSource(url=client.get_tile_url(client=True))
bounds = client.bounds(projection=projection)
print("Image bounds:", bounds)
basemap = get_provider(CARTODBPOSITRON)

# plot
plot = figure(plot_width=plot_width, plot_height=int(0.7*plot_width),
              # x_range=(bounds[2], bounds[3]), y_range=(bounds[0], bounds[1]),
              x_range=Range1d(), y_range=Range1d(),
              x_axis_type="mercator", y_axis_type="mercator", tools='',
              output_backend=output_backend,
              toolbar_location=toolbar_location,
              lod_threshold=lod_threshold)
              
# add map layers
plot.add_tile(basemap)
plot.add_tile(raster_provider)

# turn off labels?
# https://discourse.bokeh.org/t/how-speed-up-zoom-pan-in-layouts-with-many-plots/7305/6
# plot.axis.visible = False
# p.xaxis[0].formatter = PrintfTickFormatter(format="%1.3f")
# p.yaxis[0].formatter = PrintfTickFormatter(format="%1.3f")
 
# histo plot           
plot_histo = figure(plot_width=plot_width, plot_height=histo_height,
                    x_range=Range1d(), y_range=Range1d(),
                    tools='',
                    output_backend="webgl")
plot_histo.xaxis.axis_label = edge_weight

# global variales - init to null:
osm, node_histogram, ntype_dic, X_dic, G0, G, G_osm, \
    ecurve_dic, kd_idx_dic, kdtree, df_bbox, \
    g_node_props_dic, cat_value_counts, source_bbox, auglist, filenames = None, None, None,\
    None, None, None, None, None, None, None, None, None, None, None, None, None
outroot, outdir, graphfile, queryfile, html_raw, html_ref_straight, \
    html_ref, html_bbox, html_aug, bboxfile = None, None, None, None,\
    None, None, None, None, None, None
ecurve_dic_plot = None
cds_good_aug, cds_bad_aug = None, None
xmin, xmax, ymin, ymax = 0, 0, 0, 0
road_glyph = ''
is_edge_hover_displayed = False  # switch to hover over roads
bounds_dict = {}
edge_update_dict = {}


#############
# DASHBOARD DISPLAY OPTIONS
show_mst = False
show_evac = False
show_hulls = False
show_overlap = False
show_risk = False
show_contours = False
show_bbox = False
show_aug = False
show_force_proj = False
show_target = False
show_crit = True
show_sources = True
compute_subgraph_centrality = False
compute_secondary_routes = False
concave = False
goodroutes = True
skipnodes = []
end_nodes = []
auglist = []
target = ''
#############


#############
# init global sources to empty
Gnsource = bokeh_utils.set_nodes_source_empty()
Gesource = bokeh_utils.set_edge_source_empty()
Gnsource_osm = bokeh_utils.set_nodes_source_empty()
Gesource_osm = bokeh_utils.set_edge_source_empty()
esource = bokeh_utils.set_edge_source_empty()
esourcerw = bokeh_utils.set_edge_source_empty()
source_target = bokeh_utils.set_nodes_source_empty()
source_sourcenodes = bokeh_utils.set_nodes_source_empty()
source_crit = bokeh_utils.set_nodes_source_empty()
source_missing = bokeh_utils.set_nodes_source_empty()
source_histo_cum = bokeh_utils.set_rect_source_empty()
source_histo_bin = bokeh_utils.set_rect_source_empty()
source_subgraph_centrality = bokeh_utils.set_nodes_source_empty()
source_hull_node = bokeh_utils.set_nodes_source_empty()
source_hull_patch = bokeh_utils.set_hull_patch_source_empty()
source_hull_source = bokeh_utils.set_nodes_source_empty()
source_overlap = bokeh_utils.set_nodes_source_empty()
source_risk = bokeh_utils.set_nodes_source_empty()
source_contour = bokeh_utils.set_edge_source_empty()
source_force_proj = bokeh_utils.set_nodes_source_empty()

# init plot sources to empty
Gnsource_plot = bokeh_utils.set_nodes_source_empty()
Gesource_plot = bokeh_utils.set_edge_source_empty()
Gnsource_plot_osm = bokeh_utils.set_nodes_source_empty()
Gesource_plot_osm = bokeh_utils.set_edge_source_empty()
nsource_plot = bokeh_utils.set_nodes_source_empty()
esource_plot = bokeh_utils.set_edge_source_empty()
esourcerw_plot = bokeh_utils.set_edge_source_empty()
source_bbox_plot = bokeh_utils.set_bbox_source_empty()
source_bbox_text_plot = bokeh_utils.set_bbox_source_empty()
cds_good_aug_plot = bokeh_utils.set_nodes_source_empty()
cds_bad_aug_plot = bokeh_utils.set_nodes_source_empty()
source_target_plot = bokeh_utils.set_nodes_source_empty()
source_target_text_plot = bokeh_utils.set_nodes_source_empty()
source_sourcenodes_plot = bokeh_utils.set_nodes_source_empty()
source_sourcenodes_text_plot = bokeh_utils.set_nodes_source_empty()
source_crit_plot = bokeh_utils.set_nodes_source_empty()
source_crit_text_plot = bokeh_utils.set_nodes_source_empty()
source_missing_plot = bokeh_utils.set_nodes_source_empty()
source_missing_text_plot = bokeh_utils.set_nodes_source_empty()
source_subgraph_centrality_plot = bokeh_utils.set_nodes_source_empty()
source_subgraph_centrality_text_plot = bokeh_utils.set_nodes_source_empty()
source_hull_node_plot = bokeh_utils.set_nodes_source_empty()
source_hull_patch_plot = bokeh_utils.set_hull_patch_source_empty()
source_overlap_plot = bokeh_utils.set_nodes_source_empty()
source_overlap_text_plot = bokeh_utils.set_nodes_source_empty()
source_risk_plot = bokeh_utils.set_nodes_source_empty()
source_risk_text_plot = bokeh_utils.set_nodes_source_empty()
source_contour_plot = bokeh_utils.set_edge_source_empty()
source_histo_cum_plot = bokeh_utils.set_rect_source_empty()
source_histo_bin_plot = bokeh_utils.set_rect_source_empty()
source_force_proj_plot = bokeh_utils.set_nodes_source_empty()
#############

#############
# CREATE DOCUMENT?
document = curdoc()
#############


###############################################################################
# Map Widgets
###############################################################################

##############
# labels
text = 'Select Options'
blank_0 = Paragraph(text='', width=panel_width, height=20)
blank_1 = Paragraph(text='', width=panel_width, height=20)
blank_2 = Paragraph(text='', width=panel_width, height=20)
blank_3 = Paragraph(text='', width=panel_width, height=20)
blank_4 = Paragraph(text='', width=panel_width, height=20)
blank_5 = Paragraph(text='', width=panel_width, height=20)
blank_6 = Paragraph(text='', width=panel_width, height=20)
blank_7 = Paragraph(text='', width=panel_width, height=20)
blank_8 = Paragraph(text='', width=panel_width, height=20)
blank_9 = Paragraph(text='', width=panel_width, height=20)
blank_10 = Paragraph(text='', width=panel_width, height=20)
blank_100 = Paragraph(text='', width=panel_width, height=15)
edge_para0 = Paragraph(text='ROAD NETWORK', width=panel_width, height=20)
route_para0 = Paragraph(text='ANALYTICS', width=panel_width, height=20)
route_para1 = Paragraph(text='ROUTE OPTIONS', width=panel_width, height=20)
route_para2 = Paragraph(text='NODE DISPLAY OPTIONS', width=panel_width, height=20)
route_para3 = Paragraph(text='COMPUTE EXTERNALLY',
                        width=panel_width, height=20)


#############
def update_target(target, size_mult=2):
    global source_target, show_hulls
    if show_hulls:
        label = ['Source']
    else:
        label = ['Target']
    new_source = bokeh_utils.set_nodes_source(G, [target],
                                              size=size_mult * global_dic['maxS'],
                                              color=color_dic['target_color'],
                                              fill_alpha=alpha_dic['target'], shape='invertedtriangle',
                                              label=label, count=[], val=['Target'],
                                              # lat=G.nodes[target]['lat'],
                                              # lon=G.nodes[target]['lon'],
                                              # osm=target,
                                              name=label)
    source_target.data = dict(new_source.data)
    # source_target.data = new_source.data
#############


###############################################################################
# edge weight radio group
radio_group_edge_weight = RadioGroup(
    labels=["Length", "Speed Limit", "Traffic Speed"],
    active=1, name='Select Plot Options')
def radio_group_handler_edge_weight(active):
    # need to update skipnodes too
    global edge_weight
    # detault to empty datasources
    if active == 0:
        edge_weight = 'Travel Time (h) default'
    elif active == 1:
        edge_weight = 'Travel Time (h)'
    elif active == 2:
        edge_weight = 'Travel Time (h) Traffic'

    if show_mst or show_evac:
        update_routes()

radio_group_edge_weight.on_click(radio_group_handler_edge_weight)


###############################################################################
# show bbox and aug points
# button_type:  primary = blue
#               success = green,
#               warning = yellow
#               danger = red
#               link = blue text, no border
bbox_group = CheckboxButtonGroup(labels=["Find Vehicles"],
                                     active=[],
                                     name='Select Plot Options')  # , type='default')
vehicle_para = Div(text='', #style=div_style_dict,
    width=int(0.25*panel_width), height=56)
    
def bbox_group_handler(active):
    print("Update plot options")
    global show_bbox, source_bbox_plot, auglist, source_bbox, cat_value_counts
    [ngood, nbad, ngood_aug, nbad_aug, cds_good_aug, cds_bad_aug] = auglist
    if 0 in active:
        source_bbox_plot.data = dict(source_bbox.data)
        show_bbox = True
        vehicle_para.text = str(cat_value_counts.to_dict())[1:-1].replace("'", '').replace(",", '')
    else:
        show_bbox = False
        vehicle_para.text = ''
        source_bbox_plot.data = dict(bokeh_utils.set_bbox_source_empty().data)
bbox_group.on_click(bbox_group_handler)
    

###############################################################################
bbox_aug_group = CheckboxButtonGroup(labels=[#"Boxes", 
                                             "Augment",
                                             "Target",
                                              "Projection"
                                              ],
                                     active=[],  # [0],
                                     name='Select Plot Options')  # , type='default')
def bbox_aug_group_handler(active):
    print("Update plot options")
    verbose = False
    global show_aug, show_force_proj, source_force_proj_plot, \
        source_bbox_plot, cds_good_aug_plot,\
        cds_bad_aug_plot, auglist, \
        source_bbox, cds_good_aug, cds_bad_aug, source_force_proj, \
        target, show_target, source_target, source_target_plot

    [ngood, nbad, ngood_aug, nbad_aug, cds_good_aug, cds_bad_aug] = auglist

    if 0 in active:
        cds_good_aug_plot.data = dict(cds_good_aug.data)
        cds_bad_aug_plot.data = dict(cds_bad_aug.data)
        show_aug = True
    else:
        cds_good_aug_plot.data = dict(bokeh_utils.set_nodes_source_empty().data)
        cds_bad_aug_plot.data = dict(bokeh_utils.set_nodes_source_empty().data)
        show_aug = False

    if 1 in active:
        show_target = True
        update_target(target)
        source_target_plot.data = dict(source_target.data)
    else:
        show_target = False
        source_target_plot.data = dict(bokeh_utils.set_nodes_source_empty().data)

    if 2 in active:
        source_force_proj_plot.data = dict(source_force_proj.data)
        if verbose:
            print("source_force_proj.data:", source_force_proj.data)
            print("source_force_proj_plot.data:", source_force_proj_plot.data)
        show_force_proj = True
    else:
        source_force_proj_plot.data = dict(bokeh_utils.set_nodes_source_empty().data)
        show_force_proj = False

bbox_aug_group.on_click(bbox_aug_group_handler)


###############################################################################
def update_routes():

    global esource, esourcerw, source_target, source_sourcenodes, source_crit,\
        source_missing, source_histo_cum, source_histo_bin, \
        source_subgraph_centrality, auglist, binsize, \
        compute_subgraph_centrality, plot, plot_histo, edge_weight, \
        goodroutes

    # set target
    if show_evac:
        target0 = target
    else:
        target0 = None

    print("Updating routes, target:", target)
    print("compute_subgraph_centrality:", compute_subgraph_centrality)
    #print ("help(graph_funcs.get_route_sources):", help(graph_funcs.get_route_sources))
    esource, esourcerw, source_target, \
        source_sourcenodes, source_crit, \
        source_missing, source_histo_cum, source_histo_bin, \
        source_subgraph_centrality = graph_funcs.get_route_sources(
            G, end_nodes, auglist,
            g_node_props_dic, ecurve_dic=ecurve_dic_plot,
            target=target0,
            skipnodes=skipnodes, goodroutes=goodroutes,
            compute_secondary_routes=compute_secondary_routes,
            compute_subgraph_centrality=compute_subgraph_centrality,
            binsize=binsize,
            edge_weight=edge_weight, crit_perc=None)

    # update *_plot data (skip datasources with switches turned off)
    esource_plot.data = dict(esource.data)
    if compute_secondary_routes:
        esourcerw_plot.data = dict(esourcerw.data)
        #print ("esourcerw_plot.data:", esourcerw_plot.data)
    if show_sources:
        source_sourcenodes_plot.data = dict(source_sourcenodes.data)
    if show_crit:
        source_crit_plot.data = dict(source_crit.data)
    if compute_subgraph_centrality:
        source_subgraph_centrality_plot.data = dict(source_subgraph_centrality.data)
    source_missing_plot.data = dict(source_missing.data)
    source_histo_cum_plot.data = dict(source_histo_cum.data)
    source_histo_bin_plot.data = dict(source_histo_bin.data)
    source_target_plot.data = dict(source_target.data)

    # update plot titles
    plot_title, centr_title, histo_title = \
        bokeh_utils.set_routes_title(end_nodes,
                                     target=target0,
                                     skipnodes=skipnodes)
    plot.title.text = title # plot_title
    plot_histo.title.text = histo_title
    # update histo plot range
    plot_histo.x_range.start = 0
    plot_histo.y_range.start = 0
    if len(source_histo_cum.data['y']) > 0:
        plot_histo.y_range.end = 1.2*np.max(source_histo_cum.data['y'])
    else:
        plot_histo.y_range.end = 1.0
    if len(source_histo_cum.data['x']) > 0:
        plot_histo.x_range.end = np.max(source_histo_cum.data['x']) + binsize/2
    else:
        plot_histo.x_range.end = 1.0
    print("plot_histo.x_range.end:", plot_histo.y_range.end)
    print("plot_histo.x_range.end:", plot_histo.x_range.end)


###############################################################################
def update_hulls():

    global esource, esourcerw, source_target, source_sourcenodes, source_crit,\
        source_missing, source_histo_cum, source_histo_bin, \
        source_subgraph_centrality, source_hull_node, source_hull_patch, \
        concave, binsize, source_hull_patch_plot, concave_alpha
    # reset sources to empty
    esource_plot.data = dict(bokeh_utils.set_edge_source_empty().data)
    esourcerw_plot.data = dict(bokeh_utils.set_edge_source_empty().data)
    source_target_plot.data = dict(bokeh_utils.set_nodes_source_empty().data)
    source_sourcenodes_plot.data = dict(bokeh_utils.set_nodes_source_empty().data)
    source_crit_plot.data = dict(bokeh_utils.set_nodes_source_empty().data)
    source_missing_plot.data = dict(bokeh_utils.set_nodes_source_empty().data)
    source_histo_cum_plot.data = dict(bokeh_utils.set_rect_source_empty().data)
    source_histo_bin_plot.data = dict(bokeh_utils.set_rect_source_empty().data)
    source_subgraph_centrality_plot.data = dict(bokeh_utils.set_nodes_source_empty().data)
    source_hull_node_plot.data = dict(bokeh_utils.set_nodes_source_empty().data)
    source_hull_patch_plot.data = dict(bokeh_utils.set_hull_patch_source_empty().data)

    print("Computing spread from:", target)
    source_histo_bin, source_histo_cum, source_hull_node, source_hull_patch, \
        source_hull_source = \
        graph_funcs.compute_hull_sources(G, target,
                                         source_bbox, skipnodes=set([]),
                                         binsize=binsize, concave=concave,
                                         concave_alpha=concave_alpha,
                                         coords=coords)

    source_histo_cum_plot.data = dict(source_histo_cum.data)
    source_histo_bin_plot.data = dict(source_histo_bin.data)
    source_target_plot.data = dict(source_hull_source.data)
    source_hull_node_plot.data = dict(source_hull_node.data)
    # print ("source_hull_node_plot.data", source_hull_node_plot.data
    source_hull_patch_plot.data = dict(source_hull_patch.data)
    print("source_hull_patch_plot.data:", source_hull_patch_plot.data)

    # update plot titles
    plot_title, centr_title, histo_title = \
        bokeh_utils.set_routes_title(end_nodes,
                                     target=target,
                                     skipnodes=skipnodes,
                                     use_hull=True)
    # plot.title.text = plot_title
    plot_histo.title.text = histo_title
    # update histo plot range
    plot_histo.x_range.start = 0
    plot_histo.y_range.start = 0
    if len(source_histo_cum.data['y']) > 0:
        plot_histo.y_range.end = 1.2*np.max(source_histo_cum.data['y'])
    else:
        plot_histo.y_range.end = 1.0
    if len(source_histo_cum.data['x']) > 0:
        plot_histo.x_range.end = np.max(source_histo_cum.data['x']) + binsize/2
    else:
        plot_histo.x_range.end = 1.0


###############################################################################
def update_risk():
    '''Update risk estimates.
    Also have the option of computing risk for all nodes, not just goodnodes
    or badnodes'''

    # ,end_nodes, goodroutes, skip_nodes
    global source_risk, ecurve_dic_plot, auglist, edge_weight, goodroutes

    [ngood, nbad, ngood_aug, nbad_aug, cds_good_aug, cds_bad_aug] = auglist

    skipnodes_r = []
    if goodroutes:
        node_set_name = 'alliednodes'
        sourcenodes = nbad
        ignorenodes = nbad_aug
        if len(skipnodes) > 0:
            skipnodes_r = ngood_aug
        targetnodes = ngood  # set to [] for all nodes

    else:
        node_set_name = 'adversarynodes'
        sourcenodes = ngood
        ignorenodes = ngood_aug
        if len(skipnodes) > 0:
            skipnodes_r = nbad_aug
        targetnodes = nbad  # set to [] for all nodes

    print("risk sourcenodes:", sourcenodes)
    print("risk targetnodes:", targetnodes)
    source_risk = graph_funcs.compute_risk(G, sourcenodes,
                                           g_node_props_dic,
                                           target_nodes=targetnodes,
                                           skipnodes=skipnodes_r,
                                           ignore_nodes=ignorenodes,
                                           weight=edge_weight,
                                           kdtree=kdtree,
                                           kd_idx_dic=kd_idx_dic,
                                           r_m=r_m,
                                           verbose=False)

    if len(skipnodes) > 0:
        title = 'Inferred Risk - ' + node_set_name + ' - skipnodes'
    else:
        title = 'Inferred Risk - ' + node_set_name

    return source_risk, title


##############
radio_labels_edges = [" Raw Imagery",
                      # " Import OSM Roads",
                      " Show CRESI Roads",
                      "  Infer Speed Limit",
                      "  Infer Traffic Speed",
                      "  Infer Congestion"]
radio_group_edges = RadioGroup(labels=radio_labels_edges, active=0,
                               name='Select Road Options')
def radio_group_handler_edges(active):
    global plot, Gesource, Gesource_osm, Gesource_plot, Gesource_plot_osm, \
        Gnsource, Gnsource_osm, Gnsource_plot, Gnsource_plot_osm, \
        road_glyph, is_edge_hover_displayed

    # default to osm off
    Gnsource_plot_osm.data = dict(bokeh_utils.set_nodes_source_empty().data)
    if len(G_osm.nodes()) > 0:
        Gesource_plot_osm.data['plot_color'] = len(Gesource_osm.data['speed_mph']) * ['#ffffff00'] 
    
    if active == 0:
        Gnsource_plot.data = dict(bokeh_utils.set_nodes_source_empty().data)
        Gesource_plot.data['plot_color'] = len(Gesource.data['speed_mph']) * ['#ffffff00'] 
    # elif active == 1:
    #     if len(G_osm.nodes()) > 0:
    #         Gnsource_plot_osm.data = Gnsource_osm.data
    #         Gesource_plot_osm.data['plot_color'] = Gesource_osm.data['raw_color']
    elif active == 1:
        Gnsource_plot.data = dict(Gnsource.data)
        Gesource_plot.data['plot_color'] = Gesource.data['raw_color']
    elif active == 2:
        Gnsource_plot.data = dict(Gnsource.data)
        Gesource_plot.data['plot_color'] = Gesource.data['ecolor']
    elif active == 3:
        Gnsource_plot.data = dict(Gnsource.data)
        Gesource_plot.data['plot_color'] = Gesource.data['ecolor2']
    elif active == 4:
        Gnsource_plot.data = dict(Gnsource.data)
        Gesource_plot.data['plot_color'] = Gesource.data['cong_color']
        Gesource_plot.data['speed_mph'] = Gesource.data['speed_mph']
        Gesource_plot.data['speed2'] = Gesource.data['speed2']
    else:
        pass

    # show line hover
    if active > 0 and not is_edge_hover_displayed:
    # if active > 0 and is_edge_hover_displayed:
        _ = bokeh_utils.add_line_hover(plot, renderers=[road_glyph])
        is_edge_hover_displayed = True

radio_group_edges.on_click(radio_group_handler_edges)


##############
# compute routes or hulls
radio_labels = ["None", 
                "Evacuation / Logistics", 
                "Min Spanning Tree",
                "MST Overlap", 
                "Inferred Risk",
                "Show Spread"]
radio_group_comp = RadioGroup(labels=radio_labels, active=0,
                              name='Select Plot Options')

def radio_group_handler_comp(active):
    # global end_nodes, auglist, g_node_props_dic, target, skipnodes, \
    #    goodroutes, compute_secondary_routes, compute_subgraph_centrality
    global target, show_mst, show_evac, show_hulls, show_target, \
        show_overlap, show_risk, plot, plot_histo
    # reset sources to empty
    esource_plot.data = dict(bokeh_utils.set_edge_source_empty().data)
    esourcerw_plot.data = dict(bokeh_utils.set_edge_source_empty().data)
    source_target_plot.data = dict(bokeh_utils.set_nodes_source_empty().data)
    source_sourcenodes_plot.data = dict(bokeh_utils.set_nodes_source_empty().data)
    source_crit_plot.data = dict(bokeh_utils.set_nodes_source_empty().data)
    source_missing_plot.data = dict(bokeh_utils.set_nodes_source_empty().data)
    source_histo_cum_plot.data = dict(bokeh_utils.set_rect_source_empty().data)
    source_histo_bin_plot.data = dict(bokeh_utils.set_rect_source_empty().data)
    source_subgraph_centrality_plot.data = dict(bokeh_utils.set_nodes_source_empty().data)
    source_hull_node_plot.data = dict(bokeh_utils.set_nodes_source_empty().data)
    source_hull_patch_plot.data = dict(bokeh_utils.set_hull_patch_source_empty().data)
    source_overlap_plot.data = dict(bokeh_utils.set_nodes_source_empty().data)
    source_risk_plot.data = dict(bokeh_utils.set_nodes_source_empty().data)

    # target_plot = None
    # show_routes = False
    show_mst = False
    show_overlap = False
    show_risk = False
    show_evac = False
    show_hulls = False
    show_target = False
    # plot.title.text = 'Road Network'
    plot_histo.title.text = 'Movement Histogram'

    if active == 0:
        print("No analytics shown")
        # reset sources to empty
        esource_plot.data = dict(bokeh_utils.set_edge_source_empty().data)
        esourcerw_plot.data = dict(bokeh_utils.set_edge_source_empty().data)
        source_target_plot.data = dict(bokeh_utils.set_nodes_source_empty().data)
        source_sourcenodes_plot.data = dict(bokeh_utils.set_nodes_source_empty().data)
        source_crit_plot.data = dict(bokeh_utils.set_nodes_source_empty().data)
        source_missing_plot.data = dict(bokeh_utils.set_nodes_source_empty().data)
        source_histo_cum_plot.data = dict(bokeh_utils.set_rect_source_empty().data)
        source_histo_bin_plot.data = dict(bokeh_utils.set_rect_source_empty().data)
        source_subgraph_centrality_plot.data = dict(bokeh_utils.set_nodes_source_empty().data)
        source_hull_node_plot.data = dict(bokeh_utils.set_nodes_source_empty().data)
        source_hull_patch_plot.data = dict(bokeh_utils.set_hull_patch_source_empty().data)

    elif active == 1:
        print("Compute Evacuation")
        # EVAC
        # if target_plot == None:
        #    target = target_ex0
        #target_plot = target
        show_evac = True
        #show_mst = False
        #show_hulls= False
        show_target = True
        update_routes()

    elif active == 2:
        print("Compute MST")
        # MST
        # if target == None:
        #    target = target_ex0
        #target_plot = None
        show_mst = True
        # show_evac=False
        # show_hulls=False
        #show_target = False
        update_routes()

    elif active == 3:
        print("Compute MST Overlap")
        show_overlap = True
        overlap_target = None
        if len(skipnodes) == 0:
            use_skipnodes = False
        else:
            use_skipnodes = True
        source_overlap = graph_funcs.route_overlap(None, G,
                                                   source_bbox,
                                                   auglist, g_node_props_dic, ecurve_dic=ecurve_dic_plot,
                                                   use_skipnodes=use_skipnodes,
                                                   show_plots=False,
                                                   target=overlap_target,
                                                   compute_secondary_routes=compute_secondary_routes)
        source_overlap_plot.data = dict(source_overlap.data)
        plot.title.text = 'Minimum Spanning Tree Overlap Between Good and Bad Nodes'

    elif active == 4:
        print("Compute Risk")
        show_risk = True
        source_risk, rtitle = update_risk()
        source_risk_plot.data = dict(source_risk.data)
        plot.title.text = rtitle
        # session.store_document(document)

    elif active == 5:
        print("Compute Spread")
        #target_plot = target
        #show_mst = False
        #show_evac = False
        show_hulls = True
        show_target = True
        update_hulls()
        plot.title.text = 'Spread'

radio_group_comp.on_click(radio_group_handler_comp)


##############
# radio group for nodes to route to/from
radio_group_endnodes = RadioGroup(labels=["Allied Nodes", "Adversary Nodes"],
                                  active=0, name='Select Plot Options')

def radio_group_handler_endnodes(active):
    # need to update skipnodes too
    global goodroutes, end_nodes, node_set_name, skipnodes, show_mst, \
        show_evac, auglist, source_risk
    [ngood, nbad, ngood_aug, nbad_aug, cds_good_aug, cds_bad_aug] = auglist
    # detault to empty datasources
    if active == 0:
        goodroutes = True
        node_set_name = 'alliednodes'
        end_nodes = ngood
        if len(skipnodes) > 0:
            skipnodes = nbad_aug
    elif active == 1:
        goodroutes = False
        node_set_name = 'adversarynodes'
        end_nodes = nbad
        if len(skipnodes) > 0:
            skipnodes = ngood_aug

    if show_mst or show_evac:  # if show_routes:
        update_routes()
    if show_risk:
        source_risk, rtitle = update_risk()
        source_risk_plot.data = dict(source_risk.data)
        plot.title.text = rtitle

radio_group_endnodes.on_click(radio_group_handler_endnodes)
##############

##############
# convex radio group
radio_group_convex = RadioButtonGroup(labels=["Convex", "Concave"],
                                      active=0, name='Select Plot Options')


def radio_group_handler_convex(active):
    # need to update skipnodes too
    global show_hulls, concave
    # detault to empty datasources
    if active == 0:
        concave = False
    elif active == 1:
        concave = True
    if show_hulls:
        update_hulls()

radio_group_convex.on_click(radio_group_handler_convex)
##############

##############
radio_group_skipnodes = CheckboxButtonGroup(labels=["Avoid OPFOR"],
                                            active=[], name='Select Plot Options')

def radio_group_handler_skipnodes(active):
    global goodroutes, skipnodes, show_mst, show_evac, auglist, \
        source_overlap, source_overlap_plot, source_risk, \
        source_risk_plot
    [ngood, nbad, ngood_aug, nbad_aug, cds_good_aug, cds_bad_aug] = auglist
    # detault to no skipnodes
    skipnodes = []
    for p in active:
        if p == 0:
            if goodroutes:
                skipnodes = nbad_aug
            else:
                skipnodes = ngood_aug
    # print ("active", active)
    # print ("skipnodes", skipnodes)
    if show_mst or show_evac:
        update_routes()

    if show_overlap:
        overlap_target = None
        if len(skipnodes) == 0:
            use_skipnodes = False
        else:
            use_skipnodes = True
        source_overlap = graph_funcs.route_overlap(None, G,
                                                   source_bbox,
                                                   auglist, g_node_props_dic,
                                                   use_skipnodes=use_skipnodes,
                                                   show_plots=False,
                                                   target=overlap_target,
                                                   compute_secondary_routes=compute_secondary_routes)
        source_overlap_plot.data = dict(source_overlap.data)
        # session.store_document(document)

    if show_risk:
        source_risk, rtitle = update_risk()
        source_risk_plot.data = dict(source_risk.data)
        plot.title.text = rtitle

radio_group_skipnodes.on_click(radio_group_handler_skipnodes)
##############



##############
# plot options
plot_options_group = CheckboxGroup(labels=["Source Nodes", 
                                           "Critical Nodes",
                                           #"Secondary Routes", 
                                           "Centrality",
                                           #"Control Contours"
                                           ],
                                   active=[0, 1], name='Select Plot Options')


def plot_options_group_handler(active):
    #print("plot_options_group_handler2: %s" % active)
    global show_sources, show_crit, compute_subgraph_centrality, \
        compute_secondary_routes, show_mst, show_evac, \
        source_overlap, source_overlap_plot, \
        show_contours, source_contour, source_contour_plot, \
        esourcerw

    show_sources, show_crit, compute_secondary_routes, compute_subgraph_centrality, show_contours = \
        False, False, False, False, False
    # set global vars
    for p in active:
        if p == 0:
            show_sources = True
        elif p == 1:
            show_crit = True
        #elif p == 2:
        #    compute_secondary_routes = True
        elif p == 2:
            compute_subgraph_centrality = True
        elif p == 4:
            show_contours = True

    # reset objects if needed
    if not show_sources:
        source_sourcenodes_plot.data = dict(bokeh_utils.set_nodes_source_empty().data)
    if not show_crit:
        source_crit_plot.data = dict(bokeh_utils.set_nodes_source_empty().data)
    # if not compute_secondary_routes:
    #    esourcerw_plot.data = bokeh_utils.set_edge_source_empty().data
    if not compute_subgraph_centrality:
        source_subgraph_centrality_plot.data = dict(bokeh_utils.set_nodes_source_empty().data)
    if not show_contours:
        source_contour_plot.data = dict(bokeh_utils.set_edge_source_empty().data)

    # update plots if showing routes
    if show_mst or show_evac:  # if show_routes:
        for p in active:
            if p == 0:
                source_sourcenodes_plot.data = dict(source_sourcenodes.data)
                # print ("source_sourcenodes.data:", source_sourcenodes.data)
            elif p == 1:
                source_crit_plot.data = dict(source_crit.data)
            # elif p == 2:
            #    pass
            #    # not sure why the line below breaks everything....!
            #    #esourcerw_plot.data = esourcerw.data
            elif p == 2:
                source_subgraph_centrality_plot.data = dict(source_subgraph_centrality.data)
        # only if computing subgraph_centrality or secondary, update data
        if 2 or 3 in active:
            update_routes()

    if show_overlap:
        # recompute overlap
        overlap_target = None
        if len(skipnodes) > 0:
            use_skipnodes = True
        else:
            use_skipnodes = False
        source_overlap = graph_funcs.route_overlap(None, G,
                                                   source_bbox,
                                                   auglist, g_node_props_dic,
                                                   use_skipnodes=use_skipnodes,
                                                   ecurve_dic=ecurve_dic_plot, show_plots=False,
                                                   target=overlap_target,
                                                   compute_secondary_routes=compute_secondary_routes)
        source_overlap_plot.data = dict(source_overlap.data)
        # session.store_document(document)

    if show_contours:
        remove_empties = True    # slow, but cleans up display
        use_aug = False
        mpl_plot = False
        res = 500
        theta0 = 0.1
        corr = 'linear'  # 'absolute_exponential'#'linear'
        nugget = 0.05
        polys, polyx, polyy, poly_colors = \
            graph_funcs.get_gauss_contours(source_bbox,
                                           auglist, G,
                                           remove_empties=remove_empties,
                                           theta0=theta0, res=res,
                                           mpl_plot=mpl_plot, use_aug=use_aug,
                                           corr=corr,
                                           nugget=nugget)
        source_contour = bk.ColumnDataSource(dict(
            xs=polyx,
            ys=polyy,
            color=poly_colors))
        source_contour_plot.data = dict(source_contour.data)
    else:
        pass

plot_options_group.on_click(plot_options_group_handler)
##############


##############
# plot options
sec_route_group = CheckboxButtonGroup(labels=["Secondary Routes"],
                                   active=[], name='Secondary Routes?')

def sec_route_group_handler(active):
    #print("plot_options_group_handler2: %s" % active)
    global compute_secondary_routes, esourcerw

    compute_secondary_routes = False
    # set global vars
    for p in active:
        if p == 0:
            compute_secondary_routes = True

    if not compute_secondary_routes:
        esourcerw_plot.data = dict(bokeh_utils.set_edge_source_empty().data)

    # update plots if showing routes
    if show_mst or show_evac:  # if show_routes:
        # only if computing subgraph_centrality or secondary, update data
        if 0 in active:
            update_routes()

sec_route_group.on_click(sec_route_group_handler)
##############


###############
## plot options

##############
# set max aug distance
#slider_dist.on_change('value', on_dist_change)
slider_aug_gui = Slider(title="Control Distance (m)",
                        start=r_m, end=10*r_m, value=r_m, step=r_m)

def on_rm_change(attr, old, new):
    print("Max Aug Dist (m)", new)
    global r_m, auglist, cds_good_aug, cds_bad_aug, cds_good_aug_plot, \
        cds_bad_aug_plot, skipnodes, goodroutes, show_mst, show_evac,\
        show_aug, auglist, df_bbox, \
        source_force_proj, source_force_proj_plot

    r_m = new
    # updsate auglist
    auglist = graph_utils.get_aug(G, df_bbox, kdtree, kd_idx_dic,
                                  node_size=5, r_m=r_m)

    [ngood, nbad, ngood_aug, nbad_aug, cds_good_aug0, cds_bad_aug0] = auglist
    # update cds
    cds_good_aug.data = dict(cds_good_aug0.data)
    cds_bad_aug.data = dict(cds_bad_aug0.data)
    # update force project cds
    len_tmp = len(source_bbox.data['Xmid_wmp'])
    source_force_proj.data['size'] = len_tmp * [r_m]
    #source_force_proj.data['size'] = len_tmp * [1000. * r_km]

    # plot
    if show_aug:
        cds_good_aug_plot.data = dict(cds_good_aug0.data)
        cds_bad_aug_plot.data = dict(cds_bad_aug0.data)
    if show_force_proj:
        source_force_proj_plot.data = dict(source_force_proj.data)

    # update skipnodes
    if len(skipnodes) > 0:
        if goodroutes:
            skipnodes = nbad_aug
        else:
            skipnodes = ngood_aug

    # update routes
    if (show_mst or show_evac) and len(skipnodes) > 0:
        update_routes()
    else:
        pass

slider_aug_gui.on_change('value', on_rm_change)
##############


##############
# set max aug distance
#slider_dist.on_change('value', on_dist_change)
slider_binsize = Slider(title="Time Bins (h)",
                        start=binsize, end=10*binsize, value=binsize, step=binsize)


def on_binsize_change(attr, old, new):
    print("Bin Size (h)", new)
    global binsize, show_hulls, show_mst, show_evac
    binsize = new
    if show_hulls:
        update_hulls()
    elif show_mst or show_evac:
        update_routes()

slider_binsize.on_change('value', on_binsize_change)
##############


### Core
###############################################################################
def load_data():
    # load data

    global osm, node_histogram, ntype_dic, X_dic, G0, G, G_osm, \
        ecurve_dic, kd_idx_dic, kdtree, \
        df_bbox, g_node_props_dic, cat_value_counts, \
        source_force_proj, r_m, \
        df_bbox, source_bbox, \
        auglist, ecurve_dic_plot, \
        straight_lines, \
        CRESI_PATH, \
        YOLT_PATH, \
        xmin, xmax, ymin, ymax, \
        init_page_status_para

    print("Loading graph pickle:", CRESI_PATH, "...")
    # init_page_status_para.text = 'Loading data...'
    G = nx.read_gpickle(CRESI_PATH)
    # the following should already be completed...
    # G = graph_init.create_osm_graph(pkl=CRESI_PATH,
    #                 network_type=network_type,
    #                 to_crs=to_crs,
    #                 speed_dict=speed_dict,
    #                 speed_key=speed_key,
    #                 travel_time_key=travel_time_key,
    #                 plot=False, verbose=True)

    # print some graph properties
    print("Num G_gt_init.nodes():", len(G.nodes()))
    print("Num G_gt_init.edges():", len(G.edges()))
    # print random node prop
    node_tmp = random.choice(list(G.nodes()))
    print(("G random node props:", node_tmp, ":", G.nodes[node_tmp]))
    # print random edge properties
    edge_tmp = random.choice(list(G.edges()))
    print("G random edge props:", edge_tmp, ":",
          G.edges[edge_tmp[0], edge_tmp[1], 0])

    # get graph extent
    xmin, ymin, xmax, ymax = graph_utils.get_G_extent(G)
    print("xmin, ymin, xmax, ymax:", xmin, ymin, xmax, ymax)
    
    # skip OSM for now
    G_osm = nx.DiGraph()
    # print("Get OSM Graph...")
    # lat0, lon0 = utils.wmp_to_latlon(xmin, ymin)
    # lat1, lon1 = utils.wmp_to_latlon(xmax, ymax)
    # bbox = [lat1, lat0, lon1, lon0]
    # print("  bbox:", bbox)
    # # if G creating G_osm fails, continue (if not connected to internet can't get OSM graph)
    # try:
    #     G_osm = graph_init.create_osm_graph(bbox=bbox, poly_shapely=None, pkl=None,
    #                  network_type='drive',
    #                  to_crs={'init': 'epsg:3857'},
    #                  speed_mph_key='speed_mph',
    #                  speed_mps_key='speed_m/s',
    #                  travel_time_s_key='travel_time',
    #                  travel_time_key='Travel Time (h)',
    #                  road_type_key='highway',
    #                  plot=False, simplify=True,
    #                  verbose=True)
    # except:
    #     G_osm = nx.DiGraph()

    # random bbox data?
    create_rand_data = False
    if create_rand_data:
        min_box_size, max_box_size = 20, 50
        N_boxes = 99
        print("Creating bbox_csv:", bbox_csv)
        # load in output of YOLT
        df_tmp = pd.read_csv(csv_part_loc)
        N_boxes_tmp = min(len(df_tmp), N_boxes)
        print("N boxes to create:", N_boxes_tmp)
        df_tmp = df_tmp.iloc[:N_boxes_tmp]

        # create random boxes
        x0s = np.random.uniform(xmin, xmax, N_boxes_tmp)
        y0s = np.random.uniform(ymin, ymax, N_boxes_tmp)
        widths = np.random.randint(
            min_box_size, max_box_size, size=N_boxes_tmp)
        heights = np.random.randint(
            min_box_size, max_box_size, size=N_boxes_tmp)
        x1s = x0s + widths
        y1s = y0s + heights

        # add wmp coords to dataframe
        df_tmp['Xmin_wmp'] = x0s
        df_tmp['Xmax_wmp'] = x1s
        df_tmp['Ymin_wmp'] = y0s
        df_tmp['Ymax_wmp'] = y1s

        # asssign a random value of 0 or 1
        df_tmp['Val'] = np.random.randint(0, 2, size=len(x0s))
        df_tmp['count'] = np.ones(len(x0s))
        df_tmp['num'] = np.ones(len(x0s))

        # write to file
        df_tmp.to_csv(bbox_csv)

    # create YOLT bbox df and source (update G as well...)
    G, df_bbox, source_bbox, g_node_props_dic, cat_value_counts = graph_utils.load_YOLT(
        YOLT_PATH,
        # bbox_csv,
        nearest_edge=True,
        G=G, categories=[], min_prob=min_YOLT_prob,
        scale_alpha_prob=True, max_dist_m=max_dist_m,
        dist_buff=dist_buff,
        verbose=False)
    print("df_bbox.iloc[0]:", df_bbox.iloc[0])
    print("source_bbox.data.keys():", source_bbox.data.keys())
    # print("source_bbox.data:", source_bbox.data)

    # ensure node coords contain latlon
    xnode_tmp = [data['x'] for _, data in G.nodes(data=True)]
    ynode_tmp = [data['y'] for _, data in G.nodes(data=True)]
    lats, lons = utils.wmp_to_latlon(xnode_tmp, ynode_tmp)
    for i, (node, data) in enumerate(G.nodes(data=True)):
        data['lat'] = lats[i]
        data['lon'] = lons[i]

    # get curve dic
    ecurve_dic = graph_utils.make_ecurve_dic(G)
    ecurve_dic_plot = ecurve_dic

    # get kd tree
    kd_idx_dic, kdtree = graph_utils.G_to_kdtree(G)

    # get aug
    print("\nCreating aug...")
    #r_m = 1000 * r_km
    print("r_m:", r_m)
    auglist = graph_utils.get_aug(G, df_bbox, kdtree, kd_idx_dic, r_m=r_m,
                                  special_nodes=set([]), node_size=6, 
                                  shape='circle', verbose=False)
    [ngood, nbad, ngood_aug, nbad_aug, cds_good_aug, cds_bad_aug] = auglist

    # get source that can be used to plot the extend of the force projection
    #print ("source_bbox.data['Xmid_wmp']:", source_bbox.data['Xmid_wmp'])
    len_tmp = len(source_bbox.data['Xmid_wmp'])
    source_force_proj = bk.ColumnDataSource(data=dict(
        x=source_bbox.data['Xmid_wmp'],
        y=source_bbox.data['Ymid_wmp'],
        color=source_bbox.data['color'],
        alpha=len_tmp * [alpha_dic['force_proj']],
        size=len_tmp * [r_m],
        #size=len_tmp * [1000. * r_km],
    ))
    
    #print ("source_force_proj.data:", source_force_proj.data)
    print("Data successfully loaded!")
    return


###############################################################################
def update_G():
    """
    Update G weights based on density of cars.
    """
    
    global G, edge_update_dict
    edge_update_dict = graph_utils.compute_traffic(G, verbose=False)
    G = graph_utils.update_Gweights(G, edge_update_dict,
                                    speed_key1='inferred_speed_mph',
                                    speed_key2='speed2',
                                    edge_id_key='uv',
                                    travel_time_key='Travel Time (h)',
                                    travel_time_key2='Travel Time (h) Traffic',
                                    travel_time_key_default='Travel Time (h) default',
                                    verbose=False)
    return


###############################################################################
def init_plot():
    '''Initial plot'''

    print("Create initial plots")
    global G, G_osm, plot, global_dic, map_background, Gesource, Gnsource, \
        Gnsource_osm, Gesource_osm, \
        Gesource_plot, Gnsource_plot, Gesource_plot_osm, Gnsource_plot_osm, \
        coords, bounds_dict, tabs, \
        plot_histo, source_histo_bin_plot, source_histo_cum_plot, \
        road_glyph, line_width_mult, title

    # Turn G into bokeh plot
    htmlout = None 
    show_line_hover = False  # We'll do this later
    show_road_labels = False
    show_nodes = True

    # Create plot properties
    constant_node_size = global_dic['gnode_size']
    # print( "constant_node_size:", constant_node_size)
    Gnsource = bokeh_utils.G_to_nsource(G, nsize_key='deg',
                                        constant_node_size=constant_node_size,
                                        ncolor_key='ntype', verbose=False)
    Gesource = bokeh_utils.G_to_esource(G, use_multiline=True,
                                        ewidth_key='inferred_speed_mph',
                                        ecolor_key='inferred_speed_mph',
                                        cong_key='congestion',
                                        width_mult=line_width_mult,
                                        verbose=False)
    _, glyph_node_list, bounds_dict = \
        bokeh_utils.cds_to_bokeh(Gnsource, Gesource, plot_in=plot, 
                               htmlout=htmlout, map_background=map_background,
                               show_nodes=show_nodes, show_road_labels=False,
                               plot_width=plot_width,  title='None',
                               add_glyphs=False, coords=coords, use_multiline=True,
                               show=False, verbose=True)
    
    # OSM
    if len(G_osm.nodes()) > 0:
        Gnsource_osm = bokeh_utils.G_to_nsource(G_osm, nsize_key='deg',
                                            constant_node_size=constant_node_size,
                                            ncolor_key='ntype', verbose=False)
        Gesource_osm = bokeh_utils.G_to_esource(G_osm, use_multiline=True,
                                            ewidth_key='inferred_speed_mph',
                                            ecolor_key='inferred_speed_mph',
                                            cong_key='congestion',
                                            width_mult=line_width_mult,
                                            verbose=False)
        _, glyph_node_list_osm, bounds_dict_osm = \
            bokeh_utils.cds_to_bokeh(Gnsource_osm, Gesource_osm, plot_in=plot, 
                                   htmlout=htmlout, map_background=map_background,
                                   show_nodes=show_nodes, show_road_labels=False,
                                   plot_width=plot_width,  title='None',
                                   add_glyphs=False, coords=coords, use_multiline=True,
                                   show=False, verbose=True)

 
    # Set gnsource_plot as gnsource 
    #  (actually, do this in radio_group_handler_edges())
    # Gnsource_plot.data = Gnsource.data
    
    # set gesource_plot as gesource, except for color and alpha
    Gesource_plot.data = dict(Gesource.data)
    Gesource_plot.data['plot_color'] = len(Gesource.data['ealpha']) * ['#ffffff00']   

    if len(G_osm.nodes()) > 0:
        Gesource_plot_osm.data = dict(Gesource_osm.data)
        Gesource_plot_osm.data['plot_color'] = len(Gesource_osm.data['ealpha']) * ['#ffffff00']                
                      
    #plot.x_range = x_range
    #plot.y_range = y_range
    plot.plot_height = bounds_dict['plot_height']
   
    # add items to plot?
    x_range = bounds_dict['x_range']
    y_range = bounds_dict['y_range']
    plot.x_range.start = x_range[0]
    plot.x_range.end = x_range[1]
    plot.y_range.start = y_range[0]
    plot.y_range.end = y_range[1]
    
    # set bounds? 
    plot.x_range.bounds = (bounds_dict['x_range'][0], bounds_dict['x_range'][1])
    plot.y_range.bounds = (bounds_dict['y_range'][0], bounds_dict['y_range'][1])
    #plot.x_range.bounds = (x_range[0], x_range[1])
    #plot.y_range.bounds = (y_range[0], y_range[1])
    plot.x_range.min_interval = plot_xrange_min
    plot.y_range.min_interval = plot_yrange_min

    # create plot?
    #  Can't create plot here without screwing up some interactivity,
    #  instead create a placeholder at the start and update it here
    # plot = figure(x_range=bounds_dict['x_range'],
    #              y_range=bounds_dict['y_range'],
    #              plot_width=plot_width,
    #              plot_height=bounds_dict['plot_height'],
    #             x_axis_type="mercator", y_axis_type="mercator", tools='')

    if title:
        plot.title.text = title
    wheezeetool = WheelZoomTool()
    plot.add_tools(PanTool(), wheezeetool, BoxZoomTool(),
                   SaveTool(), ResetTool(), UndoTool(), RedoTool())
    plot.toolbar.active_scroll = wheezeetool    
    plot.toolbar.logo = None

    # add roads
    # print("shackleton_dashboard.py - init_plot(): Gesource_plot.data:", Gesource_plot.data)
    rline = bokeh_utils.get_paths_glyph_line(coords=coords, color_key='plot_color')
    road_glyph = plot.add_glyph(Gesource_plot, rline)
    # osm
    rline_osm = bokeh_utils.get_paths_glyph_line(coords=coords, color_key='plot_color')
    road_glyph_osm = plot.add_glyph(Gesource_plot_osm, rline_osm)

    # print("Gesource_plot.data.keys():", Gesource_plot.data.keys())
    # print("len Gesource.data['speed_mph']", len(Gesource.data['speed_mph']))
    # print("len Gesource.data['eid']", len(Gesource.data['eid']))
    # print("len Gesource.data['length']", len(Gesource.data['length']))
    print("len Gesource.data", len(Gesource.data['length']))
    if show_line_hover:
        _ = bokeh_utils.add_line_hover(plot, renderers=[road_glyph])
    if show_road_labels:
        seg_labels = bokeh_utils.get_paths_labels_glyph(
            Gesource, coords=coords)
        seg_labels_glyph = plot.add_glyph(Gesource, seg_labels)


    # seems adding an empty cds causes problems???
    circ = bokeh_utils.get_nodes_glyph(coords=coords, shape='circle')
    print ("G nodes glyph", circ)
    print ("shackleton_dashboard.py - init_plot(): Gnsource_plot.data", Gnsource_plot.data)
    #print ("shackleton_dashboard.py - init_plot(): Gnsource.data", Gnsource.data)
    
    # circ = Circle(x="lon", y="lat", size='size', fill_color="color", \
    #        fill_alpha='alpha', line_color=None)#, legend='Intersections/Endpoints'
    node_glyph = plot.add_glyph(Gnsource_plot, circ)#   = this line yields a blank plot...
    # node_glyph = None
    glyph_node_list.extend([node_glyph])
    # osm
    # node_glyph_osm = plot.add_glyph(Gnsource_plot_osm, circ)
    # glyph_node_list.extend([node_glyph_osm])
    
    # test, add legend?
    add_legend = False
    if add_legend:
        #legends=[("Intersections/Endpoints", [circ_glyph])]
        legends = [("Intersections/Endpoints", [circ_glyph]),
                   ("Road", [road_glyph])]
        plot.add_layout(Legend(orientation="vertical", items=legends))

    #bokeh_utils.add_hover_save(plot, htmlout=htmlout, show=False,
    #                           add_hover=False,
    #                           renderers=glyph_node_list)

    print("Plots successfully initialized")
    
    return node_glyph, road_glyph


###############################################################################
def run_dashboard():
    '''
    Execute Dashboard
    '''

    global show_mst, show_evac, show_hulls, show_overlap, show_aug, \
        show_target, show_crit, show_risk, show_contours, straight_lines, \
        show_bbox, show_sources, auglist, target, end_nodes, skipnodes, \
        concave, node_set_name, goodroutes, skipnodes, compute_subgraph_centrality,  \
        compute_secondary_routes, r_m, panel_width_gui, binsize, \
        Gesource, esourcerw, source_target, source_sourcenodes,\
        source_crit, source_missing, source_histo_cum, source_histo_bin,\
        source_subgraph_centrality, source_hull_node, source_hull_patch, \
        source_hull_source, source_overlap, source_risk, source_contour, \
        cds_good_aug, cds_bad_aug, \
        esource_plot, esourcerw_plot, source_target_plot, \
        source_sourcenodes_plot, source_crit_plot, source_missing_plot, \
        source_histo_cum_plot, source_histo_bin_plot, source_subgraph_centrality_plot, \
        source_hull_node_plot, source_hull_patch_plot, \
        source_hull_source_plot, cds_bad_aug_plot, cds_good_aug_plot, \
        source_overlap_plot, source_risk_plot, source_contour_plot, \
        source_bbox_plot, source_force_proj_plot, \
        ecurve_dic_plot, plot_histo, auglist
    global osm, node_histogram, ntype_dic, X_dic, G0, G, \
        ecurve_dic, kd_idx_dic, kdtree, \
        df_bbox, g_node_props_dic, \
        df_bbox, source_bbox, source_force_proj, \
        auglist, ecurve_dic_plot, \
        straight_lines, \
        CRESI_PATH, \
        xmin, xmax, ymin, ymax
    global plot, global_dic, map_background, esource, Gnsource, \
        coords, bounds_dict, tabs, \
        plot_histo, source_histo_bin_plot, source_histo_cum_plot
    global Gnsource_plot, Gesource_plot, edge_update_dict
    global init_page_status_para
    
    
    # load data
    init_page_status_para.text = "Loading data..."
    load_data()
        
    init_page_status_para.text += "  Updating graph..."
    update_G()
    # expand auglist
    [ngood, nbad, ngood_aug, nbad_aug, cds_good_aug, cds_bad_aug] = auglist
    # init end_nodes
    end_nodes = ngood
    
    # initialize plot
    # something weird is happening in init_plot()...
    init_page_status_para.text += "  Initializing plot..."
    node_glyph, road_glyph = init_plot()
    
    print("run_gui(), step 0 - create cds")
    init_page_status_para.text += "  Creating CDS..."

    #############
    # init global sources to empty
    #Gnsource = bokeh_utils.set_nodes_source_empty()
    esource = bokeh_utils.set_edge_source_empty()
    esourcerw = bokeh_utils.set_edge_source_empty()
    source_target = bokeh_utils.set_nodes_source_empty()
    source_sourcenodes = bokeh_utils.set_nodes_source_empty()
    source_crit = bokeh_utils.set_nodes_source_empty()
    source_missing = bokeh_utils.set_nodes_source_empty()
    source_histo_cum = bokeh_utils.set_rect_source_empty()
    source_histo_bin = bokeh_utils.set_rect_source_empty()
    source_subgraph_centrality = bokeh_utils.set_nodes_source_empty()
    source_hull_node = bokeh_utils.set_nodes_source_empty()
    source_hull_patch = bokeh_utils.set_hull_patch_source_empty()
    source_hull_source = bokeh_utils.set_nodes_source_empty()
    source_overlap = bokeh_utils.set_nodes_source_empty()
    source_risk = bokeh_utils.set_nodes_source_empty()
    source_contour = bokeh_utils.set_edge_source_empty()

    # init plot sources to empty
    # nsource_plot = bokeh_utils.set_nodes_source_empty()
    esource_plot = bokeh_utils.set_edge_source_empty()
    esourcerw_plot = bokeh_utils.set_edge_source_empty()
    source_bbox_plot = bokeh_utils.set_bbox_source_empty()
    source_bbox_text_plot = bokeh_utils.set_bbox_source_empty()
    cds_good_aug_plot = bokeh_utils.set_nodes_source_empty()
    cds_bad_aug_plot = bokeh_utils.set_nodes_source_empty()
    source_target_plot = bokeh_utils.set_nodes_source_empty()
    source_target_text_plot = bokeh_utils.set_nodes_source_empty()
    source_sourcenodes_plot = bokeh_utils.set_nodes_source_empty()
    source_sourcenodes_text_plot = bokeh_utils.set_nodes_source_empty()
    source_crit_plot = bokeh_utils.set_nodes_source_empty()
    source_crit_text_plot = bokeh_utils.set_nodes_source_empty()
    source_missing_plot = bokeh_utils.set_nodes_source_empty()
    source_missing_text_plot = bokeh_utils.set_nodes_source_empty()
    source_subgraph_centrality_plot = bokeh_utils.set_nodes_source_empty()
    source_subgraph_centrality_text_plot = bokeh_utils.set_nodes_source_empty()
    source_hull_node_plot = bokeh_utils.set_nodes_source_empty()
    source_hull_patch_plot = bokeh_utils.set_hull_patch_source_empty()
    source_overlap_plot = bokeh_utils.set_nodes_source_empty()
    source_overlap_text_plot = bokeh_utils.set_nodes_source_empty()
    source_risk_plot = bokeh_utils.set_nodes_source_empty()
    source_risk_text_plot = bokeh_utils.set_nodes_source_empty()
    source_contour_plot = bokeh_utils.set_edge_source_empty()
    source_histo_cum_plot = bokeh_utils.set_rect_source_empty()
    source_histo_bin_plot = bokeh_utils.set_rect_source_empty()
    #############

    # create title source?
    # initial target
    target = graph_utils.choose_target(
        G, skiplists=[ngood, nbad, ngood_aug, nbad_aug], direction='south')
    update_target(target)
    
    print("run_gui(), step 1 - get glyphs")    
    init_page_status_para.text += "  Retrieving glyphs..."

    # shapes
    # get route glyphs
    paths_seg, paths_seg_sec, paths_line, paths_line_sec,\
        target_shape, target_text, sources_shape,\
        sources_text, crit_shape, crit_text, missing_shape, missing_text, \
        subgraph_centrality_shape, subgraph_centrality_text, diff_shape, diff_text, \
        rect_bin, rect_cum, hull_circ, hull_patch, risk_shape = \
        bokeh_utils.get_route_glyphs_all(coords=coords)

    # bbox plots?
    quad_glyph, quad_text_glyph = bokeh_utils.get_bbox_glyph(text_alpha=0.0,
                                                             text_font_size='5pt')

                                                            
    print("run_gui(), step 2 - a few more glyphs")
    
    # overlay aug
    shape_aug_good = bokeh_utils.get_nodes_glyph(shape='circle', coords=coords)
    shape_aug_bad = bokeh_utils.get_nodes_glyph(shape='circle', coords=coords)

    shape_force_proj = bokeh_utils.get_nodes_glyph(
        shape='circle', coords=coords)

    # contour_shape
    contour_shape = MultiLine(line_alpha=alpha_dic['contours'],
                              line_color='color', line_dash='solid',
                              line_width=3, xs='xs', ys='ys')

    ##############
    # Initialize network plot
    # plot, Gnsource, Gesource = bokeh_utils.G_to_bokeh(G,
    #                htmlout=None, ecurve_dic=ecurve_dic_plot, \
    #                gmap_background=gmap_background, show_nodes=True, \
    #                show_road_labels=False, verbose=True,  plot_width=plot_width, \
    #                title='Refined OSM Network', show=False, add_glyphs=False)
    # add glyphs
    # Set add_glyphs=False in G_to_bokeh, and add the glyphs here so we can use
    # tap tool to select targets
    # G shapes:
    if ecurve_dic_plot is not None:
        #Gseg = bokeh_utils.get_paths_glyph_line()
        paths_plot, paths_plot_sec = paths_line, paths_line_sec
    else:
        #Gseg = bokeh_utils.get_paths_glyph_seg()
        paths_plot, paths_plot_sec = paths_seg, paths_seg_sec

    ##############

    # Adding empty glyphs to plots appears to yield a blank plot in bokeh 2.4.3!
    
    #############
    # add glyphs to plot
    print("run_gui(), step 3 - add glyphs to plot")
    # need new glyphs
    # add augmented points to plot
    glyph_aug_good = plot.add_glyph(cds_good_aug_plot, shape_aug_good)
    glyph_aug_bad = plot.add_glyph(cds_bad_aug_plot, shape_aug_bad)
    # add force projection size to plot
    glyph_force_proj = plot.add_glyph(source_force_proj_plot, shape_force_proj)

    # add bbox to plot  
    # Addding bbox yields blank plot. [Only if arrays are empty, if arrays are [0], it's fine]
    # print(("shackleton_dashboard.py - init_plot(): source_bbox_plot.data:", source_bbox_plot.data))
    glyph_bbox_shape = plot.add_glyph(
        source_bbox_plot, quad_glyph)  # bbox_circle_shape)
    glyph_bbox_text = plot.add_glyph(
        source_bbox_text_plot, quad_text_glyph)  # bbox_text_shape)
    
    print("run_gui(), step 4 - add paths glyphs to plot")
    # computed routes
    glyph_paths_sec = plot.add_glyph(esourcerw_plot, paths_plot_sec)
    glyph_paths = plot.add_glyph(esource_plot, paths_plot)
    
    print("run_gui(), step 5 - add analytics nodes glyphs to plot")
    # sources
    glyph_sources_shape = plot.add_glyph(
        source_sourcenodes_plot, sources_shape)
    glyph_sources_text = plot.add_glyph(
        source_sourcenodes_text_plot, sources_text)
    # critical nodes
    glyph_crit_shape = plot.add_glyph(source_crit_plot, crit_shape)
    glyph_crit_text = plot.add_glyph(source_crit_text_plot, crit_text)
    # missing nodes
    glyph_missing_shape = plot.add_glyph(source_missing_plot, missing_shape)
    glyph_missing_text = plot.add_glyph(source_missing_text_plot, missing_text)
    # subgraph_centrality
    glyph_subgraph_centrality_shape = plot.add_glyph(
        source_subgraph_centrality_plot, subgraph_centrality_shape)
    glyph_subgraph_centrality_text = plot.add_glyph(
        source_subgraph_centrality_text_plot, subgraph_centrality_text)
    # target
    glyph_target_shape = plot.add_glyph(source_target_plot, target_shape)
    glyph_target_text = plot.add_glyph(source_target_text_plot, target_text)
    # overlap
    glyph_overlap_shape = plot.add_glyph(source_overlap_plot, diff_shape)
    glyph_overlap_text = plot.add_glyph(source_overlap_text_plot, diff_text)
    # risk
    glyph_risk_shape = plot.add_glyph(source_risk_plot, risk_shape)
    # hulls
    glyph_hull_node = plot.add_glyph(source_hull_node_plot, hull_circ)
    glyph_hull_patch = plot.add_glyph(source_hull_patch_plot, hull_patch)
    # contour
    #glyph_contour = plot.add_glyph(source_contour_plot, contour_shape)
    ##############

    print("run_gui(), step 6 - add hover to plot")
    init_page_status_para.text += "  Adding hover plot..."

    # intialize bbox to on?
    # source_bbox_plot.data = source_bbox.data

    # add bbox hover
    hover_table = \
        [
            ("ID", "@index"),
            ##("Node Name",  "@name"),
            ("Category", "@Category"),
            #("Prob", "@Prob"),
            #("Val", "@Val"),
            ("(lat, lon)", "(@lat, @lon)"),
            #("(x,y)", "($x, $y)"),
            #("Nearest Node", "@nearest_node"),
            ("Distance", "@dist")
        ]
    # add bbox hover tool
    hover_bbox = HoverTool(tooltips=hover_table, renderers=[
                           glyph_bbox_shape])  # , line_policy=line_policy)
    plot.add_tools(hover_bbox)

    # add target hover
    hover_table_target = [
        ("Target", "@nid"),
        ("lat", "@lat"),
        ("lon", "@lon"),
    ]
    hover_target = HoverTool(tooltips=hover_table_target,
                             renderers=[glyph_target_shape])
    plot.add_tools(hover_target)
    
    print("run_gui(), step 7 - add rects to histo plot")

    # add rects to plot
    rect_bin = Rect(x='x', y='y_mid',  height='y',  width='width',
                    fill_color='color', fill_alpha='alpha')  # get_histo_glyph()
    rect_cum = Rect(x='x', y='y_mid',  height='y',  width='width',
                    fill_color='color', fill_alpha='alpha')  # get_histo_glyph()
    cum_glyph = plot_histo.add_glyph(source_histo_cum_plot, rect_cum)
    bin_glyph = plot_histo.add_glyph(source_histo_bin_plot, rect_bin)
    # plot_histo.add_layout(CategoricalAxis(
    #        major_label_orientation=math.pi/4,),
    #        'below')
    #plot_histo.x_range = Range1d(source_histo_cum.data['x'])
    # plot_histo.add_layout(CategoricalAxis(location='below',
    #                           ticker=CategoricalTicker()))
    # legend
    leg_glyph = Legend(orientation="vertical",  # top_left",
                       items=[("Cumulative", [cum_glyph]),
                              ("Binned", [bin_glyph])])
    plot_histo.add_layout(leg_glyph)

    # add hover
    hover_tableh = [("Bin",  "@x_str"), ("Count", "@y"), ]
    hoverh = HoverTool(tooltips=hover_tableh)
    plot_histo.add_tools(hoverh)
    plot_histo.toolbar.logo = None
    #############

    print("run_gui(), step 8 - add tap tool to plot")
    init_page_status_para.text += "  Adding tap tool to plot..."

    #############
    # Add tap to plot
    plot.add_tools(TapTool())
    #############
    # tap tool

    def on_tap_change(attr, old, new):
        global Gnsource_plot, target, source_target, show_target, show_hulls, \
            show_mst,  show_evac

        tap_verbose = True
        if tap_verbose:
            print("tap attr", attr)
            print("tap old", old)
            print("tap new", new)

        # bokeh 1.0
        #   https://stackoverflow.com/questions/42234751/bokeh-server-callback-from-tools
        # The index of the selected glyph is : new['1d']['indices'][0]
        # if tap_verbose:
        #     print("Gnsource.data.keys():", Gnsource_plot.data.keys())
        node_list = Gnsource_plot.data['nid'][new]
        #node_list = Gnsource.data['name'][new]
        print("TapTool callback executed, selected Target", node_list)

        if node_list:
            target = node_list[0]
            # update values
            update_target(target)
            show_target = True
            source_target_plot.data = dict(source_target.data)
            if show_hulls:
                update_hulls()
            if show_evac:
                update_routes()

    Gnsource_plot.selected.on_change('indices', on_tap_change)

    # https://stackoverflow.com/questions/42234751/bokeh-server-callback-from-tools
    # https://stackoverflow.com/questions/50478223/bokeh-taptool-return-selected-index
    # https://stackoverflow.com/questions/44961192/how-to-create-a-bokeh-tap-tool-to-select-all-points-of-the-columndatasource-shar

    # update plot bounds?
    plot.x_range.bounds = (bounds_dict['x_range'][0], bounds_dict['x_range'][1])
    plot.y_range.bounds = (bounds_dict['y_range'][0], bounds_dict['y_range'][1])
    
    print("run_gui() fully executed...")
    return
    

###########################################
### Shackleton
div_height = 18
div_style_dict = {'font-size': '12px',
                  #'text-align': 'center',
                   'text-indent': '20px',
                  #'text-decoration': 'overline',
                  'font-weight': 'bold',
                  'color': 'black',
                  'background-color': "#ecf9f2", # "#e6ffee", #"paleturquoise"
                  # 'border': '1px solid black'
                  'border-style': 'solid',
                  'border-top-width': '1px',
                  'border-left-width': '0px',
                  'border-right-width': '0px',
                  'border-bottom-width': '0px',
                  'vertical-align': 'middle',
                  #'margin': 'auto',
                  # 'padding': '5px'
                  # 'height': '18px',
                  }
div_style_dict_title = div_style_dict.copy()
div_style_dict_title['font-size'] = '20px'
div_style_dict_title['text_alight'] = 'center'
div_style_dict_title['border-top-width'] = '0px'

div_style_dict_subtitle = div_style_dict.copy()
div_style_dict_subtitle['font-size'] = '16px'

# Define map widgets
widget_panel = widgetbox(
    # button_reset,
    
    # https://docs.bokeh.org/en/2.4.0/docs/user_guide/interaction/widgets.html#div
    # Div(text="<img src='https://avatars.githubusercontent.com/u/53105934?s=200&v=4'>", width=panel_width, height=20),
    
    #Paragraph(text='', width=panel_width, height=2),
    # Paragraph(text='ROAD NETWORK', width=panel_width, height=20),
    Div(text='Geodesic Labs', style=div_style_dict_title,
        width=panel_width, height=div_height),        
    Paragraph(text='', width=panel_width, height=1),
    Div(text='Shackleton Dashboard', style=div_style_dict_subtitle,
        width=panel_width, height=div_height),
    Paragraph(text='', width=panel_width, height=2),

    # Paragraph(text='ROAD NETWORK', width=panel_width, height=20),
    Div(text='ROAD NETWORK', style=div_style_dict,
        width=panel_width, height=div_height),
    radio_group_edges,

    # Paragraph(text='', width=panel_width, height=10),
    Div(text='VEHICLES', style=div_style_dict,
        width=panel_width, height=div_height),
    bbox_group,
    vehicle_para,
    
    Div(text='ANALYTICS', style=div_style_dict,
        width=panel_width, height=div_height),
    radio_group_comp,
    # radio_group_convex,

    Div(text='ROUTE OPTIONS', style=div_style_dict,
        width=panel_width, height=div_height),
    radio_group_endnodes,
    radio_group_skipnodes,
    sec_route_group,
    # radio_group_contours,

    # Paragraph(text='', width=panel_width, height=10),
    Div(text='OPTIMAL PATH', style=div_style_dict,
        width=panel_width, height=div_height),
    radio_group_edge_weight,

    Div(text='NODE DISPLAY OPTIONS', style=div_style_dict,
        width=panel_width, height=div_height),
    bbox_aug_group,
    plot_options_group,
    slider_aug_gui,
    slider_binsize,

    # Paragraph(text='', width=panel_width, height=20),
    # Div(text='EXPORT', style=div_style_dict,
    #     width=panel_width, height=div_height),
    # button_mst,      # blank_6,
    # button_evac,     # blank_7,
    # button_hull,     # blank_8,
    # button_tt,       # blank_9,
    # button_overlap,  # blank_10,
    # button_risk,

    width=panel_width_gui)

########
# layout
vbox_plots = column(plot, blank_100, plot_histo)

layout_tab2 = row(widget_panel, vbox_plots)

# # Tabs?
# tab2 = Panel(child=layout_tab2, title="")
# tabs = Tabs(tabs=[tab2])

# Total
# document.add_root(tabs)
document.add_root(layout_tab2)

##############
run_dashboard()