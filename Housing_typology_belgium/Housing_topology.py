#!/usr/bin/env python
# coding: utf-8
import multiprocessing as mp

import numpy as np
import geopandas as gpd
import shapely
import pandas as pd
import time

from copy import deepcopy

import os
import networkx as nx
import osmnx as ox

def run_time(start):
    end = time.time() - start
    day = end // (24 * 3600)
    end = end % (24 * 3600)
    hour = end // 3600
    end %= 3600
    minutes = end // 60
    end %= 60
    seconds = end
    print("d:h:m:s-> %d:%d:%d:%d" % (day, hour, minutes, seconds))
               

def find_adjacent_blocks(blocks):
    
    """
    Function to find and merge blocks that touch each other
    """
    print("Processing block file. Plese wait....")
    #blocks = blocks.copy()
    crs_value = blocks.crs
    merged_blocks = []
    for index, row in blocks.iterrows():
         
        neighbours = blocks[blocks.geometry.touches(row['geometry'])].RecId.tolist()
        if len(neighbours) > 0:
            merged_blocks = merged_blocks + neighbours
            neighbours = neighbours + [row.RecId]
            blocks_sub = blocks[blocks.RecId.isin(neighbours)]
            boundary = shapely.ops.cascaded_union(blocks_sub.geometry)
            blocks.at[index, 'geometry'] = boundary
            
    blocks_left = [bid for bid in blocks.RecId if bid not in merged_blocks]
    blocks = gpd.GeoDataFrame(blocks[blocks.RecId.isin(blocks_left)])
    blocks.reset_index(inplace=True)
    blocks.crs = crs_value
    
    return blocks


def buildings_to_parcel(houses, parcels):
    
    houses = houses.copy()
    join_parcel= gpd.sjoin(houses, parcels, how="left", op="within")
    houses2parcels = pd.DataFrame(join_parcel.head(len(join_parcel)))
    
    # calculate parcel coverage
    print("Computing parcel coverage. Plese wait....")
    buildings_area = pd.DataFrame(houses2parcels.groupby("RecId_right", as_index=False).Shape_area_left
                                  .agg({"building_area": "sum"}))
    for i, row in houses2parcels.iterrows():
        try:
            bld_area = buildings_area[buildings_area.RecId_right == row.RecId_right].values[0][1]
        except:
             bld_area = 0
        
        parcel_coverage = 100*(bld_area/row.Shape_area_right)
        houses2parcels.at[i,'parcel_coverage'] = round(parcel_coverage, 2)
        #houses2parcels.at[i,'parcel_geometry'] = parcels[parcels.RecId == row.RecId_right].geometry.tolist[0]
        
    houses2parcels.reset_index(inplace=True)
    houses2parcels = gpd.GeoDataFrame(houses2parcels)
    houses2parcels.crs = houses.crs
    
    return houses2parcels


def distance_to_all_nodes(place, network_type, cutoff=5000, simplify=False):
    """
    Calculate distance (in meters) to all nodes in a graph
    Input:
        place: name of the place to generate the graph (check osnmx documentation)
        network_type: one of the following values
            drive - get drivable public streets (but not service roads)
            drive_service - get drivable streets, including service roads
            walk - get all streets and paths that pedestrians can use (this network type ignores one-way directionality)
            bike - get all streets and paths that cyclists can use
            all - download all non-private OSM streets and paths
            all_private - download all OSM streets and paths, including private-access ones
        cutoff: maximum distance between nodes
        simplify: allow osnmx to automatically simplify the graph nodes
    """
    
    graph = ox.graph_from_place(place, network_type=network_type, simplify=simplify)
    distance_generator = nx.all_pairs_dijkstra_path_length(graph, weight="length", cutoff=cutoff)
    #distance_generator = nx.all_pairs_dijkstra_path_length(graph, weight="length")
    
    distance_dist = {}
    for element in distance_generator:
        node = element[0]
        node_dict = element[1]
        try:
            distance_dist[node] = node_dict
        except:
            distance_dist[node] = {}
    
    return graph, distance_dist


def min_distance(origin_nodes, target_nodes, distance_dictionary):
    distance_to_poi = []
    for current_node in origin_nodes:
        try:
            distance = [dist for node, dist in distance_dictionary[current_node].items() if node in target_nodes]        
            distance_to_poi.append(min(distance))
        except:
            min_dist = None
            distance_to_poi.append(min_dist)
            
    return distance_to_poi


def distance_to_POI(distance_dictionary, graph, origin_df, target_bus, target_poi):

    # loaction houses
    origin_xy = np.c_[origin_df.Long.values, origin_df.Lat.values]
    origin_nodes = ox.get_nearest_nodes(graph, origin_xy[:, 0], origin_xy[:, 1], method='balltree')
    
    # bus stops
    bus = target_bus[target_bus.fclass == 'bus_stop']
    bus_xy = np.c_[bus.Long.values, bus.Lat.values]
    nodes_bus = ox.get_nearest_nodes(graph, bus_xy[:, 0], bus_xy[:, 1], method='balltree')
    
    # train stops
    train = target_bus[target_bus.fclass == 'train']
    train_xy = np.c_[train.Long.values, train.Lat.values]
    nodes_train = ox.get_nearest_nodes(graph, train_xy[:, 0], train_xy[:, 1], method='balltree')
    
    # school
    school = target_poi[target_poi.fclass == 'school']
    school_xy = np.c_[school.Long.values, school.Lat.values]
    nodes_school = ox.get_nearest_nodes(graph, school_xy[:, 0], school_xy[:, 1], method='balltree')
 
    # kindergarten
    kindergarten = target_poi[target_poi.fclass == 'kindergarten']
    kindergarten_xy = np.c_[kindergarten.Long.values, kindergarten.Lat.values]
    nodes_kindergarten = ox.get_nearest_nodes(graph, kindergarten_xy[:, 0], kindergarten_xy[:, 1], method='balltree')
                   
    # Calculate distances
    distance_to_bus = []
    distance_to_train = []
    distance_to_kindergarten = []
    distance_to_school = []
    #distance_to_town_hall = []
    
    for current_node in origin_nodes:
        
        # bus stops 
        try:
            dist_bus = [dist for node, dist in distance_dictionary[current_node].items() if node in nodes_bus]        
            distance_to_bus.append(min(dist_bus))
        except:
            min_dist = None
            distance_to_bus.append(min_dist)
                   
        # train stations
        try:
            dist_train = [dist for node, dist in distance_dictionary[current_node].items() if node in nodes_train]        
            distance_to_train.append(min(dist_train))
        except:
            min_dist = None
            distance_to_train.append(min_dist)                   
        
        # school
        try:
            dist_school = [dist for node, dist in distance_dictionary[current_node].items() if node in nodes_school]        
            distance_to_school.append(min(dist_school))
        except:
            min_dist = None
            distance_to_school.append(min_dist)    
                   
        # kindergarten
        try:
            dist_kindergarten = [dist for node, dist in distance_dictionary[current_node].items() if node in nodes_kindergarten]        
            distance_to_kindergarten.append(min(dist_kindergarten))
        except:
            min_dist = None
            distance_to_kindergarten.append(min_dist) 
            
    # describe how to add other point of interest
                   
    return distance_to_bus, distance_to_train, distance_to_kindergarten, distance_to_school


def house_topology(houses, parcels, blocks, buffer_g=2.5, buffer_n = 2):
    """
        Function to compute house typology
    """
    houses = deepcopy(houses)
    blocks = find_adjacent_blocks(blocks)
    
    bounds = houses.centroid.bounds
    houses['LambertX'] = bounds.minx
    houses['LambertY'] = bounds.miny
            
    # Assign buildings to blocks
    join_block = gpd.sjoin(houses, blocks, how="left", op="within")
    houses2blocks = pd.DataFrame(join_block.head(len(join_block)))
    houses2blocks = houses2blocks[~houses2blocks.index.duplicated()]
    houses2blocks = gpd.GeoDataFrame(houses2blocks)
    houses2blocks.crs = houses.crs
    houses['BlockID'] = houses2blocks['RecId_right']
    
    # Assign buildings to parcels    
    houses2parcels = buildings_to_parcel(houses, parcels)
    houses2parcels = houses2parcels[~houses2parcels.index.duplicated()]
    houses['ParcelID'] = houses2parcels['RecId_right']
    houses["parcel_area"] = houses2parcels.Shape_area_right
    houses["parcel_cover"] = houses2parcels.parcel_coverage
    
    houses['centroid'] = houses['geometry'].centroid
    houses['buffer_g'] = houses.buffer(buffer_g)
    houses['buffer_n'] = houses.buffer(buffer_n)
    
    # keep larget house (polygon in each parcel)
    houses = houses.loc[houses.groupby(["ParcelID"])["Shape_area"].idxmax()]
    houses.reset_index(inplace=True)
    
    print("Computing house typology. Please wait....")
    for i, row in houses.iterrows():
        
        #Check if there is a front garden
        block_RecId = row.BlockID
        houses_in_block = houses[houses.BlockID == block_RecId].RecId.tolist()
        
        ###--- check if there is front garden i.e if the house is at least buffer_g meters from the edge of the block
        current_house_g = gpd.GeoDataFrame({'RecId':row.RecId, 'geometry': [row.buffer_g]})
        current_house_g.crs = blocks.crs
            
        current_block = gpd.GeoDataFrame({'RecId':blocks[blocks.RecId == block_RecId].RecId, 'geometry': blocks[blocks.RecId == block_RecId].geometry})
        current_block.crs = blocks.crs
        
        join_house_g = gpd.sjoin(current_house_g, current_block, how="left", op="within")
        join_house_g = pd.DataFrame(join_house_g.head(len(join_house_g)))
        front_garden = join_house_g.RecId_right.isna().values[0]
                
        ###--- Count number of facades
        # remove all houses in current parcel this is to avoid any overlap with garden houses in parcel 
        parcel_id = row.ParcelID
        buildings_in_parcel = houses[houses.ParcelID == parcel_id].RecId.tolist()       # create a list of all buildings in the current land parcel
        houses_not_in_parcels = [hid for hid in houses_in_block if hid not in buildings_in_parcel]  # create a list of all houses in the block excluding those in the parcel of interest
        
        houses_tmp = houses[houses.RecId.isin(houses_not_in_parcels)]   # Geodataframe of houses within the current block but not in the current parcel
        houses_tmp.reset_index(inplace=True)
        houses_tmp = gpd.GeoDataFrame(houses_tmp)
        houses_tmp.crs = houses.crs
        
        neighbours = houses_tmp[houses_tmp.geometry.touches(row['geometry'])].RecId.tolist()    # get the list of all houses that touches the current house
        num_neighbours = len(houses[houses.RecId.isin(neighbours)].ParcelID.unique())      # count one house per parcel that touches the current house
        num_facades = 4 - num_neighbours

        # IF there are more than one house in the block adjust the number of facade by accounting for a distance of buffer_n meters between buildings
        if len(houses_tmp) > 0:
            
            current_house_n = gpd.GeoDataFrame({'RecId':row.RecId, 'geometry': [row.buffer_n]}) # Geodataframe for current house based on geometry butffer_n
            current_house_n.crs = houses.crs
            
            # check  if the buffer_n extension of the house overlaps another building in the block
            intersection = gpd.overlay(current_house_n, houses_tmp, how='intersection')
            try:
                new_neighbours = intersection.RecId_2.tolist()
            except:
                new_neighbours = []

            # Adjust neigbours and consider only one polygon (building) per parcel as neighbour
            all_neighbours = neighbours + new_neighbours
            num_neighbours = len(houses_tmp[houses_tmp.RecId.isin(all_neighbours)].ParcelID.unique())
            num_facades = 4 - num_neighbours
        
        # facades less than 2 are not allowed and most likely not a house 
        if num_facades < 2:
            num_facades = None
            
        houses.at[i, "has_front_garden"] = not front_garden
        houses.at[i, "num_neighbours"] = num_neighbours
        houses.at[i, "num_facades"] = num_facades
    
    return houses


def topology_per_commune(commune, tmp_folder, cadastral_folder, buffer_g, buffer_n, simplify, cutoff, point_of_interest, transport, poi_usage):
    
    commune_name = commune.split('/')[-1]
    print(commune_name)
    Bpn_CaBu_file = commune + "/Bpn_CaBu.shp"
    Bpn_CaPa_file = commune + "/Bpn_CaPa.shp"
    Bpn_CaBl_file = commune + "/Bpn_CaBl.shp"
        
    Bpn_CaBu = gpd.read_file(Bpn_CaBu_file)
    Bpn_CaBu.dropna(subset=['geometry'], inplace = True)

    Bpn_CaPa = gpd.read_file(Bpn_CaPa_file)
    Bpn_CaPa.dropna(subset=['geometry'], inplace = True)

    Bpn_CaBl = gpd.read_file(Bpn_CaBl_file)
    Bpn_CaBl.dropna(subset=['geometry'], inplace = True)

    # calculate housing typology
    housing_top = house_topology(Bpn_CaBu, Bpn_CaPa, Bpn_CaBl, buffer_g=buffer_g, buffer_n = buffer_n)

    if poi_usage == True:            
        # Project dataframe to similar geometry as osmnx
        houses_prj = ox.project_gdf(Bpn_CaBu, to_latlong=True)
        bounds = houses_prj.centroid.bounds
        houses_prj['Long'] = bounds.minx
        houses_prj['Lat'] = bounds.miny

        ##--- Graphs of commune
        place = commune_name + ', Belgium'
        graph, distance_dictionary = distance_to_all_nodes(place, network_type=network_type, simplify=simplify, cutoff=cutoff)

        ###---- Goefabrik Distance to public transport
        bounds = transport.centroid.bounds
        transport['Long'] = bounds.minx
        transport['Lat'] = bounds.miny
        transport['fclass'] = np.where(transport.fclass.isin(['bus_station','bus_stop','tram_stop']),'bus_stop', transport.fclass)
        transport['fclass'] = np.where(transport.fclass.isin(['railway_station','railway_halt']),'train', transport.fclass)
        public_transport = transport[transport.fclass.isin(['bus_stop','train'])]

        ####----- point of interest
        bounds = point_of_interest.centroid.bounds
        point_of_interest['Long'] = bounds.minx
        point_of_interest['Lat'] = bounds.miny

        ###---   Distance calculation 
        distance_to_bus, distance_to_train, distance_to_kindergarten, distance_to_school = distance_to_POI(distance_dictionary, graph, houses_prj, public_transport, point_of_interest)
        housing_top['distance_to_bus_stop'] = distance_to_bus
        housing_top['distance_to_train_station'] = distance_to_train
        housing_top['distance_to_kindergarten'] = distance_to_kindergarten
        housing_top['distance_to_school'] = distance_to_school
            
    cols2drop = ['index', 'geometry','centroid','buffer_g','buffer_n']
    housing_top.drop(columns=cols2drop, inplace=True)
    outputfile = tmp_folder + commune_name + ".csv"
    housing_top.to_csv(outputfile, index=False)
    
    return outputfile
    

if __name__ == '__main__':
    
    start = time.time()
    
    from datetime import datetime
    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    print("Start time =", current_time)

    buffer_n = 2.5      # buffer lenght for facade counts
    buffer_g = 2.5      # buffer lenght for front gardens
    network_type = "walk"
    simplify = False
    cutoff = 5000
    poi_usage = False
    nproc = 2
    
    # free POI dat from geofabrik
    point_of_interest = gpd.GeoDataFrame()
    transport =  gpd.GeoDataFrame()
    if poi_usage == True:
        poi_file = "./data/gis_osm_pois_a_free_1.shp"
        point_of_interest = gpd.read_file(poi_file)  # free point of interst data
        
        transport_file = "./data/gis_osm_transport_free_1.shp"
        transport = gpd.read_file(transport_file)    # free public tranaport
    
    # path to other source files
    tmp_folder = "./data/Temp_commune/"
    cadastral_folder = "./data/cadastral/"
    
    # dataframe to concatenate across all cities
    outputfiles = []
    subfolders = [f.path for f in os.scandir(cadastral_folder) if f.is_dir()]
    
    # Normal section un comment the codes below and comment the multiprocessing section to keep only the single processor code
    #for communes in subfolders:
    #    temp_file = topology_per_commune(communes, tmp_folder, cadastral_folder, buffer_g, buffer_n,simplify, cutoff, point_of_interest, transport, poi_usage)
    #    outputfiles.append(outputfiles)
    
    ###--- Multiprocessing section []
    if nproc < 0:
        nproc = mp.cpu_count()
    pool = mp.Pool(nproc)
    outputfiles = pool.starmap_async(topology_per_commune, [(communes, tmp_folder, cadastral_folder, buffer_g, buffer_n, 
                                                             simplify, cutoff, point_of_interest, transport, poi_usage)  for communes in subfolders]).get()
    pool.close()
    pool.join()
        
    # combined all typology files
    housing_typology = pd.DataFrame()
    for file in outputfiles:
        tmp_file = pd.read_csv(file)
        housing_typology = pd.concat([tmp_file, housing_typology], axis=1)
        
    housing_typo_out = "./data/housing_typology.csv"
    housing_typology.to_csv(housing_typo_out, index=False)
    
    print(run_time(start))
