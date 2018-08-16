# Import packages
import os
import pandas as pd
import osmnx as ox
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
import pandana
from pandana.loaders import osm


#######################
## UTILITY FUNCTIONS ##
#######################

def get_pairwise_distances(stations_df, G):
    """
    Calculates the shortest pairwise distance between all pairwise combinations of 
    fuel stations using the osmnx method shortest_path_length(). Distances in metre.
    
    TODO: Annotate the nested for loops for better readibility

    Args:
     stations_df: Dataframe with station ID, lat, lon
     G: OSMnx graph object of region with street network

    Returns:
     results_df: results dataframe with shortest paths between station pairs
    """

    # Set up list to collect results
    results = []

    # Nested for loops to calculate pairwise distances
    # Loop 1 for station i
    # Loop 2 for distance calculation to every other station j
    # Inefficient method because includes i-i and re-calcuates j-i
    # Even after i-j has been calculated.
    for i in range(len(stations_df)):
        row_i = stations_df.loc[i]
        lat_i = row_i['lat']
        lon_i = row_i['lon']
        id_i = row_i['id']
        origin_node = ox.get_nearest_node(G, (lat_i, lon_i))
        for j in range(len(stations_df)):
            row_j = stations_df.loc[j]
            lat_j = row_j['lat']
            lon_j = row_j['lon']
            id_j = row_j['id']       
            destination_node = ox.get_nearest_node(G, (lat_j, lon_j))
            distance = nx.shortest_path_length(G, origin_node, destination_node, weight='length')
            df = pd.DataFrame({'from': [id_i],'to': [id_j], 'distance': [distance]})
            results.append(df)

    # Compress list of individual results into one large dataframe
    results_df = pd.concat(results)
    return results_df


def prettify_pairwise_distance_df(pairwise_df_raw, stations_df):
    """
    Prettifies the pairwise distance matrix. Removes i-i. 
    Adds in station names for better readibility. 

    Args:
     pairwise_df_raw:
     stations_df:
    
    Returns:
     pairwise_df: 
    """
    
    station_names = stations_df[['id', 'name']]
    pairwise_df = (pd.merge(pd.merge(pairwise_df_raw, station_names,
                                     left_on='from', right_on='id'),
                            station_names, left_on='to', right_on='id')
                   .drop(columns=['from', 'to'])
                   .rename(columns={'name_x': 'from', 'name_y': 'to',
                                    'id_x': 'id_from', 'id_y': 'id_to'})
                   .query('distance > 0.0'))

    return pairwise_df


def interbrand_pairwise_distance_df(interbrand_pairwise_final):
    """
    Creates closest stations df from interbrand pairwise distances
    
    Args:
     interbrand_pairwise_final:
    
    Returns:
     closest_stations
    """
    
    closest_stations = (interbrand_pairwise_final
                        .groupby('from')['distance']
                        .agg('min')
                        .reset_index(name='distance'))

    closest_stations = (pd.merge(interbrand_pairwise_final, 
                                 closest_stations)[['from', 'to', 'distance']]
                        .sort_values('distance'))

    closest_stations['from_brand'] = (closest_stations['from']
                                      .apply(lambda x: str(x)[0:2].replace(" ", "")))
    closest_stations['to_brand'] = (closest_stations['to']
                                    .apply(lambda x: str(x)[0:2].replace(" ", "")))
    return closest_stations


def get_shortest_path(origin_point, destination_point, G):
    """
    Calculates and returns the shortest path between two points on an OSM network. 
    Also returns the length (in metres) of the shortest path

    Args:
     origin_point: (lat,lon) of the origin
     destinatiion_point: (lat,lon) of the destination
     G: OSMnx network object
    
    Returns:
     route: list of nodes in the shortest path.
     distance: length of the shortest path
    """
    
    origin_node = ox.get_nearest_node(G, origin_point)
    destination_node = ox.get_nearest_node(G, destination_point)

    distance = nx.shortest_path_length(G, origin_node, destination_node, weight='length')
    route = nx.shortest_path(G, origin_node, destination_node, weight='length')
    return route, distance


def station_degree(g):
    """
    Returns unweighted degree per fuel station in graph object.
    
    Args:
     g:
    
    Returns:
     degree_df:
    """

    degree_df = (pd.DataFrame(list(g.degree()))
                 .rename(columns={0: 'station', 1: 'degree'})
                 .sort_values('degree', ascending=False))
    return degree_df


########################
## PLOTTING FUNCTIONS ##
########################

def visualise_station_network(pairwise_df, neighbour_radius = 10000.0, station_brand='Z'):
    """
    Visualise 'network' of Z stations that are at most 10km away from each other. 

    Args:
     pairwise_df:
     neighbour_radius: 
     station_brand:

    Returns:
     Networkx graph object
    """
         
    # Get the data
    stations_radius = (pairwise_df
                       .query('distance <= @neighbour_radius')
                       .rename(columns={'distance': 'weight'}))

    # Create the graph
    g = nx.from_pandas_edgelist(stations_radius, 'from', 'to', ['weight']) 
    g.add_nodes_from(nodes_for_adding=stations_radius['from'].tolist())

    # Create Edge Weights where edges are stronger if closer together
    edges = g.edges()
    for u,v in edges:
        g[u][v]['inv_weight'] = (1.0 / (g[u][v]['weight'] / neighbour_radius)) * 0.5
    edge_width = [g[u][v]['inv_weight'] for u,v in edges]

    # Create a layout for our nodes 
    layout = nx.spring_layout(g, iterations=1000, weight='inv_weight', random_state=252)

    # Draw the network
    plt.figure(figsize=(9, 9))
    plt.title('Station network for {}'.format(station_brand))
    nx.draw(g, layout, width=edge_width)
    nx.draw_networkx_labels(g, layout, font_size=11, font_family='sans-serif');
    return g


def plot_shortest_paths(pairwise_df, station_brand='Z'):
    """
    Plot a histogram of the shortest path between fuel stations. 

    Args:
     pairwise_df:
     station_brand:
    
    Returns:
     Histogram of shortest distances from any fuel station in data.
    """
    
    closest_stations = (pairwise_df
                        .groupby('from')['distance']
                        .agg('min')
                        .sort_values())
    title_string = 'Distance between {} \
    stations \n Mean = {} m. Median  = {} m'.format(station_brand,
                                                    closest_stations.mean(), 
                                                    closest_stations.median())
    # Plot the distances
    plt.hist(closest_stations);
    plt.title(title_string);


def plot_interstation_distance_multibrand(pairwise_df_multibrand):
    closest_stations = (pairwise_df_multibrand
                        .groupby(['from', 'brand'])['distance']
                        .agg('min')
                        .reset_index()
                       # .drop(columns='index')
                        .sort_values('distance'))
    
    def vertical_average_lines(x, **kwargs):
        plt.axvline(x.mean(), color='r',
                    label= 'Mean = '+str(int(x.mean())))
        plt.axvline(x.median(),
                    color='k',
                    ls='--',
                    label='Median = '+str(int(x.median())))

        # txkw = dict(size=16, color = 'r', rotation=0)
        # tx = "mean: {:.2f},\nmedian: {:.2f}".format(x.mean(),x.median())
        # plt.text(x.mean() + 20, 5, tx, **txkw)
        plt.legend(loc='upper right')

    
    grid = sns.FacetGrid(closest_stations, col='brand', size=5)
    grid.map(plt.hist, 'distance')
    grid.map(vertical_average_lines, 'distance')
    grid.add_legend()
    
    return


###########################
## PANDANA ACCESSIBILITY ##
###########################

def get_pandana_network(bbox, tags):
    """
    """
    
    # Define some parameters
    pandana_bbox = [bbox[1], bbox[0], bbox[3], bbox[2]]
    bbox_string = '_'.join([str(x) for x in pandana_bbox])
    num_categories = len(tags) + 1
    net_filename = 'data/network_{}.h5'.format(bbox_string)

    if os.path.isfile(net_filename):
        # if a street network file already exists, just load the dataset from that
        network = pandana.network.Network.from_hdf5(net_filename)
    else:
        # otherwise, query the OSM API for the street network within the specified bounding box
        network = osm.pdna_network_from_bbox(pandana_bbox[0],
                                             pandana_bbox[1],
                                             pandana_bbox[2],
                                             pandana_bbox[3], 
                                             network_type='drive')


        # identify nodes that are connected to fewer than some threshold
        # of other nodes within a given distance
        lcn = network.low_connectivity_nodes(impedance=1000, count=10, imp_name='distance')
        network.save_hdf5(net_filename, rm_nodes=lcn)


    return network, pandana_bbox


def get_accessibility(network, pois_df, distance=5000, num_pois=10):
    """
    """
    
    network.precompute(distance + 1)
    network.set_pois(category='all',
                     x_col=pois_df['lon'],
                     y_col=pois_df['lat'],
                     maxdist=distance,
                     maxitems=num_pois)
    accessibility = network.nearest_pois(distance=distance, category='all', num_pois=num_pois)
    return accessibility


def plot_accessibility(network, accessibility,
                       pandana_bbox,
                       amenity_type = 'Z Fuel Station',
                       place_name='Wellington',
                       fig_kwargs={}, plot_kwargs={},
                       cbar_kwargs={}, bmap_kwargs={}):
    """
    """

    
    title = 'Driving distance (m) to nearest {} around {}'.format(amenity_type,
                                                                  place_name)
 
    # network aggregation plots are the same as regular scatter plots,
    # but without a reversed colormap
    agg_plot_kwargs = plot_kwargs.copy()
    agg_plot_kwargs['cmap'] = 'viridis'
    
    # Plot
    bmap, fig, ax = network.plot(accessibility, 
                                 bbox=pandana_bbox, 
                                 plot_kwargs=plot_kwargs, 
                                 fig_kwargs=fig_kwargs, 
                                 bmap_kwargs=bmap_kwargs, 
                                 cbar_kwargs=cbar_kwargs)
    ax.set_title(title,  fontsize=15)
    return
