# Import packages
import pandas as pd
import osmnx as ox
import networkx as nx
import matplotlib.pyplot as plt


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
    
    closest_stations = pairwise_df.groupby('from')['distance'].agg('min')
    closest_stations.sort_values()
    # Plot the distances
    plt.hist(closest_stations);
    plt.title('Distance between {} stations \n Mean = {} m. Median  = {} m'.format(station_brand,
                                                                                   closest_stations.mean(), 
                                                                                   closest_stations.median()));
