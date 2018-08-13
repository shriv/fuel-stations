# Import packages
import pandas as pd
import osmnx as ox
import networkx as nx
import matplotlib.pyplot as plt


def get_pairwise_distances(stations_df, G):
    """
    Calculates the shortest pairwise distance between all pairwise combinations of 
    fuel stations using the osmnx method shortest_path_length(). Distances in metre.
    
    TODO: Annotate the nested for loops for better readibility

    Args:
     stations_df: Dataframe with station ID, lat, lon
     G: graph of region with street network

    Returns:
     results_df: results dataframe with shortest paths between station pairs
    """
    results = []
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

    results_df = pd.concat(results)
    return results_df


def visualise_station_network(pairwise_df, neighbour_radius = 10000.0, station_brand='Z'):
    """
    Visualise 'network' of Z stations that are at most 10km away from each other. 

    Args:
     pairwise_df:
     neighbour_radius: 
     station_brand:

    Returns:
     None
    """
         
    # 1. Get the data
    stations_radius = (pairwise_df
                       .query('distance <= @neighbour_radius')
                       .rename(columns={'distance': 'weight'}))

    # 2. Create the graph
    g = nx.from_pandas_edgelist(stations_radius, 'from', 'to', ['weight']) 
    g.add_nodes_from(nodes_for_adding=stations_radius['from'].tolist())

    # 3. Create Edge Weights where edges are stronger if closer together
    edges = g.edges()
    for u,v in edges:
        g[u][v]['inv_weight'] = (1.0 / (g[u][v]['weight'] / neighbour_radius)) * 0.5
    edge_width = [g[u][v]['inv_weight'] for u,v in edges]

    # 4. Create a layout for our nodes 
    layout = nx.spring_layout(g, iterations=1000, weight='inv_weight', random_state=252)

    # 5. Draw the network
    plt.figure(figsize=(9, 9))
    plt.title('Station network for {}'.format(station_brand))
    nx.draw(g, layout, width=edge_width)
    nx.draw_networkx_labels(g, layout, font_size=11, font_family='sans-serif');
    return g
