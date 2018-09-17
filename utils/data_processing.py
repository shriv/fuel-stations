import pandas as pd
import requests
import os


##############
## OVERPASS ##
##############

def generate_overpass_query(tags, objects,
                            osm_bbox,
                            entity="amenity"):
    """
    Generate and return Overpass query string
 
    Args:
     tags: list of tags (e.g. 'fuel')
     objects: list of objects (e.g. nodes, ways)
     osm_bbox: vertex list of OSM bounding box convention. Order is: (S, W, N, E)
     entity: querying entity type (amenity by default)
    
    Returns:
     compactOverpassQLstring: query string
    """

    compactOverpassQLstring = '[out:json][timeout:60];('
    for tag in tags:
        for obj in objects:
            compactOverpassQLstring += '%s["%s"="%s"](%s,%s,%s,%s);' % (obj, entity, tag,
                                                                        osm_bbox[0],
                                                                        osm_bbox[1],
                                                                        osm_bbox[2],
                                                                        osm_bbox[3])
    compactOverpassQLstring += ');out body;>;out skel qt;'    
    return compactOverpassQLstring

    
def get_osm_data(compactOverpassQLstring, osm_bbox):
    """
    Get Data from OSM via Overpass. Convert JSON to Pandas dataframe. Save.
    If data has been downloaded previously, read from csv

    Args:
     compactOverpassQLstring: Query string
     osm_bbox: OSM-spec'd bounding box as list  

    Returns:
     osmdf: pandas dataframe of extracted JSON 
    """

    # Filename
    bbox_string = '_'.join([str(x) for x in osm_bbox])
    osm_filename = 'data/osm_data_{}.csv'.format(bbox_string)

    if os.path.isfile(osm_filename):
        osm_df = pd.read_csv(osm_filename)

    else:
        # Request data from Overpass
        osmrequest = {'data': compactOverpassQLstring}
        osmurl = 'http://overpass-api.de/api/interpreter'

        # Ask the API
        osm = requests.get(osmurl, params=osmrequest)

        # Convert the results to JSON and get the requested data from the 'elements' key
        # The other keys in osm.json() are metadata guff like 'generator', 'version' of API etc. 
        osmdata = osm.json()
        osmdata = osmdata['elements']
        # Convert JSON output to pandas dataframe
        for dct in osmdata:
            if dct.has_key('tags'):
                for key, val in dct['tags'].iteritems():
                    dct[key] = val
                del dct['tags']
            else:
                pass
        osm_df = pd.DataFrame(osmdata)
        osm_df.to_csv(osm_filename)
        
    return osm_df
