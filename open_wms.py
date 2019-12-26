# from PIL import Image
from io import StringIO
# import requests
from owslib.wms import WebMapService

    
def check_blank(content):
    """
    Uses PIL (Python Imaging Library to check if an image is a single colour - i.e. blank
    Images are loaded via a binary content string
    Checks for blank images based on the answer at:
    http://stackoverflow.com/questions/1110403/how-can-i-check-for-a-blank-image-in-qt-or-pyqt
    """

    im = Image.open(StringIO(content))
    # we need to force the image to load (PIL uses lazy-loading)
    # otherwise get the following error: AttributeError: 'NoneType' object has no attribute 'bands'
    im.load() 
    bands = im.split()

    # check if the image is completely white or black, if other background colours are used
    # these should be accounted for
    is_all_white = all(band.getextrema() == (255, 255)  for band in bands)
    is_all_black = all(band.getextrema() == (0, 0)  for band in bands)

    is_blank = (is_all_black or is_all_white)

    return is_blank

def get_default_parameters():

    params = {
        'TRANSPARENT': 'TRUE',
        'SERVICE': 'WMS',
        'VERSION': '1.1.1',
        'REQUEST': 'GetMap',
        'STYLES': '',
        'FORMAT': 'image/png',
        'WIDTH': '256',
        'HEIGHT': '256'}

    return params

def get_params_and_bounding_box(url, layer_name):

    params = get_default_parameters()
    
    # get bounding box for the layer
    wms = WebMapService(url, version='1.1.1')
    bounds = wms[layer_name].boundingBox

    if bounds is None:
        # some WMS servers only support a WGS84 boundingbox
        bounds = wms[layer_name].boundingBoxWGS84
        crs = 'EPSG:4326'
    else:
        # a bounding box and projection code are returned in the following
        # format: (0.0, 0.0, 500000.0, 500000.0, 'EPSG:29902')
        crs = bounds[4]
        
    bbox = ",".join([str(b) for b in bounds[:4]])

    # set the custom parameters for the layer
    params['LAYERS'] = layer_name
    params['BBOX'] = bbox
    params['SRS'] = crs

    return params

def check_blank_image(url, layer_name):
    """
    Check if the WMS layer at the WMS server specified in the
    URL returns a blank image when at the full extent
    """

    params = get_params_and_bounding_box(url, layer_name)
    resp = requests.get(url, params=params)
    print ("The full URL request is '%s'" % resp.url)

    # this should be 200
    print ("The HTTP status code is: %i" % resp.status_code)

    if resp.headers['content-type'] == 'image/png':
        # a PNG image was returned
        is_blank = check_blank(resp.content)
        if is_blank:
            print ("A blank image was returned!")
        else:
            print ("The image contains data.")
    else:
        # if there are errors then these can be printed out here
        print (resp.content)

def get_layers():
    """
    Get a list of all the WMS layers available on the server
    and loop through each one to check if it is blank
    """
   
    wms = WebMapService(url, version='1.3.0')
    layer_names = list(wms.contents)
    print(layer_names)
    for l in layer_names:
        print ("Checking '%s'..." % l)
        # check_blank_image(url, l)
    
if __name__ == "__main__":
    # set the URL of the WMS server here
    url = 'http://mapas.dgterritorio.pt/wms-inspire/cos2015v1'
    get_layers()