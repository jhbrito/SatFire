import requests
from requests.auth import HTTPDigestAuth
import json

##################################################################################################################
#API Obligatory Parameters: (https://www.worldweatheronline.com/developer/premium-api-explorer.aspx)
#key=49ce45bfb2c0498cb5b134052190308
#q=Viana do Castelo
#format=json

# class prepare_url:
#   def __init__(self, key, local, file_format):
#     self.base_url = "http://api.worldweatheronline.com/premium/v1/past-weather.ashx?"
#     self.key = key
#     self.local = local
#     self.file_format = file_format

#   def build_url(self):
#     self.url = self.base_url + '&' + self.key + '&' + self.local + '&' + self.file_format
#     return self.url

# p1 = prepare_url("49ce45bfb2c0498cb5b134052190308", "Viana do Castelo", "json" )
# url = p1.build_url()
# print(url)
########################################################################################################

# Replace with the correct URL
url = "http://api.worldweatheronline.com/premium/v1/past-weather.ashx?key=49ce45bfb2c0498cb5b134052190308&q=Viana do Castelo&format=json"

# It is a good practice not to hardcode the credentials. So ask the user to enter credentials at runtime
myResponse = requests.get(url)#,auth=HTTPDigestAuth(input("username: "), input("Password: ")), verify=True)
#print (myResponse.status_code)

# For successful API call, response code will be 200 (OK)
if(myResponse.ok):

    # Loading the response data into a dict variable
    # json.loads takes in only binary or string variables so using content to fetch binary content
    # Loads (Load String) takes a Json file and converts into python data structure (dict or list, depending on JSON)
    jData = json.loads(myResponse.content)

    print("The response contains {0} properties".format(len(jData)))
    print("\n")
    for key in jData:
        print(str(key) + " : " + str(jData[key]))
else:
  # If response code is not ok (200), print the resulting http error code with description
    myResponse.raise_for_status()