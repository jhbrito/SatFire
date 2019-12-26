import xlrd
import requests
import json

ExcelFileName = r'C:\Users\Guilhe5\Desktop\Tese\dados_tese\dados_2017.xlsx'
workbook = xlrd.open_workbook(ExcelFileName)
worksheet = workbook.sheet_by_name("dados_2017") # We need to read the data
date_begin = worksheet.col_values(17,1) 
date_end = worksheet.col_values(36,1) 
workbook_datemode = workbook.datemode
hour = worksheet.col_values(18,1)
lat = worksheet.col_values(26,1) 
longt = worksheet.col_values(27,1) 
local_e = worksheet.col_values(21,1) 
for i in range(worksheet.nrows - 1):
# treat begin_date field
    y_b, m_b, d_b, h_b, mi_b, s_b = xlrd.xldate_as_tuple(date_begin[i], workbook_datemode)
    month_b = '{:02d}'.format(m_b)
    day_b = '{:02d}'.format(d_b)
    year_b = '{:04d}'.format(y_b)
    # print('{:04d}-{:02d}-{:02d}'.format(y_b, m_b, d_b))
# treat end_date field
    y_e, m_e, d_e, h_e, mi_e, s_e = xlrd.xldate_as_tuple(date_end[i], workbook_datemode)
    month_e = '{:02d}'.format(m_e)
    day_e = '{:02d}'.format(d_e)
    year_e = '{:04d}'.format(y_e)
    # print('{:04d}-{:02d}-{:02d}'.format(y_e, m_e, d_e))
# treat latitude and longitude
    f_lat = str(lat[i]).replace('.',',')    
    f_longt = str(longt[i]).replace('.',',')    
    base_url = 'http://api.worldweatheronline.com/premium/v1/past-weather.ashx?'
    key = 'key=49ce45bfb2c0498cb5b134052190308'
    # local = 'q=' + f_lat + '/' + f_longt
    local = 'q=' + str(local_e[i])
    fformat = 'format=json'
    date = 'date=' + '{:04d}-{:02d}-{:02d}'.format(y_b, m_b, d_b)
    enddate = 'enddate=' + '{:04d}-{:02d}-{:02d}'.format(y_e, m_e, d_e)

    url = base_url + key + '&' + local + '&' + fformat + '&' + date + '&' + enddate

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


