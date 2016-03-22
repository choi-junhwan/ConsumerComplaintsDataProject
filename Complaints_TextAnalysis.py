import requests
import simplejson as json

def read_data():
    MyAPIKey  =  "MyKgbhd5QyIT2ILDjnyNF1iZy"
    api_url   =  'https://data.consumerfinance.gov/resource/jhzv-w97w.json'
    json_data =  requests.get(api_url)
    print json_data.json().key





if __name__=="__main__":
    read_data()
