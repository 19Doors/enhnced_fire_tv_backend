import pymongo
import requests
import json

mongo_url = "mongodb://firetv:password@34.47.135.240:27017"
client = pymongo.MongoClient(mongo_url)

db = client.get_database("firetv_content")
content = db.get_collection("content")

netflix = requests.get('http://34.47.135.240:8080/content-aggregation/getContentNetflix')
prime = requests.get('http://34.47.135.240:8080/content-aggregation/getContentPrime')
hotstar = requests.get('http://34.47.135.240:8080/content-aggregation/getContentHotstar')

netflixO = json.loads(netflix.text)
primeO = json.loads(prime.text)
hotstarO = json.loads(hotstar.text)

all_content = []
for i in netflixO['content']:
    all_content.append(i)

for i in primeO['content']:
    all_content.append(i)

for i in hotstarO['content']:
    all_content.append(i)

content.insert_many(all_content)
