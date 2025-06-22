import os
from typing import Any, Dict, Optional
import fastapi
from datetime import datetime, time
import pymongo
from pydantic import BaseModel
import json
from kafka_content import RecommendationProducer


producer = RecommendationProducer()
print("Producer Loaded")
client = None
app = fastapi.FastAPI()
# Pydantic model for request validation
class UserInteraction(BaseModel):
    user_id: str
    interaction_type:str
    content_id: str
    content_type: str
    content_platform: str
    watchProgress: Optional[float] = 0.0
    context_data: Optional[Dict[str, Any]] = {}

class Recommendation(BaseModel):
    user_id: str

@app.on_event("startup")
async def startup_event():
    mongo_url = os.getenv("MONGO_URL")
    # mongo_url = os.getenv("mongodb://firetv:password@34.47.135.240:27017")
    """Initialize MongoDB client when server starts"""
    global client
    try:
        # client = pymongo.MongoClient("mongodb://firetv:password@localhost:27017")
        client = pymongo.MongoClient(mongo_url)
        client.admin.command('ping')
        print("✅ MongoDB client connected successfully")

    except Exception as e:
        print(f"❌ Failed to connect to MongoDB: {e}")
        raise e

@app.get("/getContentNetflix")
def contentNetflix():
    with open("netflix_content.json","r") as f:
        netflix=f.read()

    netflixO = json.loads(netflix)

    netflixContent = netflixO['content']
    return {"content": netflixContent}

@app.get("/getContentPrime")
def contentPrime():
    with open("prime_video_content.json","r") as f:
        prime=f.read()

    primeO= json.loads(prime)

    primeContent = primeO['content']
    return {"content": primeContent}

@app.get("/getContentHotstar")
def contentHotstar():
    with open("hotstar_content.json","r") as f:
        hotstar=f.read()

    hotstarO= json.loads(hotstar)

    hotstarContent = hotstarO['content']
    return {"content": hotstarContent}

@app.post("/getRecommendation")
def contentRecommended(req: Recommendation):
    try: 
        col = client.get_database("firetv_content").get_collection(f"user_{req.user_id}_recommendations")
        content_all = client.get_database("firetv_content").get_collection("content")
        cursor = col.find({})

        content = []
        for doc in cursor:
            if '_id' in doc:
                doc['_id'] = str(doc['_id'])
            content.append(doc)

        ans = []
        real_return = []
        for i in content:
            content_id = i['content_id']
            fetch_content = content_all.find_one({'id':content_id})
            ans.append(fetch_content)

        for doc in ans:
            if '_id' in doc:
                doc['_id'] = str(doc['_id'])
            real_return.append(doc)

        return {"content": real_return}

    except Exception as e:
        return {"ERROR": e}
@app.post("/track-interaction")
async def tract_interaction_click(interaction: UserInteraction):
    try:
        interaction_doc = {
            "user_id": interaction.user_id,
            "event_type": interaction.interaction_type,
            "content_type": interaction.content_type,
            "content_id": interaction.content_id,
            "content_platform": interaction.content_platform,
            "context_data": interaction.context_data,
            "timestamp": datetime.now()
        }
        if interaction.interaction_type == "watch":
            interaction_doc["watchProgress"] = interaction.watchProgress
        firetv = client.get_database("firetv_content")
        user_interaction = firetv.get_collection("user_interaction")
        print(interaction_doc)
        producer.send_for_recommendation(user_id=interaction.user_id)
        result = user_interaction.insert_one(interaction_doc)
        return {"status":"success", "interaction_id": str(result.inserted_id)}
    except Exception as e:
        return {"Error": e};

# netflix=""
# hotstar=""
# prime=""
#
# all_items = []
# for item in netflixContent:
#     all_items.append(item)
# for item in primeContent:
#     all_items.append(item)
# for item in hotstarContent:
#     all_items.append(item)
#
# print(client.list_database_names())
# dbs = client.list_database_names()
# firetv = client.get_database("firetv_content")
# collection_content = firetv.get_collection("content")
# # collection_content.delete_many({})
# result = collection_content.insert_many(all_items)
# print(f"Done, inserted {len(result.inserted_ids)}")
