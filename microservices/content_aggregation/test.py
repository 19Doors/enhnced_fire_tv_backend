import json
import os
from typing import Any, Dict, Optional
import fastapi
from datetime import datetime, time
import pymongo
from pydantic import BaseModel

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

@app.on_event("startup")
async def startup_event():
    mongo_url = os.getenv("MONGO_URL")
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

@app.post("/track-interaction")
async def tract_interaction_click(interaction: UserInteraction):
    try:
        print(interaction.watchProgress)
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
        print(interaction_doc)
        firetv = client.get_database("firetv_content")
        user_interaction = firetv.get_collection("user_interaction")
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
