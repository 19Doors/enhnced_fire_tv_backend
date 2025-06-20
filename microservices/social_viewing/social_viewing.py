import json
from typing import Any, Dict, List, Optional
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
import uuid
import datetime
from pydantic import BaseModel

class ConnectionManager:
    def __init__(self):
        # Store connections by room_id
        self.active_connections: Dict[str, List[WebSocket]] = {}

    async def connect(self, websocket: WebSocket, room_id: str):
        await websocket.accept()
        if room_id not in self.active_connections:
            self.active_connections[room_id] = []
        self.active_connections[room_id].append(websocket)
        print(f"✅ Connection added to room {room_id}. Total: {len(self.active_connections[room_id])}")

    def disconnect(self, websocket: WebSocket, room_id: str):
        if room_id in self.active_connections:
            if websocket in self.active_connections[room_id]:
                self.active_connections[room_id].remove(websocket)
                print(f"🔌 Connection removed from room {room_id}. Remaining: {len(self.active_connections[room_id])}")

    async def broadcast_to_room(self, room_id: str, message: dict, exclude: WebSocket = None):
        if room_id not in self.active_connections:
            return
        
        message_str = json.dumps(message)
        dead_connections = []
        
        print(f"📡 Broadcasting to {len(self.active_connections[room_id])} connections in room {room_id}")
        
        for connection in self.active_connections[room_id]:
            if connection != exclude:
                try:
                    await connection.send_text(message_str)
                    print(f"✅ Message sent to connection")
                except Exception as e:
                    print(f"❌ Failed to send to connection: {e}")
                    dead_connections.append(connection)
        
        # Remove dead connections
        for dead_connection in dead_connections:
            self.disconnect(dead_connection, room_id)

    def get_connection_count(self, room_id: str) -> int:
        return len(self.active_connections.get(room_id, []))

class CreateRoom(BaseModel):
    host_user_id: str
    content_id: str
    room_name: str
    content: Dict[str,Any]

class JoinRoom(BaseModel):
    room_id: str
    user_id: str

app = FastAPI(title="Social Viewing Service")
manager = ConnectionManager()

class RoomManager:
    def __init__(self):
        self.rooms = {}

    def create_room(self, host_user_id: str, content_id: str, room_name: str, content):
        room_id = str(uuid.uuid4())[:8]

        room_data = {
            "room_id": room_id,
            "host_user_id": host_user_id,
            "content_id": content_id,
            "content": content,
            "room_name": room_name,
            "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "participants": [host_user_id],
            "max_participants": 10,
            "is_active": True,
        }

        self.rooms[room_id] = room_data

        return {
                "room_id": room_id,
                "room_data": room_data
                }

    def join_room(self, room_id: str, user_id: str):
        if room_id not in self.rooms:
            raise HTTPException(status_code=404, detail="Room not found")

        room = self.rooms[room_id]

        if not room["is_active"]:
            raise HTTPException(status_code=410, detail="Room is no longer active")

        if user_id not in room["participants"]:
            room["participants"].append(user_id)

        return room

    def get_room_info(self, room_id: str) -> Dict:
        if room_id not in self.rooms:
            raise HTTPException(status_code=404, detail="Room not found")

        return self.rooms[room_id]


room_manager = RoomManager()

@app.post("/create-room")
async def create_room(rq: CreateRoom):
    host_user_id = rq.host_user_id
    room_name = rq.room_name
    content_id = rq.content_id
    content = rq.content

    try:
        result = room_manager.create_room(host_user_id=host_user_id, room_name=room_name, content_id=content_id, content=content)
        return {
                "success": True,
                "room_id": result["room_id"],
                "room_name": room_name
                }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/join-room")
async def join_room(rq: JoinRoom):
    room_id = rq.room_id
    user_id = rq.user_id
    try:
        room_info = room_manager.join_room(room_id, user_id)
        return {
            "success": True,
            "room": room_info,
            "message": f"Successfully joined {room_info['room_name']}"
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.websocket("/ws/watch/{room_id}")
async def websocket_endpoint(websocket: WebSocket, room_id: str):
    await manager.connect(websocket= websocket, room_id=room_id)
    try:
        print("WebSocket Connected")
        await manager.broadcast_to_room(room_id, {
            "type": "user_joined",
            "message": "A user joined the room",
            "participant_count": manager.get_connection_count(room_id)
        }, exclude=websocket)
        print("✅ User joined notification sent")

        while True:
            try:
                print("🔄 Waiting for message...")
                data = await websocket.receive_text()
                print(f"📨 Raw data received from room {room_id}: {data}")
                
                message = json.loads(data)
                print(f"🔍 Parsed message: {message}")
                print(f"📋 Message type: {message.get('type')}")

                if message.get('type') == 'playback_sync':
                    await handle_playback_sync(room_id=room_id, message=message, websocket=websocket)
                
            except json.JSONDecodeError as e:
                print(f"❌ JSON decode error: {e}")
                await websocket.send_text(json.dumps({
                    "type": "error",
                    "message": "Invalid JSON format"
                }))
            except WebSocketDisconnect:
                print(f"🔌 WebSocket disconnected from room: {room_id}")
                break
            except Exception as e:
                print(f"❌ Error processing message: {e}")
                break
                
    except WebSocketDisconnect:
        print(f"🔌 Connection closed for room: {room_id}")
    except Exception as e:
        print(f"❌ WebSocket error: {e}")
    finally:
        manager.disconnect(websocket, room_id)
        await manager.broadcast_to_room(room_id, {
            "type": "user_left",
            "message": "A user left the room",
            "participant_count": manager.get_connection_count(room_id)
        })


async def handle_playback_sync(room_id: str, message: dict, websocket: WebSocket):
    if room_id in room_manager.rooms:
        room_manager.rooms[room_id]["playback_state"] = {
            "is_playing": message.get("is_playing", False),
            "current_time": message.get("currentTime", 0)
        }

    await manager.broadcast_to_room(room_id, {
        "type": "playback_sync",
        "is_playing": message.get("is_playing"),
        "current_time": message.get("currentTime")
    }, exclude=websocket)
