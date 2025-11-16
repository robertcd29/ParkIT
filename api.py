from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
from sqlmodel import Field, SQLModel, Relationship, create_engine, Session, select
import psycopg2
from psycopg2.extras import RealDictCursor
import asyncio
import traceback
from decimal import Decimal
from starlette.websockets import WebSocketDisconnect, WebSocketState


class ParkingCoordinate(SQLModel, table=True):
    __tablename__ = "parking_coordinates"
    
    coord_id: Optional[int] = Field(default=None, primary_key=True)
    latitude: float
    longitude: float
    point_order: int
    parking_number: int = Field(foreign_key="parking_spots.parking_number")
    spot: "ParkingSpot" = Relationship(back_populates="coordinates")


class ParkingSpot(SQLModel, table=True):
    __tablename__ = "parking_spots"
    
    parking_number: Optional[int] = Field(default=None, primary_key=True)
    parking_name: Optional[str]
    empty_spots: int
    occupied_spots: int
    total_spots: int                     
    price_per_hour: Optional[float] = None
    schedule: Optional[str] = None
    has_surveillance: Optional[bool] = True 
    has_disabled_access: Optional[bool] = True
    has_ev_charging: Optional[bool] = True

    coordinates: List["ParkingCoordinate"] = Relationship(back_populates="spot")



class CoordinateRead(SQLModel):
    latitude: float
    longitude: float
    point_order: int


class ParkingSpotRead(SQLModel):
    parking_number: int
    parking_name: Optional[str]
    empty_spots: int
    occupied_spots: int 
    total_spots: int
    price_per_hour: Optional[float] = None 
    schedule: Optional[str] = None 
    has_surveillance: Optional[bool] = True
    has_disabled_access: Optional[bool] = True
    has_ev_charging: Optional[bool] = True
    coordinates: List[CoordinateRead]



class ParkingCoordinateIn(SQLModel):
    latitude: float
    longitude: float
    point_order: int

class ParkingSpotCreate(SQLModel):
    parking_name: Optional[str]
    total_spots: int                      
    coordinates: List[ParkingCoordinateIn]
    empty_spots: int = 0
    occupied_spots: int = 0
    price_per_hour: Optional[float] = None
    schedule: Optional[str] = None
    has_surveillance: Optional[bool] = True
    has_disabled_access: Optional[bool] = True
    has_ev_charging: Optional[bool] = True

class ParkingSpotUpdate(BaseModel):
    parking_number: int | None = None
    name: Optional[str] = None
    empty_spots: int | None = 0
    occupied_spots: int | None = 0
    price_per_hour: Optional[float] = None
    schedule: Optional[str] = None
    total_spots: int | None = None

class DetectionData(BaseModel):
    parking_number: int
    free_spots: int
    total_spots: int


DATABASE_URL = "postgresql://postgres:d@localhost:5432/smart_park"
engine = create_engine(DATABASE_URL, echo=True)


def get_session():
    with Session(engine) as session:
        yield session



app = FastAPI()
active_connections: List[WebSocket] = []
@app.websocket("/ws/parking")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    active_connections.append(websocket)
    try:
        while True:
            data = await websocket.receive_text()
            for connection in active_connections:
                try:
                    await connection.send_text(f"Update: {data}")
                except:
                    if connection in active_connections:
                        active_connections.remove(connection)
    except WebSocketDisconnect:
        if websocket in active_connections:
            active_connections.remove(websocket)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def get_db_connection():
    try:
        conn = psycopg2.connect(
            dbname="smart_park",
            user="postgres",
            password="d",
            host="localhost",
            port="5432",
            cursor_factory=RealDictCursor,
            sslmode="require"
        )
        return conn
    except Exception as e:
        print(f"EROARE CRITICĂ DB: {e}")
        raise e


@app.get("/api/v1/parcari/all")
def get_all_parking_data(session: Session = Depends(get_session)):
    try:
        statement = select(ParkingSpot)
        results = session.exec(statement).all()
        
        final_results = []
        for spot in results:
            spot.coordinates.sort(key=lambda coord: coord.point_order)
            spot_dict = {
                "parking_number": spot.parking_number,
                "parking_name": spot.parking_name,
                "empty_spots": spot.empty_spots,
                "occupied_spots": spot.occupied_spots,
                "total_spots": spot.total_spots,
                "price_per_hour": float(spot.price_per_hour) if spot.price_per_hour is not None else None,
                "schedule": spot.schedule,
                "has_surveillance": spot.has_surveillance,
		        "has_disabled_access": spot.has_disabled_access,
		        "has_ev_charging": spot.has_ev_charging,
                "coordinates": [
                    {
                        "latitude": float(c.latitude),
                        "longitude": float(c.longitude),
                        "point_order": c.point_order
                    } for c in spot.coordinates
                ]
            }
            final_results.append(spot_dict)
        return final_results
    except Exception as e:
        traceback.print_exc()
        return {"status": "error", "message": str(e)}
        

@app.post("/api/v1/parcari", status_code=201) 
def create_parking_spot(spot_data: ParkingSpotCreate, session: Session = Depends(get_session)):
    try:
        parking_spot_dict = spot_data.model_dump(exclude={"coordinates"})
        parking_spot = ParkingSpot(**parking_spot_dict)
        
        session.add(parking_spot)
        session.commit()
        session.refresh(parking_spot)
        
        for coord_in in spot_data.coordinates:
            parking_coordinate = ParkingCoordinate(
                latitude=coord_in.latitude,
                longitude=coord_in.longitude,
                point_order=coord_in.point_order,
                parking_number=parking_spot.parking_number
            )
            session.add(parking_coordinate)
        session.commit()
        return {"status": "success", "parking_number": parking_spot.parking_number}
    except Exception as e:
        traceback.print_exc()
        return {"status": "error", "message": str(e)}



@app.websocket("/ws/parking")   
async def websocket_parking(websocket: WebSocket):
    await websocket.accept()
    print("Client connected to /ws/parking")
    
    try:
        while True:
            try:
                conn = get_db_connection()
                cur = conn.cursor()
                cur.execute("""
                    SELECT 
                        ps.parking_number,
                        ps.parking_name,
                        ps.empty_spots,
                        ps.occupied_spots,
                        ps.total_spots,
                        ps.price_per_hour,
                        ps.schedule
			            ps.has_surveillance,
			            ps.has_disabled_access, 
			            ps.has_ev_charging,,
                        COALESCE(
                            json_agg(
                                json_build_object(
                                    'latitude', pc.latitude,
                                    'longitude', pc.longitude,
                                    'point_order', pc.point_order
                                ) ORDER BY pc.point_order
                            ) FILTER (WHERE pc.coord_id IS NOT NULL),
                            '[]'::json
                        ) as coordinates
                    FROM parking_spots ps
                    LEFT JOIN parking_coordinates pc ON ps.parking_number = pc.parking_number
                    GROUP BY ps.parking_number, ps.parking_name, ps.empty_spots, 
                             ps.occupied_spots, ps.total_spots, ps.price_per_hour, ps.schedule, 
                             ps.has_surveillance, ps.has_disabled_access, ps.has_ev_charging

                    ORDER BY ps.parking_number
                """)
                rows = cur.fetchall()
                cur.close()
                conn.close()
                
                serializable_rows = []
                for row in rows:
                    clean_row = {
                        k: (float(v) if isinstance(v, Decimal) else v) 
                        for k, v in row.items()
                    }
                    serializable_rows.append(clean_row)
                
                try:
                    await websocket.send_json(serializable_rows)
                except Exception:
                    break
                    
            except Exception as db_e:
                # Doar erori de bază de date (nu de WebSocket)
                if "ClientDisconnected" not in str(type(db_e).__name__):
                    print(f"WebSocket DB Error: {db_e}")
                    traceback.print_exc()
                else:
                    # Client deconectat
                    break
                
            await asyncio.sleep(5)
            
    except WebSocketDisconnect:
        print("Client deconectat normal")
    except Exception as e:
        if "ClientDisconnected" not in str(type(e).__name__):
            print(f"Eroare neașteptată în WebSocket: {e}")
            traceback.print_exc()
    finally:
        print("🔌 Conexiune WebSocket închisă")



@app.get("/")
def read_root():
    return {
        "status": "FastAPI Running!",
        "endpoints": [
            "/api/v1/parcari (POST - create spot)",
            "/api/v1/parcari/all (GET - recommended)",
            "/ws/parking (WebSocket live updates)"
        ]
    }


@app.post("/api/detection")
def receive_detection(data: DetectionData):
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("SELECT total_spots FROM parking_spots WHERE parking_number = %s", (data.parking_number,))
        result = cur.fetchone()
        if not result:
            cur.close()
            conn.close()
            return {"status": "error", "message": f"Parking {data.parking_number} not found!"}

        total_spots = result["total_spots"]
        occupied_spots = total_spots - data.free_spots

        cur.execute(
            "UPDATE parking_spots SET empty_spots=%s, occupied_spots=%s WHERE parking_number=%s",
            (data.free_spots, occupied_spots, data.parking_number)
        )
        conn.commit()
        cur.close()
        conn.close()

        return {
            "status": "success",
            "data": {
                "parking_number": data.parking_number,
                "empty_spots": data.free_spots,
                "occupied_spots": occupied_spots,
                "total_spots": total_spots
            }
        }
    except Exception as e:
        traceback.print_exc()
        return {"status": "error", "message": str(e)}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)