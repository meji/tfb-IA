from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from utils import generate_image_async
import asyncio
import uuid
import logging

app = FastAPI()

# Permitir CORS para todos los orígenes (solo durante el desarrollo)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Montar la carpeta static para servir archivos estáticos
app.mount("/static", StaticFiles(directory="static"), name="static")

class ImageRequest(BaseModel):
    style: str
    prompt: str

@app.post("/generate_image/")
async def generate_image_endpoint(request: ImageRequest):
    try:
        logging.info(f"Received request for style: {request.style}, prompt: {request.prompt}")

        # Crear y almacenar la nueva tarea de generación de imagen
        image_path = await generate_image_async(request.style, request.prompt)
        
        return FileResponse(image_path, media_type="image/png")
    except ValueError as e:
        logging.error(f"ValueError: {e}")
        return JSONResponse(status_code=400, content={"detail": str(e)})
    except Exception as e:
        logging.error(f"Unexpected error: {e}")
        return JSONResponse(status_code=500, content={"detail": "Internal Server Error"})

@app.get("/", response_class=HTMLResponse)
async def read_index():
    with open("static/index.html") as f:
        return f.read()

# Ejecución del servidor de desarrollo
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
