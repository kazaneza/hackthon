from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from .stt import router as stt_router
from .tts import router as tts_router

def create_app():
    app = FastAPI(title="Speech-to-Text API")
    
    # Configure CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Include routers
    app.include_router(stt_router)
    app.include_router(tts_router)
    
    @app.get("/health")
    async def health_check():
        return {"status": "healthy"}
    
    return app