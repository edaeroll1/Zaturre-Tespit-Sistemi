from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from .routers import predict
from .services.model_service import preload_all_models


@asynccontextmanager
async def lifespan(app: FastAPI):
    preload_all_models()
    yield


app = FastAPI(
    title="Zatürre Tespit Sistemi API",
    description="Göğüs röntgeni görüntülerinden zatürre tespiti yapan REST API.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    return JSONResponse(
        status_code=500,
        content={"detail": "Beklenmeyen bir hata oluştu."},
    )


app.include_router(predict.router)


@app.get("/", tags=["Sağlık"])
async def root():
    return {"status": "ok", "message": "Zatürre Tespit Sistemi API çalışıyor."}


@app.get("/health", tags=["Sağlık"])
async def health():
    return {"status": "ok"}
