import logging
import time
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from app import config
from app.inference import DefectClassifier, load_image

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("casting-api")

state = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    state["clf"] = DefectClassifier(config.MODEL_PATH)   # load once at startup
    yield
    state.clear()


app = FastAPI(
    title="Casting Defect Detection API",
    description="Upload a casting product image and get Normal / Defective with a confidence score.",
    version="1.0.0",
    lifespan=lifespan,
)


class PredictionResponse(BaseModel):
    request_id: str
    predicted_class: str
    confidence: float
    defect_probability: float
    threshold: float
    latency_ms: float


@app.middleware("http")
async def log_requests(request: Request, call_next):
    request_id = uuid.uuid4().hex[:12]
    request.state.request_id = request_id
    start = time.perf_counter()
    try:
        response = await call_next(request)
    except Exception:
        logger.exception("[%s] Unhandled error on %s", request_id, request.url.path)
        return JSONResponse(status_code=500,
                            content={"detail": "Internal server error", "request_id": request_id})
    ms = (time.perf_counter() - start) * 1000
    response.headers["X-Request-ID"] = request_id
    logger.info("[%s] %s %s -> %d (%.1f ms)", request_id, request.method,
                request.url.path, response.status_code, ms)
    return response


@app.get("/health", tags=["ops"])
def health():
    return {"status": "ok", "model_loaded": "clf" in state}


@app.post("/predict", response_model=PredictionResponse, tags=["inference"])
async def predict(request: Request, file: UploadFile = File(...)):
    rid = request.state.request_id

    if file.content_type not in config.ALLOWED_CONTENT_TYPES:
        raise HTTPException(415, f"Unsupported type '{file.content_type}'. "
                                 f"Allowed: {sorted(config.ALLOWED_CONTENT_TYPES)}")

    max_bytes = config.MAX_UPLOAD_MB * 1024 * 1024
    data = await file.read(max_bytes + 1)         # read at most limit + 1 byte
    if len(data) == 0:
        raise HTTPException(400, "Empty file.")
    if len(data) > max_bytes:
        raise HTTPException(413, f"File exceeds {config.MAX_UPLOAD_MB} MB limit.")

    try:
        image = load_image(data, config.MIN_IMAGE_SIDE, config.MAX_IMAGE_SIDE)
    except ValueError as e:
        logger.warning("[%s] Rejected image: %s", rid, e)
        raise HTTPException(400, str(e))

    start = time.perf_counter()
    result = await run_in_threadpool(state["clf"].predict, image)
    latency = round((time.perf_counter() - start) * 1000, 2)

    logger.info("[%s] prediction=%s conf=%.3f inference=%.1f ms",
                rid, result["predicted_class"], result["confidence"], latency)
    return PredictionResponse(request_id=rid, latency_ms=latency, **result)