import io
from datetime import datetime
from functools import wraps
from http import HTTPStatus
from pathlib import Path
from typing import Dict, List

from fastapi import FastAPI, File, Request, UploadFile
from fastapi.encoders import jsonable_encoder
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image

from config import config
from config.config import logger
from plantcare import main, predict
from plantcare.utils import utils

# Define application
app = FastAPI(
    title="PlantCare-MLOps",
    description="Check Plant Health",
    version="0.1",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def construct_response(f):
    """Construct a JSON response for an endpoint."""

    @wraps(f)
    def wrap(request: Request, *args, **kwargs) -> Dict:
        results = f(request, *args, **kwargs)
        response = {
            "message": results["message"],
            "method": request.method,
            "status-code": results["status-code"],
            "timestamp": datetime.now().isoformat(),
            "url": request.url._url,
        }
        if "data" in results:
            response["data"] = results["data"]
        return response

    return wrap


@app.get("/", tags=["General"])
@construct_response
def _index(request: Request) -> Dict:
    """Health check."""
    response = {
        "message": HTTPStatus.OK.phrase,
        "status-code": HTTPStatus.OK,
        "data": {},
    }
    return response

@app.on_event("startup")
def load_artifacts():
    global artifacts
    run_id = open(Path(config.CONFIG_DIR, "run_id.txt")).read()
    artifacts = main.load_artifacts(run_id= run_id, mode="test")
    logger.info("Ready for inference!")


@app.get("/performance", tags=["Performance"])
@construct_response
def _performance(request: Request, filter: str = None) -> Dict:
    """Get the performance metrics."""
    data = {"performance": artifacts["performance"].get(filter, artifacts["performance"])}
    response = {
        "message": HTTPStatus.OK.phrase,
        "status-code": HTTPStatus.OK,
        "data": data,
    }
    return response


@construct_response
@app.post("/predict", tags=["Prediction"])
async def _predict(request: Request, files: List[UploadFile] = File(...)):
    """Predict tags for a list of texts."""
    images = []
    image_names = []
    
    for file in files:
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        images.append(image)
        image_names.append(file.filename)
        
    predictions = predict.predict(images, artifacts=artifacts)

    results = {}
    for i in range(len(image_names)):
        if len(predictions[i]) > 0:
            results[image_names[i]] = [str(p) for p in predictions[i]]
        else:
            results[image_names[i]] = ["Nothing found"]
    
    response = {
        "message": HTTPStatus.OK.phrase,
        "status-code": HTTPStatus.OK,
        "data": results,
    }
    return response


@app.get("/args", tags=["Arguments"])
@construct_response
def _args(request: Request) -> Dict:
    """Get all arguments used for the run."""
    response = {
        "message": HTTPStatus.OK.phrase,
        "status-code": HTTPStatus.OK,
        "data": {
            "args": utils.convert_namespace_to_dict(artifacts["args"]),
        },
    }
    return response
