from fastapi import FastAPI, status, HTTPException
from pydantic import BaseModel
from predict import predict_data

app = FastAPI()

class CancerData(BaseModel):
    mean_radius: float
    mean_texture: float
    mean_perimeter: float
    mean_area: float
    mean_smoothness: float
    mean_compactness: float
    mean_concavity: float
    mean_concave_points: float
    mean_symmetry: float
    mean_fractal_dimension: float

class CancerResponse(BaseModel):
    prediction: str

@app.get("/", status_code=status.HTTP_200_OK)
async def health_ping():
    return {"status": "healthy"}

@app.post("/predict", response_model=CancerResponse)
async def predict_cancer(features: CancerData):
    try:
        feature_values = [list(features.dict().values())]
        prediction = predict_data(feature_values)
        label = "Malignant" if prediction[0] == 0 else "Benign"
        return CancerResponse(prediction=label)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
