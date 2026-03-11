from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np

app = FastAPI(title="Oral Ulcer AI Clinical Risk API")

# --- 1. ENABLE CORS FOR FLUTTER CONNECTION ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins (update with specific IPs in production)
    allow_credentials=True,
    allow_methods=["*"],  # Allows all HTTP methods (POST, GET, etc.)
    allow_headers=["*"],  # Allows all headers
)

# --- 2. LOAD TRAINED MODEL ---
try:
    # Ensure this matches your downloaded file name exactly
    model = joblib.load("clinical_model.pkl") 
except Exception as e:
    print(f"Error loading model: {e}. Ensure 'clinical_model.pkl' is in the same directory.")
    model = None

# -----------------------------
# Input Schema (All Features)
# -----------------------------
class ClinicalInput(BaseModel):
    # Demographics
    age: int
    sex: str
    smoking_status: str
    smoking_duration: int
    smoking_frequency: str
    smokeless_tobacco: int
    alcohol: str
    diabetes: int
    immunocompromised: int
    autoimmune: int
    steroids: int
    chemotherapy: int
    immunosuppressants: int

    # Lesion History
    duration: str
    onset: str
    recurrence: str
    pain: str
    healing_pattern: str

    # Clinical Examination
    site: str
    size_mm: int
    shape: str
    margins: str
    edge: str
    induration: int
    bleeding: int

    # Associated Findings
    lymph_palpable: int
    tender: int
    node_mobility: str
    paraesthesia: int
    weight_loss: int
    fever: int


# -----------------------------
# Risk Classification Logic
# -----------------------------
def classify_risk(score):
    if score >= 75: # Aligned with typical clinical high-risk thresholds
        return "High", "Biopsy strongly indicated"
    elif score >= 40:
        return "Intermediate", "Close follow-up / biopsy if persists"
    else:
        return "Low", "Conservative management"


# -----------------------------
# Enhanced Explanation Generator
# -----------------------------
def generate_explanation(data):
    explanation = []
    
    # Matches features defined in Feature Requirements.docx
    if data.get("duration") in ["> 3 weeks", ">3 weeks"]:
        explanation.append("Duration > 3 weeks")
    if data.get("induration") == 1:
        explanation.append("Induration present on palpation")
    if data.get("margins") == "Ill-defined":
        explanation.append("Ill-defined lesion margins")
    if data.get("node_mobility") == "Fixed":
        explanation.append("Fixed lymph node mobility")
    if data.get("weight_loss") == 1:
        explanation.append("Unexplained weight loss")
    if data.get("smoking_status") in ["Current", "Past"]:
        explanation.append("History of tobacco use")
    if data.get("pain") == "Painless":
        explanation.append("Painless ulcer presentation")
    if "Everted" in data.get("edge", ""):
        explanation.append("Everted lesion edge")
    if data.get("bleeding") == 1:
        explanation.append("Bleeding on touch")
    if data.get("paraesthesia") == 1:
        explanation.append("Paraesthesia or anaesthesia present")
        
    # Check for High-Risk Sites
    site = data.get("site", "")
    if "Lateral" in site or "Ventral" in site or "Floor" in site:
        explanation.append(f"High-risk anatomical site ({site.replace(' ⚠️', '')})")

    if not explanation:
        explanation.append("No immediate high-risk clinical flags identified.")

    return explanation


# -----------------------------
# Clinical Suggestions
# -----------------------------
def generate_suggestions(category):
    if category == "High":
        return [
            "Oral Squamous Cell Carcinoma",
            "Potentially malignant disorder",
            "Severe epithelial dysplasia"
        ]
    elif category == "Intermediate":
        return [
            "Leukoplakia",
            "Chronic traumatic ulcer",
            "Lichen planus"
        ]
    else:
        return [
            "Aphthous ulcer",
            "Minor traumatic ulcer",
            "Benign mucosal lesion"
        ]


# -----------------------------
# Prediction Endpoint
# -----------------------------
@app.post("/predict")
def predict_clinical_risk(input_data: ClinicalInput):
    if model is None:
        raise HTTPException(status_code=500, detail="ML Model is not loaded on the server.")

    # 1. Convert input to dict safely
    input_dict = input_data.model_dump() if hasattr(input_data, 'model_dump') else input_data.dict()
    
    # Generate explanations BEFORE translating the text, so the doctor gets the readable version
    explanation = generate_explanation(input_dict)

    # =========================================================
    # 🔥 THE FIX: TRANSLATE FLUTTER TEXT TO STRICT AI MODEL TEXT
    # =========================================================
    # Fix Duration
    if input_dict.get('duration') == "< 2 weeks": input_dict['duration'] = "<2 weeks"
    elif input_dict.get('duration') == "> 3 weeks": input_dict['duration'] = ">3 weeks"

    # Fix Recurrence
    if input_dict.get('recurrence') == "First episode": input_dict['recurrence'] = "First"
    elif input_dict.get('recurrence') == "Recurrent (same site)": input_dict['recurrence'] = "Same site"
    elif input_dict.get('recurrence') == "Recurrent (different sites)": input_dict['recurrence'] = "Different site"

    # Fix Site
    if input_dict.get('site') == "Tongue (Lateral)": input_dict['site'] = "Lateral tongue"
    elif input_dict.get('site') == "Tongue (Ventral)": input_dict['site'] = "Ventral tongue"
    elif input_dict.get('site') == "Buccal Mucosa": input_dict['site'] = "Buccal mucosa"
    elif input_dict.get('site') == "Floor of Mouth": input_dict['site'] = "Floor of mouth"

    # Fix Shape
    if input_dict.get('shape') == "Round/Ovoid": input_dict['shape'] = "Round"

    # Fix Edge
    if input_dict.get('edge') == "Punched out": input_dict['edge'] = "Punched"
    # =========================================================

    # 2. Convert to DataFrame
    input_df = pd.DataFrame([input_dict])

    try:
        # 3. Get probability of Malignancy (Class 1)
        prob = float(model.predict_proba(input_df)[0][1])
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction error. Feature mismatch: {str(e)}")

    # 4. Process Results
    risk_score = round(prob * 100, 2)
    category, recommendation = classify_risk(risk_score)
    suggestions = generate_suggestions(category)
    
    # Calculate confidence
    confidence = round(abs(prob - 0.5) * 2 * 100, 1)

    return {
        "success": True,
        "clinicalRiskScore": risk_score,
        "clinicalRiskCategory": category,
        "biopsyRecommendation": recommendation,
        "confidence": f"{confidence}%",
        "riskExplanation": explanation,
        "clinicalSuggestions": suggestions
    }

# Entry point for local testing
if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='0.0.0.0', port=5000)
