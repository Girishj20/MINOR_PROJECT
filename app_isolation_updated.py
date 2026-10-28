from flask import Flask, render_template, send_file, request, session, jsonify
import os
import joblib
import pandas as pd
from datetime import datetime

from database import init_database, save_animal_record

from reportlab.lib.pagesizes import A4
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle
)
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.enums import TA_CENTER


# ============================================================
# CREATE FLASK APPLICATION
# ============================================================

app = Flask(__name__)

init_database()

# Secret key for session storage
app.secret_key = "ai_livestock_health_secret_key_2026"


# ============================================================
# LOAD FINAL COW HIERARCHICAL MODEL
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "cow_hierarchical_model.joblib"
)

cow_model_package = None

try:
    cow_model_package = joblib.load(MODEL_PATH)

    print("==================================================")
    print("Final Cow Hierarchical Model Loaded Successfully")
    print("==================================================")

    print("Stage 1 model:", type(
        cow_model_package["stage1_model"]
    ).__name__)

    print("Stage 2 model:", type(
        cow_model_package["stage2_model"]
    ).__name__)

    print(
        "Stage 2 classes:",
        len(cow_model_package["stage2_classes"])
    )

except Exception as e:

    print("WARNING: Cow model could not be loaded.")
    print("Reason:", e)


# ============================================================
# LOAD FINAL BUFFALO + SHEEP HIERARCHICAL MODELS
# ============================================================

def load_model_package(filename, animal_name):
    model_path = os.path.join(BASE_DIR, "models", filename)
    try:
        package = joblib.load(model_path)
        print("==================================================")
        print(f"Final {animal_name} Hierarchical Model Loaded Successfully")
        print("==================================================")
        print("Stage 1 model:", type(package["stage1_model"]).__name__)
        print("Stage 2 model:", type(package["stage2_model"]).__name__)
        if "stage2_classes" in package:
            print("Stage 2 classes:", len(package["stage2_classes"]))
        elif "disease_classes" in package:
            print("Disease classes:", len(package["disease_classes"]))
        return package
    except Exception as e:
        print(f"WARNING: {animal_name} model could not be loaded.")
        print("Reason:", e)
        return None

buffalo_model_package = load_model_package("buffalo_hierarchical_model.joblib", "Buffalo")
sheep_model_package = load_model_package("sheep_hierarchical_model.joblib", "Sheep")


print("==================================================")
print("MODEL LOAD STATUS")
print("==================================================")
print("Cow    :", "LOADED" if cow_model_package is not None else "NOT LOADED")
print("Buffalo:", "LOADED" if buffalo_model_package is not None else "NOT LOADED")
print("Sheep  :", "LOADED" if sheep_model_package is not None else "NOT LOADED")
print("==================================================")


# ============================================================
# MODEL INFORMATION
# ============================================================

MODEL_VERSION = "Cow Hierarchical Model v1.0"

STAGE1_TEST_ACCURACY = 99.79
STAGE1_BALANCED_ACCURACY = 99.43
STAGE1_MACRO_F1 = 99.32

STAGE2_TEST_ACCURACY = 67.09
STAGE2_BALANCED_ACCURACY = 67.09
STAGE2_MACRO_F1 = 66.38

END_TO_END_ACCURACY = 69.65
END_TO_END_BALANCED_ACCURACY = 69.65
END_TO_END_MACRO_F1 = 68.98


MODEL_NOTE = (
    "This system provides a machine-learning prediction and is not "
    "a veterinary diagnosis. Model results were evaluated on a "
    "synthetic development dataset and should not be interpreted "
    "as clinical diagnostic accuracy."
)


# ============================================================
# ISOLATION RECOMMENDATION
# ============================================================

ISOLATION_RULES = {
    "Mastitis": ("YES", "Precautionary separation is recommended until veterinary assessment."),
    "Foot_and_Mouth": ("YES", "Isolation is recommended as a precaution because of the contagious nature of the disease."),
    "Bovine_Tuberculosis": ("YES", "Separate the suspected animal as a precaution while veterinary assessment/testing is arranged."),
    "Brucellosis": ("YES", "Separate the suspected animal as a precaution while veterinary assessment/testing is arranged."),
    "Lumpy_Skin_Disease": ("YES", "Isolation is recommended as a precaution while the animal is assessed."),
    "Bovine_Respiratory_Disease": ("YES", "Separation is recommended as a precaution to reduce potential transmission."),
    "Haemorrhagic_Septicemia": ("YES", "Isolation is recommended as a precaution while urgent veterinary assessment is arranged."),
    "Johnes_Disease": ("YES", "Separation is recommended as a precaution while veterinary assessment is arranged."),
    "Leptospirosis": ("YES", "Isolation is recommended as a precaution while veterinary assessment is arranged."),
    "Ketosis_Subclinical": ("NO", "Routine isolation is not indicated by the prediction alone."),
    "Milk_Fever": ("NO", "Routine isolation is not indicated by the prediction alone; veterinary attention is more important.")
}


def get_isolation_recommendation(health_status, disease=None):
    if str(health_status).lower() == "healthy":
        return {
            "required": "NO",
            "reason": "No disease was detected by the current prediction model."
        }

    return {
        "required": ISOLATION_RULES.get(
            str(disease),
            ("YES", "Precautionary isolation is recommended until veterinary assessment.")
        )[0],
        "reason": ISOLATION_RULES.get(
            str(disease),
            ("YES", "Precautionary isolation is recommended until veterinary assessment.")
        )[1]
    }


# ============================================================
# COW PREDICTION FUNCTION
# ============================================================

def predict_cow_disease(
    animal_type,
    breed,
    age_months,
    weight_kg,
    feed_quantity_kg,
    water_intake_l,
    walking_distance_km,
    grazing_duration_hrs,
    rumination_time_hrs,
    resting_hours,
    body_temperature_c,
    heart_rate_bpm,
    respiratory_rate,
    milk_yield_l,
    body_condition_score
):

    if cow_model_package is None:
        raise RuntimeError(
            "Cow hierarchical model is not loaded."
        )

    # --------------------------------------------------------
    # Get models and feature information
    # --------------------------------------------------------

    stage1_model = cow_model_package["stage1_model"]
    stage2_model = cow_model_package["stage2_model"]

    stage1_features = cow_model_package["stage1_features"]
    stage2_features = cow_model_package["stage2_features"]

    stage1_classes = cow_model_package["stage1_classes"]
    stage2_classes = cow_model_package["stage2_classes"]

    # --------------------------------------------------------
    # Create input dataframe
    # --------------------------------------------------------

    input_data = {
        "Animal_Type": animal_type,
        "Breed": breed,
        "Age_Months": float(age_months),
        "Weight_kg": float(weight_kg),
        "Feed_Quantity_kg": float(feed_quantity_kg),
        "Water_Intake_L": float(water_intake_l),
        "Walking_Distance_km": float(walking_distance_km),
        "Grazing_Duration_hrs": float(grazing_duration_hrs),
        "Rumination_Time_hrs": float(rumination_time_hrs),
        "Resting_Hours": float(resting_hours),
        "Body_Temperature_C": float(body_temperature_c),
        "Heart_Rate_bpm": float(heart_rate_bpm),
        "Respiratory_Rate": float(respiratory_rate),
        "Milk_Yield_L": float(milk_yield_l),
        "Body_Condition_Score": float(body_condition_score)
    }

    input_df = pd.DataFrame([input_data])

    # ========================================================
    # STAGE 1
    # Healthy vs Disease
    # ========================================================

    stage1_input = input_df[stage1_features]

    stage1_prediction = stage1_model.predict(
        stage1_input
    )[0]

    health_status = str(stage1_prediction)

    # --------------------------------------------------------
    # Stage 1 probability
    # --------------------------------------------------------

    stage1_probability = None

    try:

        if hasattr(stage1_model, "predict_proba"):

            probabilities = stage1_model.predict_proba(
                stage1_input
            )[0]

            classes = (
                stage1_model.classes_
                if hasattr(stage1_model, "classes_")
                else stage1_classes
            )

            max_index = probabilities.argmax()

            stage1_probability = round(
                float(probabilities[max_index]) * 100,
                2
            )

    except Exception as e:

        print(
            "Stage 1 probability unavailable:",
            e
        )

    # ========================================================
    # HEALTHY CASE
    # ========================================================

    if health_status.lower() == "healthy":

        result = {

            "Animal": animal_type,

            "Health_Status": "Healthy",

            "Predicted_Disease": "None",

            "Stage_1_Prediction": health_status,

            "Stage_2_Prediction": None,

            "Stage_1_Probability": stage1_probability,

            "Model_Probability": stage1_probability,

            "Isolation_Recommendation": "NO",
            "Isolation_Reason": "No disease was detected by the current prediction model.",

            "Model_Version": MODEL_VERSION,

            "Model_Note": MODEL_NOTE,

            "Timestamp": datetime.now().strftime(
                "%d-%m-%Y %H:%M:%S"
            ),

            "Input_Parameters": input_data
        }

        return result

    # ========================================================
    # STAGE 2
    # Specific Disease Classification
    # ========================================================

    stage2_input = input_df[stage2_features]

    stage2_prediction = stage2_model.predict(
        stage2_input
    )[0]

    predicted_disease = str(stage2_prediction)

    # --------------------------------------------------------
    # Stage 2 probability
    # --------------------------------------------------------

    disease_probability = None

    try:

        if hasattr(stage2_model, "predict_proba"):

            probabilities = stage2_model.predict_proba(
                stage2_input
            )[0]

            classes = (
                stage2_model.classes_
                if hasattr(stage2_model, "classes_")
                else stage2_classes
            )

            max_index = probabilities.argmax()

            disease_probability = round(
                float(probabilities[max_index]) * 100,
                2
            )

    except Exception as e:

        print(
            "Stage 2 probability unavailable:",
            e
        )

    # ========================================================
    # FINAL DISEASE RESULT
    # ========================================================

    result = {

        "Animal": animal_type,

        "Health_Status": "Disease",

        "Predicted_Disease": predicted_disease,

        "Stage_1_Prediction": health_status,

        "Stage_2_Prediction": predicted_disease,

        "Stage_1_Probability": stage1_probability,

        "Model_Probability": disease_probability,

        "Isolation_Recommendation": get_isolation_recommendation(
            "Disease",
            predicted_disease
        )["required"],

        "Isolation_Reason": get_isolation_recommendation(
            "Disease",
            predicted_disease
        )["reason"],

        "Model_Version": MODEL_VERSION,

        "Model_Note": MODEL_NOTE,

        "Timestamp": datetime.now().strftime(
            "%d-%m-%Y %H:%M:%S"
        ),

        "Input_Parameters": input_data
    }

    return result


# ============================================================
# GENERIC BUFFALO / SHEEP PREDICTION
# ============================================================

def predict_animal_disease(
    model_package,
    animal_type,
    breed,
    age_months,
    weight_kg,
    feed_quantity_kg,
    water_intake_l,
    walking_distance_km,
    grazing_duration_hrs,
    rumination_time_hrs,
    resting_hours,
    body_temperature_c,
    heart_rate_bpm,
    respiratory_rate,
    milk_yield_l,
    body_condition_score
):
    if model_package is None:
        raise RuntimeError(
            f"{animal_type} hierarchical model is not loaded."
        )

    input_data = {
        "Animal_Type": animal_type,
        "Breed": breed,
        "Age_Months": float(age_months),
        "Weight_kg": float(weight_kg),
        "Feed_Quantity_kg": float(feed_quantity_kg),
        "Water_Intake_L": float(water_intake_l),
        "Walking_Distance_km": float(walking_distance_km),
        "Grazing_Duration_hrs": float(grazing_duration_hrs),
        "Rumination_Time_hrs": float(rumination_time_hrs),
        "Resting_Hours": float(resting_hours),
        "Body_Temperature_C": float(body_temperature_c),
        "Heart_Rate_bpm": float(heart_rate_bpm),
        "Respiratory_Rate": float(respiratory_rate),
        "Milk_Yield_L": float(milk_yield_l),
        "Body_Condition_Score": float(body_condition_score)
    }

    input_df = pd.DataFrame([input_data])

    stage1_model = model_package["stage1_model"]
    stage2_model = model_package["stage2_model"]

    # Cow package stores its feature lists.
    # Buffalo and Sheep packages do not store feature_columns,
    # so their final 12 trained features are used here.
    if "stage1_features" in model_package:

        stage1_features = list(
            model_package["stage1_features"]
        )

        stage2_features = list(
            model_package.get(
                "stage2_features",
                stage1_features
            )
        )

    elif "feature_columns" in model_package:

        stage1_features = list(
            model_package["feature_columns"]
        )

        stage2_features = list(
            model_package["feature_columns"]
        )

    else:

        stage1_features = [
            "Breed",
            "Age_Months",
            "Feed_Quantity_kg",
            "Walking_Distance_km",
            "Grazing_Duration_hrs",
            "Rumination_Time_hrs",
            "Resting_Hours",
            "Body_Temperature_C",
            "Heart_Rate_bpm",
            "Respiratory_Rate",
            "Milk_Yield_L",
            "Body_Condition_Score"
        ]

        stage2_features = stage1_features.copy()

    # ========================================================
    # STAGE 1
    # Healthy vs Disease
    # ========================================================

    stage1_input = input_df[stage1_features]

    stage1_prediction = str(
        stage1_model.predict(stage1_input)[0]
    )

    stage1_probability = None

    try:

        if hasattr(stage1_model, "predict_proba"):

            probabilities = stage1_model.predict_proba(
                stage1_input
            )[0]

            stage1_probability = round(
                float(probabilities.max()) * 100,
                2
            )

    except Exception as e:

        print(
            f"{animal_type} Stage 1 probability unavailable:",
            e
        )

    # ========================================================
    # HEALTHY CASE
    # ========================================================

    if stage1_prediction.lower() == "healthy":

        return {
            "Animal": animal_type,
            "Health_Status": "Healthy",
            "Predicted_Disease": "None",
            "Stage_1_Prediction": stage1_prediction,
            "Stage_2_Prediction": None,
            "Stage_1_Probability": stage1_probability,
            "Model_Probability": stage1_probability,
            "Isolation_Recommendation": "NO",
            "Isolation_Reason": (
                "No disease was detected by the current "
                "prediction model."
            ),
            "Model_Version": (
                f"{animal_type} Hierarchical Model v1.0"
            ),
            "Model_Note": MODEL_NOTE,
            "Timestamp": datetime.now().strftime(
                "%d-%m-%Y %H:%M:%S"
            ),
            "Input_Parameters": input_data
        }

    # ========================================================
    # STAGE 2
    # Specific Disease Classification
    # ========================================================

    stage2_input = input_df[stage2_features]

    stage2_prediction = str(
        stage2_model.predict(stage2_input)[0]
    )

    disease_probability = None

    try:

        if hasattr(stage2_model, "predict_proba"):

            probabilities = stage2_model.predict_proba(
                stage2_input
            )[0]

            disease_probability = round(
                float(probabilities.max()) * 100,
                2
            )

    except Exception as e:

        print(
            f"{animal_type} Stage 2 probability unavailable:",
            e
        )

    isolation = get_isolation_recommendation(
        "Disease",
        stage2_prediction
    )

    return {
        "Animal": animal_type,
        "Health_Status": "Disease",
        "Predicted_Disease": stage2_prediction,
        "Stage_1_Prediction": stage1_prediction,
        "Stage_2_Prediction": stage2_prediction,
        "Stage_1_Probability": stage1_probability,
        "Model_Probability": disease_probability,
        "Isolation_Recommendation": isolation["required"],
        "Isolation_Reason": isolation["reason"],
        "Model_Version": (
            f"{animal_type} Hierarchical Model v1.0"
        ),
        "Model_Note": MODEL_NOTE,
        "Timestamp": datetime.now().strftime(
            "%d-%m-%Y %H:%M:%S"
        ),
        "Input_Parameters": input_data
    }


def save_result_to_database(result):
    """
    Save prediction data using the existing database.py format.

    The prediction result keeps its original display/model keys such as
    "Breed", "Age_Months", etc. The database.py currently reads lowercase
    keys, so only the database copy is normalized here.
    """
    params = result.get("Input_Parameters", {})

    db_result = dict(result)
    db_result["Input_Parameters"] = {
        "breed": params.get("breed", params.get("Breed")),
        "age_months": params.get("age_months", params.get("Age_Months")),
        "weight_kg": params.get("weight_kg", params.get("Weight_kg")),
        "feed_quantity_kg": params.get("feed_quantity_kg", params.get("Feed_Quantity_kg")),
        "water_intake_l": params.get("water_intake_l", params.get("Water_Intake_L")),
        "walking_distance_km": params.get("walking_distance_km", params.get("Walking_Distance_km")),
        "grazing_duration_hrs": params.get("grazing_duration_hrs", params.get("Grazing_Duration_hrs")),
        "rumination_time_hrs": params.get("rumination_time_hrs", params.get("Rumination_Time_hrs")),
        "resting_hours": params.get("resting_hours", params.get("Resting_Hours")),
        "body_temperature_c": params.get("body_temperature_c", params.get("Body_Temperature_C")),
        "heart_rate_bpm": params.get("heart_rate_bpm", params.get("Heart_Rate_bpm")),
        "respiratory_rate": params.get("respiratory_rate", params.get("Respiratory_Rate")),
        "milk_yield_l": params.get("milk_yield_l", params.get("Milk_Yield_L")),
        "body_condition_score": params.get("body_condition_score", params.get("Body_Condition_Score")),
    }

    save_animal_record(db_result)


def handle_prediction_result(result):
    session["latest_prediction"] = result
    history = session.get("prediction_history", [])
    history.insert(0, result)
    session["prediction_history"] = history[:20]
    session.modified = True
    save_result_to_database(result)
    return jsonify({"success": True, "prediction": result})

def build_prediction_from_json(data, model_package, animal_type):
    required_fields = ["breed", "age_months", "weight_kg", "feed_quantity_kg", "water_intake_l", "walking_distance_km", "grazing_duration_hrs", "rumination_time_hrs", "resting_hours", "body_temperature_c", "heart_rate_bpm", "respiratory_rate", "milk_yield_l", "body_condition_score"]
    missing_fields = [f for f in required_fields if data.get(f) in [None, ""]]
    if missing_fields:
        return None, "Missing required fields: " + ", ".join(missing_fields)
    return predict_animal_disease(model_package, data.get("animal_type", animal_type), data.get("breed"), data.get("age_months"), data.get("weight_kg"), data.get("feed_quantity_kg"), data.get("water_intake_l"), data.get("walking_distance_km"), data.get("grazing_duration_hrs"), data.get("rumination_time_hrs"), data.get("resting_hours"), data.get("body_temperature_c"), data.get("heart_rate_bpm"), data.get("respiratory_rate"), data.get("milk_yield_l"), data.get("body_condition_score")), None


# ============================================================
# HOME
# ============================================================

@app.route("/")
def home():

    return render_template(
        "home.html"
    )


# ============================================================
# LOGIN
# ============================================================

@app.route("/login")
def login():

    return render_template(
        "login.html"
    )


# ============================================================
# REGISTER
# ============================================================

@app.route("/register")
def register():

    return render_template(
        "register.html"
    )


# ============================================================
# SELECT ANIMAL
# ============================================================

@app.route("/select-animal")
def select_animal():

    return render_template(
        "select_animal.html"
    )


# ============================================================
# COW PREDICTION FORM
# ============================================================

@app.route("/predict/cow")
def cow_prediction():

    return render_template(
        "prediction_form.html",
        animal="Cow"
    )


# ============================================================
# BUFFALO PREDICTION FORM
# ============================================================

@app.route("/predict/buffalo")
def buffalo_prediction():

    return render_template(
        "prediction_form.html",
        animal="Buffalo"
    )


# ============================================================
# GOAT PREDICTION FORM
# ============================================================

@app.route("/predict/goat")
def goat_prediction():

    return render_template(
        "prediction_form.html",
        animal="Goat"
    )


# ============================================================
# SHEEP PREDICTION FORM
# ============================================================

@app.route("/predict/sheep")
def sheep_prediction():

    return render_template(
        "prediction_form.html",
        animal="Sheep"
    )


# ============================================================
# COW PREDICTION API
# ============================================================

@app.route(
    "/api/predict/cow",
    methods=["POST"]
)
def cow_prediction_api():

    try:

        data = request.get_json()

        if not data:

            return jsonify({
                "success": False,
                "error": "No input data received."
            }), 400

        # ----------------------------------------------------
        # Validate required inputs
        # ----------------------------------------------------

        required_fields = [

            "breed",
            "age_months",
            "weight_kg",
            "feed_quantity_kg",
            "water_intake_l",
            "walking_distance_km",
            "grazing_duration_hrs",
            "rumination_time_hrs",
            "resting_hours",
            "body_temperature_c",
            "heart_rate_bpm",
            "respiratory_rate",
            "milk_yield_l",
            "body_condition_score"
        ]

        missing_fields = [

            field
            for field in required_fields
            if data.get(field) in [None, ""]
        ]

        if missing_fields:

            return jsonify({

                "success": False,

                "error": (
                    "Missing required fields: "
                    + ", ".join(missing_fields)
                )

            }), 400

        # ----------------------------------------------------
        # Prediction
        # ----------------------------------------------------

        result = predict_cow_disease(

            animal_type=data.get(
                "animal_type",
                "Cow"
            ),

            breed=data.get("breed"),

            age_months=data.get(
                "age_months"
            ),

            weight_kg=data.get(
                "weight_kg"
            ),

            feed_quantity_kg=data.get(
                "feed_quantity_kg"
            ),

            water_intake_l=data.get(
                "water_intake_l"
            ),

            walking_distance_km=data.get(
                "walking_distance_km"
            ),

            grazing_duration_hrs=data.get(
                "grazing_duration_hrs"
            ),

            rumination_time_hrs=data.get(
                "rumination_time_hrs"
            ),

            resting_hours=data.get(
                "resting_hours"
            ),

            body_temperature_c=data.get(
                "body_temperature_c"
            ),

            heart_rate_bpm=data.get(
                "heart_rate_bpm"
            ),

            respiratory_rate=data.get(
                "respiratory_rate"
            ),

            milk_yield_l=data.get(
                "milk_yield_l"
            ),

            body_condition_score=data.get(
                "body_condition_score"
            )
        )

        # ----------------------------------------------------
        # Store latest prediction in session + database
        # ----------------------------------------------------

        session["latest_prediction"] = result
        save_result_to_database(result)

        # ----------------------------------------------------
        # Store prediction in history
        # ----------------------------------------------------

        history = session.get("prediction_history", [])

        history.insert(0, result)

        # Keep only the latest 20 predictions
        session["prediction_history"] = history[:20]

        session.modified = True

        return jsonify({

            "success": True,

            "prediction": result

        })

    except ValueError as e:

        return jsonify({

            "success": False,

            "error": (
                "Invalid input value. "
                "Please enter valid numerical values."
            )

        }), 400

    except Exception as e:

        print(
            "Cow prediction error:",
            e
        )

        return jsonify({

            "success": False,

            "error": str(e)

        }), 500


# ============================================================
# BUFFALO PREDICTION API
# ============================================================

@app.route("/api/predict/buffalo", methods=["POST"])
def buffalo_prediction_api():
    try:
        data = request.get_json()
        if not data:
            return jsonify({"success": False, "error": "No input data received."}), 400
        result, error = build_prediction_from_json(data, buffalo_model_package, "Buffalo")
        if error:
            return jsonify({"success": False, "error": error}), 400
        return handle_prediction_result(result)
    except ValueError as e:
     print("BUFFALO VALUE ERROR:", repr(e))
    return jsonify({
        "success": False,
        "error": str(e)
    }), 400

# ============================================================
# SHEEP PREDICTION API
# ============================================================

@app.route("/api/predict/sheep", methods=["POST"])
def sheep_prediction_api():
    try:
        data = request.get_json()
        if not data:
            return jsonify({"success": False, "error": "No input data received."}), 400
        result, error = build_prediction_from_json(data, sheep_model_package, "Sheep")
        if error:
            return jsonify({"success": False, "error": error}), 400
        return handle_prediction_result(result)
    except ValueError as e:
     print("SHEEP VALUE ERROR:", repr(e))
    return jsonify({
        "success": False,
        "error": str(e)
    }), 400


# ============================================================
# RESULT PAGE
# ============================================================

@app.route("/result")
def result():

    # Get actual latest prediction
    prediction = session.get(
        "latest_prediction"
    )

    # No prediction available
    if not prediction:

        return render_template(
            "result.html",
            prediction=None
        )

    return render_template(
        "result.html",
        prediction=prediction
    )


# ============================================================
# HISTORY
# ============================================================

@app.route("/history")
def history():

    history_data = session.get(
        "prediction_history",
        []
    )

    return render_template(
        "history.html",
        history=history_data
    )


# ============================================================
# ABOUT
# ============================================================

@app.route("/about")
def about():

    return render_template(
        "about.html"
    )


# ============================================================
# CONTACT
# ============================================================

@app.route("/contact")
def contact():

    return render_template(
        "contact.html"
    )


# ============================================================
# GENERATE DYNAMIC PDF REPORT
# ============================================================

@app.route("/generate-report")
def generate_report():

    prediction = session.get("latest_prediction")

    if not prediction:
        return (
            "No prediction available. Please perform a livestock health prediction first.",
            400
        )

    file_path = os.path.join(
        BASE_DIR,
        "livestock_health_report.pdf"
    )

    styles = getSampleStyleSheet()
    title_style = styles["Title"]
    title_style.alignment = TA_CENTER
    heading_style = styles["Heading2"]
    normal_style = styles["Normal"]

    document = SimpleDocTemplate(
        file_path,
        pagesize=A4,
        rightMargin=45,
        leftMargin=45,
        topMargin=45,
        bottomMargin=45
    )

    content = []

    content.append(
        Paragraph("Livestock Health Prediction Report", title_style)
    )
    content.append(Spacer(1, 20))

    content.append(Paragraph("Prediction Result", heading_style))
    content.append(Spacer(1, 8))

    disease = str(
        prediction.get("Predicted_Disease", "None")
    ).replace("_", " ")

    probability = prediction.get("Model_Probability")
    probability_text = (
        f"{probability}%"
        if probability is not None
        else "Not available"
    )

    summary_data = [
        ["Animal", str(prediction.get("Animal", "Cow"))],
        ["Health Status", str(prediction.get("Health_Status", "N/A"))],
        ["Predicted Disease", disease],
        ["Prediction Probability", probability_text],
        ["Isolation Recommendation",
         str(prediction.get("Isolation_Recommendation", "Not available"))],
        ["Prediction Time", str(prediction.get("Timestamp", "N/A"))]
    ]

    summary_table = Table(
        summary_data,
        colWidths=[190, 270]
    )

    summary_table.setStyle(
        TableStyle([
            ("BACKGROUND", (0, 0), (0, -1), colors.lightgrey),
            ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("PADDING", (0, 0), (-1, -1), 7)
        ])
    )

    content.append(summary_table)
    content.append(Spacer(1, 12))

    content.append(
        Paragraph(
            "<b>Isolation Note:</b> " +
            str(prediction.get("Isolation_Reason", "")),
            normal_style
        )
    )

    content.append(Spacer(1, 20))

    content.append(Paragraph("Animal Parameters", heading_style))
    content.append(Spacer(1, 8))

    input_parameters = prediction.get("Input_Parameters", {})
    parameter_rows = [["Parameter", "Value"]]

    for key, value in input_parameters.items():
        parameter_rows.append([
            key.replace("_", " "),
            str(value)
        ])

    parameter_table = Table(
        parameter_rows,
        colWidths=[260, 200]
    )

    parameter_table.setStyle(
        TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
            ("PADDING", (0, 0), (-1, -1), 6),
            ("VALIGN", (0, 0), (-1, -1), "TOP")
        ])
    )

    content.append(parameter_table)
    content.append(Spacer(1, 20))

    content.append(
        Paragraph(
            "<b>Important:</b> This report is an AI-based prediction "
            "from the supplied animal parameters. It is not a confirmed "
            "veterinary diagnosis. The isolation recommendation is "
            "precautionary and should be confirmed with a veterinarian.",
            normal_style
        )
    )

    content.append(Spacer(1, 15))

    content.append(
        Paragraph(
            "Generated by AI Livestock Health System.",
            normal_style
        )
    )

    document.build(content)

    return send_file(
        file_path,
        as_attachment=True,
        download_name="livestock_health_report.pdf"
    )

# ============================================================
# RUN APPLICATION
# ============================================================

if __name__ == "__main__":

    app.run(
        debug=True
    )
    