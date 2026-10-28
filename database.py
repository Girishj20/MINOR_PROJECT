import sqlite3
from datetime import datetime

DATABASE_NAME = "livestock_data.db"


def get_connection():
    conn = sqlite3.connect(DATABASE_NAME)
    conn.row_factory = sqlite3.Row
    return conn


def init_database():

    conn = get_connection()

    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS animal_records (

            id INTEGER PRIMARY KEY AUTOINCREMENT,

            animal_type TEXT NOT NULL,

            breed TEXT,

            age_months REAL,
            weight_kg REAL,
            feed_quantity_kg REAL,
            water_intake_l REAL,
            walking_distance_km REAL,
            grazing_duration_hrs REAL,
            rumination_time_hrs REAL,
            resting_hours REAL,
            body_temperature_c REAL,
            heart_rate_bpm REAL,
            respiratory_rate REAL,
            milk_yield_l REAL,
            body_condition_score REAL,

            health_status TEXT,
            predicted_disease TEXT,
            stage1_prediction TEXT,
            stage2_prediction TEXT,

            stage1_probability REAL,
            model_probability REAL,

            model_version TEXT,

            created_at TEXT NOT NULL
        )
    """)

    conn.commit()
    conn.close()


def save_animal_record(record):

    conn = get_connection()

    cursor = conn.cursor()

    cursor.execute("""
        INSERT INTO animal_records (

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
            body_condition_score,

            health_status,
            predicted_disease,
            stage1_prediction,
            stage2_prediction,

            stage1_probability,
            model_probability,

            model_version,
            created_at
        )

        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                ?, ?, ?, ?, ?, ?, ?, ?)
    """, (

        record.get("Animal"),

        record.get("Input_Parameters", {}).get("breed"),

        record.get("Input_Parameters", {}).get("age_months"),
        record.get("Input_Parameters", {}).get("weight_kg"),
        record.get("Input_Parameters", {}).get("feed_quantity_kg"),
        record.get("Input_Parameters", {}).get("water_intake_l"),
        record.get("Input_Parameters", {}).get("walking_distance_km"),
        record.get("Input_Parameters", {}).get("grazing_duration_hrs"),
        record.get("Input_Parameters", {}).get("rumination_time_hrs"),
        record.get("Input_Parameters", {}).get("resting_hours"),
        record.get("Input_Parameters", {}).get("body_temperature_c"),
        record.get("Input_Parameters", {}).get("heart_rate_bpm"),
        record.get("Input_Parameters", {}).get("respiratory_rate"),
        record.get("Input_Parameters", {}).get("milk_yield_l"),
        record.get("Input_Parameters", {}).get("body_condition_score"),

        record.get("Health_Status"),
        record.get("Predicted_Disease"),
        record.get("Stage_1_Prediction"),
        record.get("Stage_2_Prediction"),

        record.get("Stage_1_Probability"),
        record.get("Model_Probability"),

        record.get("Model_Version"),

        datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    ))

    conn.commit()
    conn.close()