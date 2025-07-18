import streamlit as st
import numpy as np
import json
import os
from dotenv import load_dotenv
from PIL import Image
from tensorflow.keras.models import load_model
from groq import Groq
from joblib import load  # For loading robust_scaler.pkl

# --- Page Configuration ---
st.set_page_config(page_title="Francis-99 Fault Detection", layout="centered")

# --- Load .env and model ---
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
model = load_model("cnn+LSTM_fault_model.h5")
scaler = load("robust_scaler.pkl")  # Load RobustScaler
client = Groq(api_key=GROQ_API_KEY)

# --- Custom CSS ---
st.markdown("""
    <style>
        body {
            background-color: #f0f2f6;
        }
        .main {
            background-color: #ffffff;
            padding: 2rem;
            border-radius: 20px;
            box-shadow: 0 0 20px rgba(0,0,0,0.1);
        }
        .title {
            color: #1f77b4;
            font-size: 2.5rem;
            font-weight: bold;
            margin-bottom: 10px;
        }
        .subheader {
            font-size: 1.2rem;
            color: #444444;
        }
        .result {
            background-color: #e6f4ea;
            padding: 1rem;
            border-radius: 10px;
            font-size: 1.1rem;
        }
        .btn {
            background-color: #1f77b4;
            color: white;
            padding: 10px 24px;
            border-radius: 10px;
            border: none;
            font-size: 1rem;
        }
        .btn:hover {
            background-color: #125b88;
        }
    </style>
""", unsafe_allow_html=True)

# --- Main UI Container ---
with st.container():
    st.markdown("<div class='main'>", unsafe_allow_html=True)
    st.markdown("<div class='title'>Francis-99 Fault Detection System</div>", unsafe_allow_html=True)
    st.markdown("<p class='subheader'>Provide sensor readings below to predict turbine blade fault stage.</p>", unsafe_allow_html=True)

    decimal_format = "%.6f"
    with st.form("input_form"):
        st.markdown("<h4>📡 Enter Sensor Readings:</h4>", unsafe_allow_html=True)
        PDT1 = st.number_input("PDT1", value=0.0, step=0.0001, format=decimal_format)
        PGV2 = st.number_input("PGV2", value=0.0, step=0.0001, format=decimal_format)
        PDT3 = st.number_input("PDT3", value=0.0, step=0.0001, format=decimal_format)
        ATB1 = st.number_input("ATB1", value=0.0, step=0.0001, format=decimal_format)
        ATB2 = st.number_input("ATB2", value=0.0, step=0.0001, format=decimal_format)

        submit = st.form_submit_button("🔍 Predict Fault Stage")

    if submit:
        try:
            # Step 1: Create a 1-row array and scale it
            raw_input = np.array([[PDT1, PGV2, PDT3, ATB1, ATB2]])
            scaled_input = scaler.transform(raw_input)

            # Step 2: Repeat it 30 times and reshape to (1, 30, 5)
            user_input = np.tile(scaled_input, (30, 1)).reshape(1, 30, 5).astype(np.float32)

            # Step 3: Predict using model
            prediction = model.predict(user_input)
            fault_stage = int(np.argmax(prediction))

            st.markdown(f"<div class='result'>✅ <b>Predicted Fault Stage:</b> {fault_stage}</div>", unsafe_allow_html=True)

            # Visual image slicing
            full_image_path = "D:/AI/project/main model/crackimage.jpg"
            if os.path.exists(full_image_path):
                image = Image.open(full_image_path)
                cols, rows = 5, 2
                img_width, img_height = image.size
                cell_width = img_width // cols
                cell_height = img_height // rows

                row = fault_stage // cols
                col = fault_stage % cols
                left = col * cell_width
                upper = row * cell_height
                right = left + cell_width
                lower = upper + cell_height

                stage_image = image.crop((left, upper, right, lower))
                st.subheader("📷 Visual Reference of Predicted Stage")
                st.image(stage_image, caption=f"Stage {fault_stage}", use_column_width=True)
            else:
                st.warning("⚠️ Reference image not found.")

            # --- Prompt LLM for Explanation ---
            prompt = [
                {
                    "role": "user",
                    "content": f"""
You are a turbine fault expert AI. A Francis-99 turbine is predicted to be in fault stage `{fault_stage}` based on sensor input.

Use the following reference to explain what this stage means:
- Stage 1 to 7: Crack growth stage
- Stage 8: Detached fragment
- Stage 9: Enlarged opening of the detached fragment

Crack length mapping (approximate per stage):
- Stage 1: 15 mm
- Stage 2: 30 mm
- Stage 3: 50 mm
- Stage 4: 72 mm
- Stage 5: 95 mm
- Stage 6: 115 mm
- Stage 7: 140 mm
- Stage 8: 170 mm

Your task:
- Identify the stage meaning and condition of the turbine.
- Mention approximate crack length if applicable.
- Suggest what the engineer should do next.

Return only a JSON in this format:
{{
  "stage": "Stage number and description",
  "crack_length": "Approximate crack length in mm (if applicable)",
  "description": "What this fault stage means",
  "recommendation": "What engineers should do next"
}}
"""
                }
            ]

            response = client.chat.completions.create(
                model="llama3-70b-8192",
                messages=prompt,
                max_tokens=300,
                temperature=0.4,
            )
            output = response.choices[0].message.content

            try:
                result = json.loads(output)
                st.subheader("🧠 Expert Interpretation")
                st.json(result)
            except json.JSONDecodeError:
                st.warning("⚠️ Couldn't parse expert response. Showing raw text:")
                st.code(output)

        except Exception as e:
            st.error(f"❌ Something went wrong: {e}")

    st.markdown("</div>", unsafe_allow_html=True)  # Close .main
