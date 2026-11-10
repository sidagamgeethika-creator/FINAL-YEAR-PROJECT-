import streamlit as st
import pandas as pd
import numpy as np
import requests

# ML ALGORITHM: Random Forest Classification
from sklearn.ensemble import RandomForestClassifier

# DATA SPLITTING
from sklearn.model_selection import train_test_split

# EVALUATION
from sklearn.metrics import accuracy_score


# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="MedTest Tool", layout="wide")


# ---------------- CUSTOM CSS ----------------
st.markdown("""
<style>

/* BACKGROUND */
.stApp {
    background: linear-gradient(rgba(5,10,25,0.88), rgba(5,10,25,0.96)),
                url("https://img.freepik.com/free-photo/futuristic-doctor-with-tablet_23-2151112193.jpg");
    background-size: cover;
}

/* REMOVE SIDEBAR */
[data-testid="stSidebar"] {display: none;}

/* PANEL */
.panel {
    max-width: 520px;
    margin-left: auto;
    margin-right: 50px;
    margin-top: 60px;
    background: rgba(255,255,255,0.08);
    padding: 40px;
    border-radius: 25px;
    backdrop-filter: blur(25px);
    box-shadow: 0 0 60px rgba(0,255,255,0.2);
}

/* TEXT */
.title {font-size:48px;font-weight:900;color:#00f5ff;}
.subtitle {color:white;margin-bottom:15px;}
.section {color:white;font-size:22px;font-weight:700;margin-top:20px;}
.data {color:white;font-size:18px;}
label, .stRadio label, div[role="radiogroup"] label {color:white !important;}

/* INPUT */
input, .stNumberInput input {
    background: rgba(0,0,0,0.7) !important;
    color: white !important;
    border-radius: 10px !important;
}

/* BUTTON */
.stButton>button {
    width: 100%;
    background: linear-gradient(90deg,#00f5ff,#00ff9c);
    font-weight:bold;
}

/* RESULT */
.good {color:#00ff9c;font-size:26px;font-weight:bold;}
.bad {color:#ff4b4b;font-size:26px;font-weight:bold;}

/* AI BOX */
.ai-box {
    margin-top:20px;
    padding:15px;
    background: rgba(0,0,0,0.6);
    border-radius:15px;
    color:#ffffff;
    font-size:16px;
    line-height:1.6;
}

</style>
""", unsafe_allow_html=True)


# ---------------- LOAD DATA ----------------
data = pd.read_csv("data.csv")
data["Condition"] = data["Condition"].map({"good": 1, "bad": 0})

X = data.drop("Condition", axis=1)
y = data["Condition"]

# ---------------- MODEL ----------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = RandomForestClassifier(n_estimators=100)
model.fit(X_train, y_train)

accuracy = accuracy_score(y_test, model.predict(X_test))


# ---------------- AI EXPLANATION FUNCTION ----------------
# ALGORITHM: Rule-Based AI + Expert System Logic
def generate_explanation(pred, heart, spo2, body, room):

    explanation = ""

    if pred == 1:
        explanation += "✅ Your health condition appears NORMAL.\n\n"
    else:
        explanation += "⚠️ Your health condition shows CRITICAL signs.\n\n"

    explanation += "📊 Analysis:\n"

    if heart < 60 or heart > 100:
        explanation += "- Abnormal heart rate detected.\n"
    else:
        explanation += "- Heart rate is within normal range.\n"

    if spo2 < 95:
        explanation += "- Low oxygen level detected.\n"
    else:
        explanation += "- Oxygen level is healthy.\n"

    if body < 36 or body > 37.5:
        explanation += "- Body temperature is abnormal.\n"
    else:
        explanation += "- Body temperature is normal.\n"

    explanation += "\n💡 Suggestion:\n"
    if pred == 0:
        explanation += "Please consult a doctor immediately and monitor vitals regularly."
    else:
        explanation += "Maintain healthy lifestyle and regular checkups."

    return explanation


# ---------------- UI ----------------
st.markdown('<div class="panel">', unsafe_allow_html=True)

st.markdown('<div class="title">🩺 MEDTEST TOOL</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Virtual Doctors made in ALIET @ Vijayawada</div>', unsafe_allow_html=True)

mode = st.radio("Select Mode", ["Auto (ThingSpeak)", "Manual"])

prediction = None

# ---------------- AUTO ----------------
if mode == "Auto (ThingSpeak)":
    st.markdown('<div class="section">📡 AUTO MODE - LIVE DATA</div>', unsafe_allow_html=True)

    url = "https://api.thingspeak.com/channels/3316525/feeds/last.json?api_key=ALSA6339QNRUMTV8"

    try:
        data_ts = requests.get(url).json()

        heart = float(data_ts["field1"])
        spo2 = float(data_ts["field2"])
        body = float(data_ts["field3"])
        room = float(data_ts["field4"])

        st.markdown(f'<div class="data">❤️ {heart}</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="data">🫁 {spo2}</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="data">🌡 {body}</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="data">🏠 {room}</div>', unsafe_allow_html=True)

        input_data = np.array([[heart, spo2, body, room]])
        prediction = model.predict(input_data)[0]

    except:
        st.error("IoT Data Error")

# ---------------- MANUAL ----------------
else:
    st.markdown('<div class="section">✍ MANUAL MODE</div>', unsafe_allow_html=True)

    heart = st.number_input("Heart Rate", value=80)
    spo2 = st.number_input("SpO2", value=95)
    body = st.number_input("Body Temp", value=36.5)
    room = st.number_input("Room Temp", value=25.0)

    if st.button("Predict"):
        input_data = np.array([[heart, spo2, body, room]])
        prediction = model.predict(input_data)[0]


# ---------------- RESULT + AI ----------------
if prediction is not None:

    if prediction == 1:
        st.markdown('<div class="good">🟢 NORMAL</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="bad">🔴 CRITICAL</div>', unsafe_allow_html=True)

    # AI EXPLANATION
    explanation = generate_explanation(prediction, heart, spo2, body, room)

    st.markdown('<div class="section">🤖 AI DOCTOR ANALYSIS</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="ai-box">{explanation}</div>', unsafe_allow_html=True)


st.markdown('</div>', unsafe_allow_html=True)