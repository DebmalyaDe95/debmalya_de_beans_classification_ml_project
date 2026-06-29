import streamlit as st
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVC
from PIL import Image
import os

# ----------------------------------------------------
# PAGE CONFIG
# ----------------------------------------------------
st.set_page_config(
    page_title="Bean Classification",
    page_icon="🌱",
    layout="wide"
)

# ----------------------------------------------------
# CUSTOM CSS
# ----------------------------------------------------
st.markdown("""
<style>

.main{
    background-color:#f5f7fa;
}

.title{
    text-align:center;
    font-size:42px;
    font-weight:bold;
    color:#2E8B57;
}

.subtitle{
    text-align:center;
    color:gray;
    margin-bottom:30px;
}

div.stButton > button{
    background:#2E8B57;
    color:white;
    font-size:18px;
    border-radius:12px;
    height:55px;
    width:100%;
}

div.stButton > button:hover{
    background:#1d6f43;
}

</style>
""", unsafe_allow_html=True)

st.markdown("<div class='title'>🌱 Dry Bean Classification System</div>",
            unsafe_allow_html=True)

st.markdown("<div class='subtitle'>Predict bean variety using Machine Learning</div>",
            unsafe_allow_html=True)

# ----------------------------------------------------
# LOAD DATA
# ----------------------------------------------------
df = pd.read_csv("Worksheet in Beans Multiclass Classification (1).csv")

le = LabelEncoder()
df["Class"] = le.fit_transform(df["Class"])

X = df.drop("Class", axis=1)
y = df["Class"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

model = SVC(
    probability=True,
    class_weight="balanced",
    random_state=42
)

model.fit(X_scaled, y)

# ----------------------------------------------------
# SIDEBAR
# ----------------------------------------------------
st.sidebar.title("About")

st.sidebar.info("""
This application predicts the **Dry Bean Variety**
using an SVM classifier.

Fill in all bean measurements and click
**Predict Bean Class**.
""")

# ----------------------------------------------------
# INPUTS
# ----------------------------------------------------
st.header("🫘 Enter Bean Measurements")

col1, col2 = st.columns(2)

inputs = []

for i, col in enumerate(X.columns):

    if i % 2 == 0:
        with col1:
            value = st.number_input(
                col,
                value=float(df[col].mean()),
                format="%.4f"
            )
    else:
        with col2:
            value = st.number_input(
                col,
                value=float(df[col].mean()),
                format="%.4f"
            )

    inputs.append(value)

# ----------------------------------------------------
# PREDICT
# ----------------------------------------------------
if st.button("🔍 Predict Bean Class"):

    sample = np.array([inputs])
    sample_scaled = scaler.transform(sample)

    prediction = model.predict(sample_scaled)[0]
    probabilities = model.predict_proba(sample_scaled)[0]

    bean = le.inverse_transform([prediction])[0]

    st.success(f"✅ Predicted Bean Type : **{bean}**")

    # ---------------- Image ----------------
    image_path = f"images/{bean}.jpg"

    if os.path.exists(image_path):
        image = Image.open(image_path)

        st.image(
            image,
            width=350,
            caption=f"{bean} Bean"
        )
    else:
        st.warning("Image not found.")

    # ---------------- Confidence ----------------

    st.subheader("Prediction Confidence")

    prob_df = pd.DataFrame({
        "Bean Type": le.classes_,
        "Probability": probabilities
    })

    st.bar_chart(
        prob_df.set_index("Bean Type")
    )

    st.dataframe(
        prob_df.style.format({
            "Probability":"{:.2%}"
        }),
        use_container_width=True
    )

# ----------------------------------------------------
# FOOTER
# ----------------------------------------------------
st.markdown("---")

st.markdown(
"""
<center>
Developed using ❤️ Streamlit | Scikit-Learn | SVM
</center>
""",
unsafe_allow_html=True
)