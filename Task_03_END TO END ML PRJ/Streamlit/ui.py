import streamlit as st
import pandas as pd
import numpy as np
import pickle

st.set_page_config(
    page_title="Student Dropout Predictor",
    page_icon="🎓",
    layout="wide"
)

@st.cache_resource
def load_model():
    with open("best_model11.pkl", "rb") as f:
        model = pickle.load(f)
    return model

try:
    model = load_model()
except FileNotFoundError:
    st.error(
        "⚠️ 'best_model11.pkl' nahi mili. Is app.py ke sath usi folder mein "
        "best_model11.pkl file (jo aapne notebook se download ki thi) rakh dein."
    )
    st.stop()

CLASS_LABELS = list(model.classes_) if hasattr(model, "classes_") else None

FEATURE_ORDER = [
    "Marital status", "Application mode", "Application order", "Course",
    "Daytime/evening attendance\t", "Previous qualification",
    "Previous qualification (grade)", "Nacionality",
    "Mother's qualification", "Father's qualification",
    "Mother's occupation", "Father's occupation", "Admission grade",
    "Displaced", "Educational special needs", "Debtor",
    "Tuition fees up to date", "Gender", "Scholarship holder",
    "Age at enrollment", "International",
    "Curricular units 1st sem (credited)",
    "Curricular units 1st sem (enrolled)",
    "Curricular units 1st sem (evaluations)",
    "Curricular units 1st sem (approved)",
    "Curricular units 1st sem (grade)",
    "Curricular units 1st sem (without evaluations)",
    "Curricular units 2nd sem (credited)",
    "Curricular units 2nd sem (evaluations)",
    "Curricular units 2nd sem (approved)",
    "Curricular units 2nd sem (grade)",
    "Curricular units 2nd sem (without evaluations)",
    "Unemployment rate", "Inflation rate", "GDP",
    "Age_Category_Adult", "Age_Category_Mature",
]

YES_NO = {"No": 0, "Yes": 1}

st.title("🎓 Student Dropout Prediction")
st.caption(
    "Random Forest model — student ki academic aur demographic details daal kar "
    "predict karein ke wo Dropout karega, Enrolled rahega ya Graduate karega."
)

st.divider()

with st.form("student_form"):

    tab1, tab2, tab3, tab4 = st.tabs(
        ["👤 Personal & Academic Background", "💰 Financial & Family",
         "📘 1st Semester", "📗 2nd Semester & Economy"]
    )

    with tab1:
        c1, c2, c3 = st.columns(3)
        with c1:
            marital_status = st.number_input("Marital status (code 1-6)", 1, 6, 1)
            application_mode = st.number_input("Application mode (code)", 1, 60, 1)
            application_order = st.number_input("Application order", 0, 9, 1)
            course = st.number_input("Course (code)", 1, 10000, 33)
        with c2:
            attendance = st.selectbox("Daytime/evening attendance", ["Daytime", "Evening"])
            prev_qualification = st.number_input("Previous qualification (code)", 1, 50, 1)
            prev_qualification_grade = st.number_input("Previous qualification (grade)", 0.0, 200.0, 120.0)
            nacionality = st.number_input("Nacionality (code)", 1, 110, 1)
        with c3:
            admission_grade = st.number_input("Admission grade", 0.0, 200.0, 120.0)
            age = st.number_input("Age at enrollment", 15, 80, 20)
            gender = st.selectbox("Gender", ["Female", "Male"])
            international = st.selectbox("International student?", list(YES_NO.keys()))

    with tab2:
        c1, c2, c3 = st.columns(3)
        with c1:
            mother_qualification = st.number_input("Mother's qualification (code)", 1, 50, 1)
            father_qualification = st.number_input("Father's qualification (code)", 1, 50, 1)
        with c2:
            mother_occupation = st.number_input("Mother's occupation (code)", 0, 200, 1)
            father_occupation = st.number_input("Father's occupation (code)", 0, 200, 1)
        with c3:
            displaced = st.selectbox("Displaced?", list(YES_NO.keys()))
            special_needs = st.selectbox("Educational special needs?", list(YES_NO.keys()))
            debtor = st.selectbox("Debtor?", list(YES_NO.keys()))
            tuition_up_to_date = st.selectbox("Tuition fees up to date?", list(YES_NO.keys()), index=1)
            scholarship = st.selectbox("Scholarship holder?", list(YES_NO.keys()))

    with tab3:
        c1, c2 = st.columns(2)
        with c1:
            cu1_credited = st.number_input("Curricular units 1st sem (credited)", 0, 30, 0)
            cu1_enrolled = st.number_input("Curricular units 1st sem (enrolled)", 0, 30, 6)
            cu1_evaluations = st.number_input("Curricular units 1st sem (evaluations)", 0, 30, 6)
        with c2:
            cu1_approved = st.number_input("Curricular units 1st sem (approved)", 0, 30, 6)
            cu1_grade = st.number_input("Curricular units 1st sem (grade)", 0.0, 20.0, 12.0)
            cu1_without_eval = st.number_input("Curricular units 1st sem (without evaluations)", 0, 30, 0)

    with tab4:
        c1, c2 = st.columns(2)
        with c1:
            cu2_credited = st.number_input("Curricular units 2nd sem (credited)", 0, 30, 0)
            cu2_evaluations = st.number_input("Curricular units 2nd sem (evaluations)", 0, 30, 6)
            cu2_approved = st.number_input("Curricular units 2nd sem (approved)", 0, 30, 6)
            cu2_grade = st.number_input("Curricular units 2nd sem (grade)", 0.0, 20.0, 12.0)
            cu2_without_eval = st.number_input("Curricular units 2nd sem (without evaluations)", 0, 30, 0)
        with c2:
            unemployment_rate = st.number_input("Unemployment rate (%)", 0.0, 30.0, 10.0)
            inflation_rate = st.number_input("Inflation rate (%)", -5.0, 10.0, 1.0)
            gdp = st.number_input("GDP", -10.0, 10.0, 0.0)

    submitted = st.form_submit_button("🔮 Predict", use_container_width=True)

if submitted:

    if age <= 22:
        age_cat_adult, age_cat_mature = 0, 0
    elif age <= 30:
        age_cat_adult, age_cat_mature = 1, 0
    else:
        age_cat_adult, age_cat_mature = 0, 1

    row = {
        "Marital status": marital_status,
        "Application mode": application_mode,
        "Application order": application_order,
        "Course": course,
        "Daytime/evening attendance\t": 1 if attendance == "Daytime" else 0,
        "Previous qualification": prev_qualification,
        "Previous qualification (grade)": prev_qualification_grade,
        "Nacionality": nacionality,
        "Mother's qualification": mother_qualification,
        "Father's qualification": father_qualification,
        "Mother's occupation": mother_occupation,
        "Father's occupation": father_occupation,
        "Admission grade": admission_grade,
        "Displaced": YES_NO[displaced],
        "Educational special needs": YES_NO[special_needs],
        "Debtor": YES_NO[debtor],
        "Tuition fees up to date": YES_NO[tuition_up_to_date],
        "Gender": 1 if gender == "Male" else 0,
        "Scholarship holder": YES_NO[scholarship],
        "Age at enrollment": age,
        "International": YES_NO[international],
        "Curricular units 1st sem (credited)": cu1_credited,
        "Curricular units 1st sem (enrolled)": cu1_enrolled,
        "Curricular units 1st sem (evaluations)": cu1_evaluations,
        "Curricular units 1st sem (approved)": cu1_approved,
        "Curricular units 1st sem (grade)": cu1_grade,
        "Curricular units 1st sem (without evaluations)": cu1_without_eval,
        "Curricular units 2nd sem (credited)": cu2_credited,
        "Curricular units 2nd sem (evaluations)": cu2_evaluations,
        "Curricular units 2nd sem (approved)": cu2_approved,
        "Curricular units 2nd sem (grade)": cu2_grade,
        "Curricular units 2nd sem (without evaluations)": cu2_without_eval,
        "Unemployment rate": unemployment_rate,
        "Inflation rate": inflation_rate,
        "GDP": gdp,
        "Age_Category_Adult": age_cat_adult,
        "Age_Category_Mature": age_cat_mature,
    }

    input_df = pd.DataFrame([row])[FEATURE_ORDER]

    prediction = model.predict(input_df)[0]
    pred_label = CLASS_LABELS[prediction] if CLASS_LABELS is not None and isinstance(prediction, (int, np.integer)) else prediction

    st.divider()
    st.subheader("Result")

    if str(pred_label).lower() == "dropout":
        st.error(f"🔴 Predicted Outcome: **{pred_label}**")
    elif str(pred_label).lower() == "enrolled":
        st.warning(f"🟡 Predicted Outcome: **{pred_label}**")
    else:
        st.success(f"🟢 Predicted Outcome: **{pred_label}**")

    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(input_df)[0]
        proba_df = pd.DataFrame({
            "Class": CLASS_LABELS if CLASS_LABELS is not None else range(len(proba)),
            "Probability": proba
        }).sort_values("Probability", ascending=False)

        st.write("Prediction probabilities:")
        st.bar_chart(proba_df.set_index("Class"))
        st.dataframe(proba_df.reset_index(drop=True), use_container_width=True)

    with st.expander("Show raw input sent to the model"):
        st.dataframe(input_df, use_container_width=True)