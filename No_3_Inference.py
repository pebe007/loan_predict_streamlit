import streamlit as st
import numpy as np
import pickle

# Mapping dictionary untuk kolom bertipe kategorikal (berdasarkan unique value)
person_gender_mapping = {'male': 0, 'female': 1}
person_education_mapping = {
    'High School': 0,
    'Associate': 1,
    'Bachelor': 2,
    'Master': 3,
    'Doctorate': 4
}
person_home_ownership_mapping = {'RENT': 0, 'OWN': 1, 'MORTGAGE': 2, 'OTHER': 3}
loan_intent_mapping = {
    'EDUCATION': 0,
    'MEDICAL': 1,
    'VENTURE': 2,
    'PERSONAL': 3,
    'HOMEIMPROVEMENT': 4,
    'DEBTCONSOLIDATION': 5
}
previous_loan_defaults_mapping = {'No': 0, 'Yes': 1}

def main():
    st.title("Loan App Prediction")

    # Input dari user
    person_age = st.number_input("person_age", min_value=18, max_value=100, value=25)
    person_gender = st.selectbox("person_gender", ['male', 'female'])  # disesuaikan lowercase semua
    person_education = st.selectbox("person_education", list(person_education_mapping.keys()))
    person_income = st.number_input("person_income", min_value=0.0, value=30000.0)
    person_emp_exp = st.number_input("person_emp_exp", min_value=0, max_value=85, value=3)
    person_home_ownership = st.selectbox("person_home_ownership", list(person_home_ownership_mapping.keys()))
    loan_amnt = st.number_input("loan_amnt", min_value=100.0, value=5000.0)
    loan_intent = st.selectbox("loan_intent", list(loan_intent_mapping.keys()))
    loan_int_rate = st.number_input("loan_int_rate", min_value=0.0, max_value=100.0, value=12.5)
    loan_percent_income = st.number_input("loan_percent_income", min_value=0.0, max_value=1.0, value=0.2)
    cb_person_cred_hist_length = st.number_input("cb_person_cred_hist_length", min_value=1.0, value=5.0)
    credit_score = st.number_input("credit_score", min_value=300, max_value=850, value=650)
    previous_loan_defaults_on_file = st.selectbox("previous_loan_defaults_on_file", list(previous_loan_defaults_mapping.keys()))

    # Encoding input
    input_data = np.array([[ 
        person_age,
        person_gender_mapping[person_gender.lower()],  # lowercase input untuk jaga-jaga
        person_education_mapping[person_education],
        person_income,
        person_emp_exp,
        person_home_ownership_mapping[person_home_ownership],
        loan_amnt,
        loan_intent_mapping[loan_intent],
        loan_int_rate,
        loan_percent_income,
        cb_person_cred_hist_length,
        credit_score,
        previous_loan_defaults_mapping[previous_loan_defaults_on_file]
    ]]).astype(np.float64)

    # Load model
    try:
        with open("best_model.pkl", "rb") as file:
            model = pickle.load(file)
    except FileNotFoundError:
        st.error("❌ Model belum tersedia. Silakan training model terlebih dahulu.")
        return

    # Prediksi
    if st.button("Prediksi loan_status"):
        hasil = model.predict(input_data)
        if hasil[0] == 1:
            st.success("✅ loan_status: Disetujui")
        else:
            st.warning("❌ loan_status: Ditolak")

if __name__ == '__main__':
    main()
