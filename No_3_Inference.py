import streamlit as st
import numpy as np
import pickle

'''
[LO 1, LO 2, LO 3 – 15 Poin] Membuat code inference/prediction untuk proses deployment 
'''


# Mapping dictionary untuk kolom bertipe kategorikal (berdasarkan unique value)
gender = {'male': 0, 'female': 1}
education = {
    'High School': 0,
    'Associate': 1,
    'Bachelor': 2,
    'Master': 3,
    'Doctorate': 4
}
home_ownership = {'RENT': 0, 'OWN': 1, 'MORTGAGE': 2, 'OTHER': 3}
loan = {
    'EDUCATION': 0,
    'MEDICAL': 1,
    'VENTURE': 2,
    'PERSONAL': 3,
    'HOMEIMPROVEMENT': 4,
    'DEBTCONSOLIDATION': 5
}
previous_loan = {'No': 0, 'Yes': 1}

def main():
    st.title("Loan Prediction Application")

    # Mengambil Input dari user
    person_age = st.number_input("person_age", min_value=18, max_value=100, value=25)
    gender = st.selectbox("gender", ['male', 'female'])  # disesuaikan lowercase semua
    person_education = st.selectbox("person_education", list(education.keys()))
    person_income = st.number_input("person_income", min_value=0.0, value=30000.0)
    person_emp_exp = st.number_input("person_emp_exp", min_value=0, max_value=85, value=3)
    person_home_ownership = st.selectbox("person_home_ownership", list(home_ownership.keys()))
    loan_amnt = st.number_input("loan_amnt", min_value=100.0, value=5000.0)
    loan_intent = st.selectbox("loan_intent", list(loan.keys()))
    loan_int_rate = st.number_input("loan_int_rate", min_value=0.0, max_value=100.0, value=12.5)
    loan_percent_income = st.number_input("loan_percent_income", min_value=0.0, max_value=1.0, value=0.2)
    cb_person_cred_hist_length = st.number_input("cb_person_cred_hist_length", min_value=0.0, value=5.0)
    credit_score = st.number_input("credit_score", min_value=0, max_value=850, value=650)
    previous_loan_defaults_on_file = st.selectbox("previous_loan_defaults_on_file", list(previous_loan.keys()))

    # Encoding input
    input_data = np.array([[ 
        person_age,
        gender[gender.lower()],  # lowercase input untuk jaga-jaga
        education[person_education],
        person_income,
        person_emp_exp,
        home_ownership[person_home_ownership],
        loan_amnt,
        loan[loan_intent],
        loan_int_rate,
        loan_percent_income,
        cb_person_cred_hist_length,
        credit_score,
        previous_loan[previous_loan_defaults_on_file]
    ]]).astype(np.float64)

    # Loading model
    try:
        with open("best_model.pkl", "rb") as file:
            model = pickle.load(file)
    except FileNotFoundError:
        st.error("❌ Model belum tersedia")
        return

    # tombol Prediksi 
    if st.button("Prediksi loan_status"):
        hasil = model.predict(input_data)
        if hasil[0] == 1:
            st.success("✅ loan_status: Approved")
        else:
            st.warning("❌ loan_status: Declined")

if __name__ == '__main__':
    main()
