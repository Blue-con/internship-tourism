import streamlit as st
import pandas as pd
import pickle
import numpy as np

df = pd.read_csv("tourism_cleaned.csv")
with open("tourism_model.pkl", "rb") as file:
    model = pickle.load(file)

st.title("Tourism Recommendation System")

st.header("Predict Attraction Rating")

visit_year = st.selectbox(
    "Visit Year",
    sorted(df['VisitYear'].unique())
)

visit_month = st.selectbox(
    "Visit Month",
    sorted(df['VisitMonth'].unique())
)

visit_mode = st.selectbox(
    "Visit Mode",
    sorted(df['VisitMode'].unique())
)

attraction_type = st.selectbox(
    "Attraction Type",
    sorted(df['AttractionType'].unique())
)

country = st.selectbox(
    "Country",
    sorted(df['Country'].unique())
)

region = st.selectbox(
    "Region",
    sorted(df['Region'].unique())
)

continent = st.selectbox(
    "Continent",
    sorted(df['Continent'].unique())
)

visit_mode_encoded = df['VisitMode'].astype('category').cat.codes[
    df['VisitMode'].astype('category').cat.categories.get_loc(visit_mode)
]

attraction_type_encoded = df['AttractionType'].astype('category').cat.codes[
    df['AttractionType'].astype('category').cat.categories.get_loc(attraction_type)
]

country_encoded = df['Country'].astype('category').cat.codes[
    df['Country'].astype('category').cat.categories.get_loc(country)
]

region_encoded = df['Region'].astype('category').cat.codes[
    df['Region'].astype('category').cat.categories.get_loc(region)
]

continent_encoded = df['Continent'].astype('category').cat.codes[
    df['Continent'].astype('category').cat.categories.get_loc(continent)
]

input_data = np.array([[
    visit_year,
    visit_month,
    visit_mode_encoded,
    attraction_type_encoded,
    country_encoded,
    region_encoded,
    continent_encoded
]])

if st.button("Predict Rating"):

    prediction = model.predict(input_data)

    st.success(f"Predicted Attraction Rating: {prediction[0]:.2f}")

st.header("Attraction Recommendations")

selected_type = st.selectbox(
    "Choose Attraction Type",
    sorted(df['AttractionType'].unique())
)

recommendations = df[
    df['AttractionType'] == selected_type
]['Attraction'].value_counts().head(5)

st.write("Top Recommended Attractions:")

for attraction in recommendations.index:
    st.write(attraction)