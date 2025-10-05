import streamlit as st
import pandas as pd
import numpy as np

# Title
st.title("My First Streamlit App")

# Sidebar
st.sidebar.header("Options")
option = st.sidebar.selectbox("Choose a number", [1, 2, 3, 4, 5])
st.write(f"You selected: {option}")

# DataFrame example
st.subheader("Random Data Table")
df = pd.DataFrame(
    np.random.randn(10, 5),
    columns=[f"Col {i}" for i in range(1, 6)]
)
st.dataframe(df)

# Line chart
st.subheader("Line Chart")
st.line_chart(df)

# Checkbox example
if st.checkbox("Show raw data"):
    st.write(df)
