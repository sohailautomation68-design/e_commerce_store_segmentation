import streamlit as st
import pandas as pd
import numpy as np
import joblib


# ==========================
# PAGE CONFIG
# ==========================
st.set_page_config(
    page_title="Online Shopper Segmentation",
    page_icon="🛒",
    layout="wide",
)

# ==========================
# LOAD MODEL
# ==========================
kmeans = joblib.load("kmeans_model.joblib")
sc = joblib.load("scaler.joblib")

# ==========================
# LOAD IMAGES
# ==========================
try:
    logo = Image.open("images/logo.png")
except:
    logo = None

try:
    banner = Image.open("images/shopping.png")
except:
    banner = None

# ==========================
# SIDEBAR
# ==========================
with st.sidebar:

    if logo:
        st.image(logo, width=180)

    st.title("🛒 Customer Segmentation")

    st.markdown("---")

    st.write("### Project Information")

    st.info("""
Algorithm:
- K-Means Clustering

Dataset:
- Online Shoppers Intention

Machine Learning:
- Unsupervised Learning
""")

    st.markdown("---")

    st.write("### Developed By")

    st.success("""
Muhammad Sohail

BS Computer Science
""")

# ==========================
# HEADER
# ==========================
st.markdown("""
<div style='background:linear-gradient(90deg,#0f62fe,#42a5f5);
padding:25px;
border-radius:15px;'>

<h1 style='color:white;text-align:center;'>
🛒 Online Shopper Customer Segmentation
</h1>

<h4 style='color:white;text-align:center;'>
AI Powered Customer Behaviour Analysis using K-Means Clustering
</h4>

</div>
""", unsafe_allow_html=True)

st.write("")

if banner:
    st.image(banner, use_container_width=True)

st.write("")

st.subheader("Enter Visitor Information")

st.markdown("---")

# ==========================
# TWO COLUMNS
# ==========================

col1, col2 = st.columns(2)

# ==========================
# LEFT COLUMN
# ==========================

with col1:

    administrative = st.number_input(
        "Administrative",
        min_value=0.0,
        value=0.0
    )

    administrative_duration = st.number_input(
        "Administrative Duration",
        min_value=0.0,
        value=0.0
    )

    informational = st.number_input(
        "Informational",
        min_value=0.0,
        value=0.0
    )

    informational_duration = st.number_input(
        "Informational Duration",
        min_value=0.0,
        value=0.0
    )

    product_related = st.number_input(
        "Product Related",
        min_value=0.0,
        value=0.0
    )

    product_related_duration = st.number_input(
        "Product Related Duration",
        min_value=0.0,
        value=0.0
    )

    bounce_rates = st.number_input(
        "Bounce Rates",
        min_value=0.0,
        value=0.0
    )

    exit_rates = st.number_input(
        "Exit Rates",
        min_value=0.0,
        value=0.0
    )

# ==========================
# RIGHT COLUMN
# ==========================

with col2:

    page_values = st.number_input(
        "Page Values",
        min_value=0.0,
        value=0.0
    )

    special_day = st.slider(
        "Special Day",
        0.0,
        1.0,
        0.0
    )

    month = st.selectbox(
        "Month",
        list(range(1,13))
    )

    operating_system = st.selectbox(
        "Operating System",
        [1,2,3,4,5,6,7,8]
    )

    browser = st.selectbox(
        "Browser",
        [1,2,3,4,5,6,7,8,9,10,11,12,13]
    )

    region = st.selectbox(
        "Region",
        [1,2,3,4,5,6,7,8,9]
    )

    traffic_type = st.selectbox(
        "Traffic Type",
        list(range(1,21))
    )

    visitor_type = st.selectbox(
        "Visitor Type",
        [0,1]
    )

    weekend = st.selectbox(
        "Weekend",
        [0,1]
    )

st.write("")
st.write("")

predict = st.button(
    "🔍 Predict Customer Segment",
    use_container_width=True
)

st.write("")
st.write("")

st.markdown("---")

st.markdown("""
<center>
<h5>Online Shopper Segmentation System</h5>
Developed using Streamlit & Scikit-Learn
</center>
""", unsafe_allow_html=True)
# ==========================
# PREDICTION LOGIC
# ==========================

cluster_info = {

    0: {
        "name": "Potential Buyers 🛍️",
        "description": 
        "These visitors show buying interest. "
        "They browse valuable pages and have a good chance of conversion."
    },

    1: {
        "name": "Regular Visitors 👤",
        "description":
        "These visitors show normal browsing behaviour "
        "but do not show very strong purchase signals."
    },

    2: {
        "name": "Quick Exit Visitors 🚪",
        "description":
        "These visitors leave the website quickly, "
        "view fewer products, and have low engagement."
    },

    3: {
        "name": "Highly Engaged Customers ⭐",
        "description":
        "These visitors spend more time exploring products "
        "and show the highest level of engagement."
    }
}


if predict:

    # Create input dataframe
    input_data = pd.DataFrame({

        "Administrative": [administrative],

        "Administrative_Duration": 
        [administrative_duration],

        "Informational": 
        [informational],

        "Informational_Duration":
        [informational_duration],

        "ProductRelated":
        [product_related],

        "ProductRelated_Duration":
        [product_related_duration],

        "BounceRates":
        [bounce_rates],

        "ExitRates":
        [exit_rates],

        "PageValues":
        [page_values],

        "SpecialDay":
        [special_day],

        "Month":
        [month],

        "OperatingSystems":
        [operating_system],

        "Browser":
        [browser],

        "Region":
        [region],

        "TrafficType":
        [traffic_type],

        "VisitorType":
        [visitor_type],

        "Weekend":
        [weekend]
    })


    # Scaling input data
    scaled_input = sc.transform(input_data)


    # Predict cluster
    prediction = kmeans.predict(scaled_input)[0]


    # Get customer information
    result = cluster_info[prediction]


    st.write("")
    st.write("")

    # ==========================
    # RESULT CARD
    # ==========================

    st.markdown(
        f"""
        <div style="
        background-color:#f0f8ff;
        padding:30px;
        border-radius:20px;
        border-left:8px solid #0f62fe;
        ">

        <h2>
        🎯 Customer Segment Result
        </h2>

        <h1 style="color:#0f62fe;">
        {result['name']}
        </h1>


        <p style="
        font-size:18px;
        ">
        {result['description']}
        </p>


        <hr>

        <h3>
        Cluster Number: {prediction}
        </h3>


        </div>
        """,

        unsafe_allow_html=True
    )


    st.success(
        "Prediction completed successfully!"
    )