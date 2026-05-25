import streamlit as st
import math
import matplotlib.pyplot as plt

# -----------------------------------
# PAGE CONFIGURATION
# -----------------------------------

st.set_page_config(
    page_title="Titanic Survival Prediction",
    page_icon="🚢",
    layout="wide"
)

# -----------------------------------
# HEADER SECTION
# -----------------------------------

st.markdown(
    """
    <h1 style='text-align:center; color:#1E90FF;'>
    Titanic Survival Prediction System
    </h1>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <h4 style='text-align:center; color:gray;'>
    Deep Learning Based Passenger Survival Prediction
    </h4>
    """,
    unsafe_allow_html=True
)

# -----------------------------------
# TITANIC IMAGE
# -----------------------------------

col1, col2, col3 = st.columns([1, 2, 1])

with col2:
    st.image(
        "data:image/jpeg;base64,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",
        use_container_width=True
    )

# -----------------------------------
# PROJECT DESCRIPTION
# -----------------------------------

st.markdown("## Project Description")

st.write("""
This application predicts whether a passenger would survive during the Titanic disaster
using an Artificial Neural Network (ANN).

The prediction is calculated using:
- Passenger Class
- Age
- Fare

The ANN uses:
- Tensorflow
- Sigmoid Activation
- Backpropagation Logic
""")

# -----------------------------------
# INPUT SECTION
# -----------------------------------

st.markdown("## Passenger Input Form")

col1, col2, col3 = st.columns(3)

with col1:
    pclass = st.selectbox(
        "Passenger Class",
        [1, 2, 3]
    )

with col2:
    age = st.slider(
        "Age",
        1,
        80,
        24
    )

with col3:
    fare = st.number_input(
        "Fare",
        min_value=0.0,
        max_value=600.0,
        value=120.0
    )

# -----------------------------------
# NORMALIZATION
# -----------------------------------

# Manual Min-Max Scaling

x1 = (pclass - 1) / (3 - 1)
x2 = (age - 1) / (80 - 1)
x3 = (fare - 0) / (600 - 0)

# -----------------------------------
# INITIAL WEIGHTS
# -----------------------------------

# Input -> Hidden

w_x1_h1 = 0.11
w_x2_h1 = 0.14
w_x3_h1 = 0.17

w_x1_h2 = 0.21
w_x2_h2 = 0.24
w_x3_h2 = 0.27

# Hidden biases

b_h1 = 0.1
b_h2 = 0.1

# Hidden -> Output

w_h1_o1 = 0.31
w_h2_o1 = 0.34

# Output bias

b_o1 = 0.1

# -----------------------------------
# SIGMOID FUNCTION
# -----------------------------------

def sigmoid(x):
    return 1 / (1 + math.exp(-x))

# -----------------------------------
# PREDICTION BUTTON
# -----------------------------------

if st.button("Predict Survival"):

    # -----------------------------------
    # FORWARD PROPAGATION
    # -----------------------------------

    # Hidden Layer

    net_h1 = (x1 * w_x1_h1) + \
             (x2 * w_x2_h1) + \
             (x3 * w_x3_h1) + b_h1

    net_h2 = (x1 * w_x1_h2) + \
             (x2 * w_x2_h2) + \
             (x3 * w_x3_h2) + b_h2

    h1 = sigmoid(net_h1)
    h2 = sigmoid(net_h2)

    # Output Layer

    net_o1 = (h1 * w_h1_o1) + \
             (h2 * w_h2_o1) + b_o1

    output = sigmoid(net_o1)

    # -----------------------------------
    # PREDICTION LOGIC
    # -----------------------------------

    if output > 0.5:
        result = "Survived"
    else:
        result = "Not Survived"

    confidence = output * 100

    # -----------------------------------
    # OUTPUT SECTION
    # -----------------------------------

    st.markdown("## Prediction Output")

    col4, col5, col6 = st.columns(3)

    with col4:
        st.metric(
            label="Prediction Result",
            value=result
        )

    with col5:
        st.metric(
            label="Survival Probability",
            value=f"{output:.4f}"
        )

    with col6:
        st.metric(
            label="Confidence Score",
            value=f"{confidence:.2f}%"
        )

    # -----------------------------------
    # VISUALIZATION
    # -----------------------------------

    st.markdown("## Probability Visualization")

    survive_prob = output
    not_survive_prob = 1 - output

    # Bar Chart

    fig, ax = plt.subplots(figsize=(6, 4))

    ax.bar(
        ["Survived", "Not Survived"],
        [survive_prob, not_survive_prob]
    )

    ax.set_ylabel("Probability")
    ax.set_title("Survival Probability")

    st.pyplot(fig)

    # Pie Chart

    fig2, ax2 = plt.subplots(figsize=(5, 5))

    ax2.pie(
        [survive_prob, not_survive_prob],
        labels=["Survived", "Not Survived"],
        autopct="%1.1f%%"
    )

    ax2.set_title("Prediction Distribution")

    st.pyplot(fig2)