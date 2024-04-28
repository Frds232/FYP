import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import seaborn as sns
from datetime import date
import random, string


import time
import bcrypt
import pickle
from datetime import datetime, timedelta
import pymysql

import streamlit as st
from streamlit_option_menu import option_menu
import streamlit_authenticator as stauth
import extra_streamlit_components as stx
import smtplib
from email.message import EmailMessage
import os

from dotenv import load_dotenv


from supabase import create_client, Client
from st_login_form import login_form

st.set_option('deprecation.showPyplotGlobalUse', False)

######################### USER AUTHENTICATION ##########################################


#################################################################################################
def init_connection():
    url = st.secrets["SUPABASE_URL"]
    key = st.secrets["SUPABASE_KEY"]
    return create_client(url, key)

supabase = init_connection()

def authenticate():
    client = login_form()

    if st.session_state["authenticated"]:
        if st.session_state["username"]:
            st.success(f"Welcome {st.session_state['username']}")                   
            app()

        else:
            st.success("Welcome guest")
    else:
        st.warning("Not authenticated")

def logout():
    st.session_state["authenticated"] = False
    st.session_state["username"] = None
    st.rerun()


def validate_input(input_str):
    # Split the input string into a list
    input_lst = input_str.split(',')
    
    # Check if the length is 30
    if len(input_lst) != 30:
        return False
    
    # Check if all elements are numbers
    for element in input_lst:
        try:
            float(element)
        except ValueError:
            return False
    
    return True


def generate_otp():
    return str(random.randint(100000, 999999))

def fraudDetection():
     # Mapping between original feature names and selected features
    feature_mapping = {
        'Time': 0,
        'V1': 1, 'V2': 2, 'V3': 3, 'V4': 4, 'V5': 5,
        'V6': 6, 'V7': 7, 'V8': 8, 'V9': 9, 'V10': 10,
        'V11': 11, 'V12': 12, 'V13': 13, 'V14': 14, 'V15': 15,
        'V16': 16, 'V17': 17, 'V18': 18, 'V19': 19, 'V20': 20,
        'V21': 21, 'V22': 22, 'V23': 23, 'V24': 24, 'V25': 25,
        'V26': 26, 'V27': 27, 'V28': 28, 'Amount': 29
    }

    selected_features = ['V14', 'V10', 'V12', 'V7', 'V27', 'V17', 'V8', 'V4', 'V1', 'V9', 'V3', 'V13', 'V11', 'V20', 'V2', 'V16', 'V19']

    # Loading our trained model
    pickle_in = open("xgb_tuned.pkl", "rb")
    xgb = pickle.load(pickle_in)

    st.title("Credit Card Fraud Detection System")
    st.write("Enter the following features to check if the transaction is legitimate or fraudulent:")

    with st.form(key="fraudDetection"):
        try:
            # Get user input
            input_df = st.text_input('Input All features', key="10")

            # Validate the input
            if validate_input(input_df):
                input_df_lst = [float(x) for x in input_df.split(',')]
                if input_df and len(input_df_lst) == len(feature_mapping):
                    filtered_input = [input_df_lst[feature_mapping[col]] for col in selected_features]
                    st.write("features have been filtered")
                else:
                    st.write("features have not been filtered")
                    st.error("Input Features length mismatch!")
                    return
        except:
            st.warning('Please enter 30 comma-separated feature values.')

        predict = st.form_submit_button("Predict")

        if predict:
            features = np.array(filtered_input, dtype=np.float64)
            st.write("inside predict")
            try:
                prediction = xgb.predict(features.reshape(1, -1))
                if prediction[0] == 0:
                    st.success("Legitmate Transaction!")
                    return
                else:
                    st.error("Fraudulent Transaction!An OTP has been sent to your email.Please verify.")
                    authenticate_otp()
                    return
            except Exception as e:
                st.error(str(e))


# Function to send OTP via email
def send_otp(email, otp):
    msg = EmailMessage()
    msg.set_content(f"Your OTP is: {otp}")
    msg['Subject'] = 'OTP for Transaction Authentication'
    msg['From'] = 'fyptesting9@gmail.com' 
    msg['To'] = email
    smtp_server = 'smtp.gmail.com' 
    smtp_port = 587  
    smtp_username = 'fyptesting9@gmail.com'  
    smtp_password = 'bqugkmwobwdxgkwm'

    try:
        with smtplib.SMTP(smtp_server, smtp_port,  timeout=120) as server:
            server.starttls()
            server.login(smtp_username, smtp_password)
            server.send_message(msg)
            st.warning("OTP sent successfully!")
            return True
    except Exception as e:
        st.error(f"Failed to send OTP: {e}")
        return

def authenticate_otp():
    user_email = 'fyptesting9@gmail.com' #fakhirfirdaus7@gmail.com

    if 'otp' not in st.session_state:
        st.session_state.otp = generate_otp()
    email_sent = send_otp(user_email, st.session_state.otp)

    message_placeholder = st.empty()
    
    if email_sent:
        entered_otp = st.text_input("Enter OTP:")
        if entered_otp:

            # Check if the entered OTP is correct
            if entered_otp == st.session_state.otp:
                st.success("OTP verification successful. Transaction confirmed.")
            elif entered_otp:
                st.error("Invalid OTP. Transaction denied.")
        else:
            st.warning("Please enter OTP.")

def app():

    df = loadData()
    if 'df' not in st.session_state:
        st.session_state.df = df

    with st.sidebar:        
        option = option_menu(
            menu_title='Menu',
            options= ['Home', 'Data Analysis', 'Fraud Detection', 'About Us', 'Logout'],
            icons=['house', 'bar-chart-line', 'shield-exclamation', 'info-circle', 'escape'],
            menu_icon='chat-text-fill',
            ## default_index=0,
            styles={
                "container": {"padding": "5!important","background-color":'#1f415e'},
    "icon": {"color": "white", "font-size": "18px"}, 
    "nav-link": {"color":"white","font-size": "18px", "text-align": "left", "margin":"0px", "--hover-color": "#5c9aad"},
    "nav-link-selected": {"background-color": "#5c9aad"},}
        )

    if option == "Home":
        display_home_page()

    if option == "Data Analysis":
        data_analysis(st.session_state.df)

    if option == "Fraud Detection":
        fraudDetection()
    
    if option == "About Us":
        aboutUs()

    if option == 'Logout':
        logout()


@st.cache_data(show_spinner=False)
def display_home_page():

    st.markdown("<h1 class='title'>Credit Card Fraud Detection</h1>", unsafe_allow_html=True)

    st.markdown(
         f"""
         <style>
         .stApp {{
             background-image: url("https://cdn.pixabay.com/photo/2017/09/18/17/15/credit-2762536_1280.jpg");
             background-attachment: fixed;
             background-size: cover

         }}
         .title {{
            text-align: center;
            font-size: 36px;
            color: white;
            background: rgba(0, 0, 0, 0.5);  /* Adjust the alpha value for transparency */
            padding: 10px;
            border-radius: 10px;
         }}
         </style>
         """,
         unsafe_allow_html=True
     )

@st.cache_data(show_spinner=False)
def overall_background():
    st.markdown(
         f"""
         <style>
         .stApp {{
             background: linear-gradient(rgba(0, 0, 0, 0.6), rgba(0, 0, 0, 0.5)), url("https://cdn.pixabay.com/photo/2017/09/18/17/15/credit-2762536_1280.jpg");
             background-attachment: fixed;
             background-size: cover

         }}
         .title {{
            text-align: center;
            font-size: 36px;
            color: white;
            background: rgba(0, 0, 0, 0.7);  /* Adjust the alpha value for transparency */
            padding: 10px;
            border-radius: 10px;
            brightness: 20;
         }}
         </style>
         """,
         unsafe_allow_html=True
     )

@st.cache_data(show_spinner=False)
def loadData():
    df = pd.read_csv('creditcard.csv')

    df = df[~df.duplicated()]

    return df

@st.cache_data(show_spinner=False)
def data_analysis(df):
    st.title("Data Analysis")

    # Plot distribution of the dataset
    st.subheader("Distribution of Class")

    # Create a new figure explicitly
    fig, ax = plt.subplots()
    
    sns.countplot(data=df, x="Class")

    # Add counts (integers) on top of each bar
    for p in ax.patches:
        height = int(p.get_height())
        ax.annotate(f'{height}', (p.get_x() + p.get_width() / 2., height),
                    ha='center', va='center', xytext=(0, 10), textcoords='offset points')
    st.pyplot(fig)

    # Adding additional information about class distribution
    st.markdown("""
    #### What the Numbers Tell Us

    - **Non-Fraudulent Transactions (Class 0):** We have a substantial count of 283,253 instances representing non-fraudulent transactions.
    - **Fraudulent Transactions (Class 1):** While relatively rare, our dataset contains 473 instances of fraudulent transactions.

    Understanding this distribution helps us tailor our analysis and develop robust fraud detection techniques to protect against potential threats.
    """)

    # Custom colors for our data
    gray_color = "#CCCCCC"  # Grey for regular txs
    red_color = "#F0544F"  # Red for fraudulent txs

    color_pal = [gray_color, red_color]

    fig, ax = plt.subplots(ncols=2, figsize=(20, 10))

    sns.boxplot(data=df,
                x="Class",
                y="Amount",
                hue="Class",
                palette=color_pal,
                showfliers=True,
                ax=ax[0])

    sns.boxplot(data=df,
                x="Class",
                y="Amount",
                hue="Class",
                palette=color_pal,
                showfliers=False,
                ax=ax[1])

    ax[0].set_title("Transaction Amount Box Plot (Including Outliers)")
    ax[1].set_title("Transaction Amount Box Plot (Excluding Outliers)")

    legend_labels = ['Non-fraud', 'Fraud']
    for i in range(2):
        handles, _ = ax[i].get_legend_handles_labels()
        ax[i].legend(handles, legend_labels)

    st.subheader("Transaction Amount Box Plots")
    st.pyplot(fig)

    # Transaction Amounts: Box Plot Comparison
    st.markdown("""
    This visual comparison presents two box plots side by side, revealing differences in transaction amounts between non-fraudulent and fraudulent transactions:

    - **Including Outliers**
      - **Non-Fraud Transactions (Class 0):** Majority occur at lower amounts, with outliers extending up to approximately 25,000.
      - **Fraud Transactions (Class 1):** Slightly higher median amounts with fewer extreme outliers compared to non-fraudulent transactions.

    - **Excluding Outliers**
      - **Non-Fraud Transactions (Class 0):** Detailed view without outliers, showcasing median around 25-30 and upper quartile below 50.
      - **Fraud Transactions (Class 1):** Higher median and broader interquartile range, indicating wider spread in transaction amounts, with upper whisker reaching around 250.

    - **Observations**
      - **Outliers Impact:** Excluding outliers offers clearer comparison between typical transaction amounts, reducing distortion caused by extreme values.
      - **Transaction Amounts:** Fraudulent transactions typically involve higher amounts compared to non-fraudulent ones, evident from box plot positions and spreads.

    These plots are vital for financial analysis, aiding in identifying patterns distinguishing fraudulent from legitimate transactions, crucial for effective fraud detection systems.
    """)


@st.cache_data(show_spinner=False)
def aboutUs():
    st.title("About Us")
    st.write(
        """
        Welcome to Credit Card Fraud Detection System!

        Our mission is to provide a secure and reliable solution for identifying and preventing credit card fraud using Machine Learning.

        **How it Works:**

        1. **Data Collection:**
            We collect a diverse set of credit card transactions to build a robust dataset.

        2. **Data Preprocessing:**
            The collected data undergoes preprocessing to handle missing values, scale features, and ensure it is suitable for training our models.

        3. **Machine Learning Models:**
            We leverage state-of-the-art machine learning algorithms to create models that can detect patterns and anomalies in credit card transactions.

        4. **Real-time Monitoring:**
            Our system continuously monitors transactions in real-time, identifying potential fraud based on the learned patterns.

        5. **User-Friendly Interface:**
            This Streamlit web application provides an easy-to-use interface for users to interact with our fraud detection system.

        **Meet the Team:**

        - Firdaus Fakhir Khan: Data Scientist / ML Developer

        **Contact Us:**
        If you have any questions or feedback, feel free to reach out to us at fyptesting9@gmail.com.

        Thank you for trusting our Credit Card Fraud Detection System!
        """
    )


if __name__ == "__main__":
    overall_background()    
    authenticate()

