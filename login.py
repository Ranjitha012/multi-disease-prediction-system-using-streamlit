import streamlit as st
import sqlite3
import hashlib
from style_utils import set_page_background 

def show_login_page():
    set_page_background() 
    st.title("Login")

    with st.form("Login Form"):
        email = st.text_input("Email")
        password = st.text_input("Password", type="password")
        login_button = st.form_submit_button("Login")

        if login_button:
            if not email or not password:
                st.error("Please enter both email and password.")
            else:
                try:
                    # Hash the entered password to compare with the stored hash
                    hashed_password = hashlib.sha256(password.encode()).hexdigest()

                    # Connect to the database
                    conn = sqlite3.connect('user_data.db')
                    c = conn.cursor()

                    # Check if user exists and password is correct
                    c.execute("SELECT * FROM users WHERE email = ? AND password = ?", (email, hashed_password))
                    user = c.fetchone()
                    conn.close()

                    if user:
                        st.success("Logged in successfully!")
                        st.session_state.logged_in = True
                        st.session_state.user_email = email
                        st.session_state.page = "home"
                        st.rerun()
                    else:
                        st.error("Invalid email or password.")
                except Exception as e:
                    st.error(f"An error occurred during login: {e}")

    if st.button("Go to Register"):
        st.session_state.page = "register"
        st.rerun()