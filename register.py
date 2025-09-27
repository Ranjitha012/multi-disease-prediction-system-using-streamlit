import streamlit as st
import sqlite3
import hashlib
from style_utils import set_page_background # <-- IMPORT THE FUNCTION

def show_register_page():
    set_page_background() # <-- CALL THE FUNCTION
    st.title("Create an Account")

    with st.form("Register Form"):
        name = st.text_input("Name")
        email = st.text_input("Email")
        password = st.text_input("Password", type="password")
        confirm_password = st.text_input("Confirm Password", type="password")
        submitted = st.form_submit_button("Register")

        if submitted:
            if not name or not email or not password or not confirm_password:
                st.error("Please fill out all fields.")
            elif password != confirm_password:
                st.error("Passwords do not match.")
            else:
                try:
                    # Hash the password for security
                    hashed_password = hashlib.sha256(password.encode()).hexdigest()

                    # Connect to the database
                    conn = sqlite3.connect('user_data.db')
                    c = conn.cursor()

                    # Insert new user into the database
                    c.execute("INSERT INTO users (name, email, password) VALUES (?, ?, ?)",
                              (name, email, hashed_password))
                    conn.commit()
                    conn.close()

                    st.success("Registration successful! You can now log in.")
                    # Automatically switch to the login page
                    st.session_state.page = "login"
                    st.rerun()

                except sqlite3.IntegrityError:
                    st.error("This email address is already registered.")
                except Exception as e:
                    st.error(f"An error occurred: {e}")