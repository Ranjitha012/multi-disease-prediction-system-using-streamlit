import streamlit as st
from login import show_login_page
from register import show_register_page
from home import show_home_page

def main():
    # Initialize session state variables if they don't exist
    if 'logged_in' not in st.session_state:
        st.session_state.logged_in = False
    if 'page' not in st.session_state:
        st.session_state.page = 'login' # Start with login page

    # Page routing
    if st.session_state.logged_in:
        # If logged in, show the home page
        st.session_state.page = 'home'
        show_home_page()
    elif st.session_state.page == 'register':
        # Show the register page
        show_register_page()
    else:
        # By default, show the login page
        st.session_state.page = 'login'
        show_login_page()

if __name__ == "__main__":
    main()