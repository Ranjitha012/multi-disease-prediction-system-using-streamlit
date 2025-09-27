import streamlit as st

def set_page_background():
    """
    Sets a simple, elegant gradient background for the page.
    """
    page_bg_css = """
    <style>
    [data-testid="stAppViewContainer"] {
        background: linear-gradient(to top, #cfd9df 0%, #e2ebf0 100%);
    }

    [data-testid="stHeader"] {
        background-color: rgba(0, 0, 0, 0);
    }

    [data-testid="stToolbar"] {
        right: 2rem;
    }
    
    /* Style for the login/register forms */
    div[data-baseweb="form"] {
        background-color: #ffffff;
        padding: 2rem;
        border-radius: 10px;
        box-shadow: 0 4px 8px 0 rgba(0,0,0,0.2);
    }

    /* Style for sidebar to make it consistent */
    [data-testid="stSidebar"] {
        background: linear-gradient(to bottom, #cfd9df 0%, #e2ebf0 100%);
    }
    </style>
    """
    st.markdown(page_bg_css, unsafe_allow_html=True)