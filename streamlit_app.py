import streamlit as st
import os
import sys

# Add the src directory to the path so we can import modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

# Import main dashboard functionality
from visualization.dashboard import main

# Display a startup message
st.set_page_config(
    page_title="IntelliStock: Supply Chain Analytics",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Run the main dashboard function
if __name__ == "__main__":
    main() 