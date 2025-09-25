import streamlit as st
import requests
import os

# Define the backend URL
BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")

st.set_page_config(
    page_title="AI Agent System",
    page_icon="🤖",
    layout="wide",
)

st.title("AI Data & Research Assistant")
st.markdown("---")

st.markdown("### Upload a file")
uploaded_file = st.file_uploader(
    "Upload a document (.pdf, .docx) or a data file (.csv, .xlsx)",
    type=["pdf", "docx", "csv", "xlsx"]
)

if st.button("Upload"):
    if uploaded_file:
        st.info(f"Uploading {uploaded_file.name}...")
        try:
            # Correctly package the file to send to the FastAPI backend
            files = {
                'file': (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)
            }
            response = requests.post(f"{BACKEND_URL}/upload_file", files=files)
            
            if response.status_code == 200:
                st.success(response.json().get("status"))
            else:
                st.error(f"Error: {response.status_code} - {response.json().get('detail', 'Unknown error')}")

        except requests.exceptions.ConnectionError:
            st.error("Connection error. Is the FastAPI backend running and accessible?")
        except Exception as e:
            st.error(f"An unexpected error occurred: {e}")
    else:
        st.warning("Please upload a file first.")

st.markdown("---")

st.markdown("### Ask a question")
query = st.text_area("Enter your query:", height=100)

if st.button("Analyze"):
    if query:
        st.info("Analyzing query...")
        try:
            payload = {"query": query}
            response = requests.post(f"{BACKEND_URL}/analyze_query", json=payload)
            
            if response.status_code == 200:
                result = response.json()
                st.success(f"Response from {result.get('agent')} agent:")
                st.write(result.get('response'))
            else:
                st.error(f"Error: {response.status_code} - {response.json().get('detail', 'Unknown error')}")
        
        except requests.exceptions.ConnectionError:
            st.error("Connection error. Is the FastAPI backend running and accessible?")
        except Exception as e:
            st.error(f"An unexpected error occurred: {e}")
    else:
        st.warning("Please enter a query.")
