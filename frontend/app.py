import streamlit as st
import requests
import os
import plotly.io as pio  # ✅ Needed for rendering JSON plot

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
            files = {
                'file': (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)
            }
            response = requests.post(f"{BACKEND_URL}/upload_file", files=files)

            if response.status_code == 200:
                result = response.json()
                st.session_state["uploaded_file_name"] = result.get("file_name")
                st.success(result.get("status"))
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
            payload = {
                "query": query,
                "file_name": st.session_state.get("uploaded_file_name")
            }

            response = requests.post(f"{BACKEND_URL}/analyze_query", json=payload)

            if response.status_code == 200:
                result = response.json()
                st.success(f"Response from {result.get('agent')} agent:")

                response_content = result.get("response")

                # ✅ Case 1: Natural text answer
                if isinstance(response_content, str):
                    st.write(response_content)

                # ✅ Case 2: Plot returned as JSON
                elif isinstance(response_content, dict) and "plot" in response_content:
                    fig = pio.from_json(response_content["plot"])
                    st.plotly_chart(fig, use_container_width=True)
                    if "caption" in response_content:
                        st.caption(response_content["caption"])

                else:
                    st.warning("Unexpected response format.")

            else:
                st.error(f"Error: {response.status_code} - {response.json().get('detail', 'Unknown error')}")

        except requests.exceptions.ConnectionError:
            st.error("Connection error. Is the FastAPI backend running and accessible?")
        except Exception as e:
            st.error(f"An unexpected error occurred: {e}")
    else:
        st.warning("Please enter a query.")

