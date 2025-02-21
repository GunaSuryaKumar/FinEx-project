import streamlit as st
import pandas as pd
import os
import uuid
from mistralai import Mistral
from unstructured.partition.pdf import partition_pdf
import nltk
nltk.download('all')


## Table extraction function (using unstructured package)
def extract_tables_from_pdf(filename, strategy='hi_res'):
    """
    Extracts all tables from the given PDF file.

    Args:
        filename (str): Path to the PDF file.
        strategy (str): Strategy for table extraction ('hi_res' or other supported strategies).

    Returns:
        Tuple[List[pd.DataFrame], List[str]]: A tuple containing a list of DataFrames for each table 
        and a list of HTML representations of the tables.
    """
    elements = partition_pdf(
        filename=filename,
        infer_table_structure=True,
        strategy=strategy,
    )

    # Filter out all elements categorized as "Table"
    tables = [el for el in elements if el.category == "Table"]
    
    if not tables:
        print("No tables found in the PDF.")
        return [], []

    # Extract HTML from each table's metadata and convert to DataFrame
    tables_html = [table.metadata.text_as_html for table in tables]
    dfs = []
    for idx, html in enumerate(tables_html, start=1):
        try:
            # pd.read_html returns a list of DataFrames; take the first one
            df = pd.read_html(html)[0]
            dfs.append(df)
            print(f"Table {idx} extracted successfully.")
        except ValueError as ve:
            print(f"Failed to parse HTML for Table {idx}: {ve}")
            continue

    return dfs, tables_html


## Summarization code (using Mistral)
def summarize_table(table_text, max_new_tokens=100, num_return_sequences=1):
    """
    Summarizes the table data using an LLM (Mistral).

    Args:
        table_text (str): Text data extracted from the table.
        max_new_tokens (int): Maximum number of tokens in the summary.
        num_return_sequences (int): Number of summaries to return.

    Returns:
        str: Summary of the table.
    """
    prompt = (
        "Summarize the following table data:\n\n" 
        f"{table_text}\n\n"
        "Provide a concise summary of the key points."
    )

    api_key = "MMNlnPxuMxBfeIGG4pIGIfSwBdIgjlVA"  # Mistral API key

    model = "mistral-large-latest"

    client = Mistral(api_key=api_key)

    chat_response = client.chat.complete(
        model=model,
        messages=[
            {
                "role": "user",
                "content": prompt,
            },
        ]
    )

    return chat_response.choices[0].message.content


def main():
    # Set page configuration
    st.set_page_config(
        page_title="📄 PDF Table Extraction & Summarization",
        layout="wide",
        page_icon="📈",
    )

    # Custom CSS for additional styling
    st.markdown("""
        <style>
            .header-container {
                background: linear-gradient(135deg, #F8F9FA 0%, #E9ECEF 100%);
                padding: 20px;
                border-radius: 15px;
                margin-bottom: 30px;
                box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            }

            .section {
                background: linear-gradient(135deg, #FFFFFF 0%, #F8F9FA 100%);
                padding: 25px;
                border-radius: 15px;
                margin-bottom: 30px;
                box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            }

            .header {
                font-size: 2.5em;
                font-weight: 800;
                text-align: center;
                background: linear-gradient(120deg, #0D6EFD 0%, #0B5ED7 100%);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                margin: 0;
                padding: 10px;
                letter-spacing: 1px;
            }

            .subheader {
                font-size: 1.8em;
                font-weight: 600;
                background: linear-gradient(120deg, #0D6EFD 0%, #0B5ED7 100%);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                margin-top: 20px;
                margin-bottom: 10px;
                text-align: left;
            }

            .table-container {
                background: linear-gradient(135deg, #FFFFFF 0%, #F8F9FA 100%);
                padding: 20px;
                border-radius: 10px;
                margin: 20px 0;
                box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
            }

            @media (prefers-color-scheme: dark) {
                .header-container {
                    background: linear-gradient(135deg, #212529 0%, #343A40 100%);
                }
                
                .section {
                    background: linear-gradient(135deg, #212529 0%, #2B3035 100%);
                }
                
                .table-container {
                    background: linear-gradient(135deg, #212529 0%, #2B3035 100%);
                }
                
                .header {
                    background: linear-gradient(120deg, #6EA8FE 0%, #9EC5FE 100%);
                    -webkit-background-clip: text;
                    -webkit-text-fill-color: transparent;
                }
                
                .subheader {
                    background: linear-gradient(120deg, #6EA8FE 0%, #9EC5FE 100%);
                    -webkit-background-clip: text;
                    -webkit-text-fill-color: transparent;
                }
            }

            .stButton > button {
                background: linear-gradient(135deg, #2196F3 0%, #1976D2 100%);
                color: white;
                border: none;
                padding: 10px 20px;
                border-radius: 5px;
                transition: all 0.3s ease;
            }

            .stButton > button:hover {
                background: linear-gradient(135deg, #1976D2 0%, #1565C0 100%);
                box-shadow: 0 4px 6px rgba(0, 0, 0, 0.2);
            }
        </style>
    """, unsafe_allow_html=True)

    # Update the header with container
    st.markdown("""
        <div class="header-container">
            <h1 class="header">📄 PDF Table Extraction and Summarization</h1>
        </div>
    """, unsafe_allow_html=True)

    # Move temp_filename to session state to persist across reruns
    if 'temp_filename' not in st.session_state:
        st.session_state.temp_filename = None

    # Sidebar for file upload
    with st.sidebar:
        st.markdown('<h2 style="color: white;">Upload Your PDF</h2>', unsafe_allow_html=True)
        uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

        if uploaded_file is not None:
            # Clean up previous temporary file if it exists
            if st.session_state.temp_filename and os.path.exists(st.session_state.temp_filename):
                try:
                    os.remove(st.session_state.temp_filename)
                except Exception as e:
                    st.warning(f"Could not delete previous temporary file: {e}")

            # Save uploaded file to a unique temporary location
            st.session_state.temp_filename = f"temp_{uuid.uuid4().hex}.pdf"
            with open(st.session_state.temp_filename, "wb") as f:
                f.write(uploaded_file.getbuffer())
            st.markdown('<div class="success">✅ PDF Uploaded Successfully!</div>', unsafe_allow_html=True)

    if uploaded_file is not None:
        try:
            # ------------------ Table Extraction ------------------
            st.markdown('<div class="section">', unsafe_allow_html=True)

            st.markdown('<div class="subheader">📊 Table Extraction</div>', unsafe_allow_html=True)

            dfs, html_tables = extract_tables_from_pdf(st.session_state.temp_filename)
            if not dfs:
                st.warning("No tables found in the PDF.")
            else:
                for idx, df in enumerate(dfs):
                    st.markdown(f"### Table {idx+1}")
                    st.dataframe(df)
                    table_text = df.to_string()
                    summary = summarize_table(table_text)
                    st.markdown(f"**Summary:** {summary}")

            st.markdown('</div>', unsafe_allow_html=True)

        except Exception as e:
            st.error(f"An error occurred: {e}")
    else:
        st.warning("Please upload a PDF to proceed.")

if __name__ == "__main__":
    main()
