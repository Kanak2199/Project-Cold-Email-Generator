import streamlit as st
from langchain_community.document_loaders import WebBaseLoader

from chains import Chain
from portfolio import Portfolio
from utils import clean_text

#cold email generator
st.set_page_config(layout="wide", page_title="Cold E-mail Generator", page_icon="📧")

st.markdown(
    """
    <style>
    body {
        background-color: #f0f0f5;  /* Light background */
    }
    
    /* Title style - Dark color */
    .main-title {
        font-size: 40px;
        color: #2c3e50;  /* Dark title color */
        text-align: center;
        margin-top: 20px;
        margin-bottom: 0;
    }
    
    /* Subtitle style - Dark color */
    .subtitle {
        font-size: 20px;
        color: #2c3e50;  /* Dark subtitle color */
        text-align: center;
        margin-bottom: 30px;
    }
    
    /* Input label styling - Dark color */
    .input-label {
        font-size: 18px;
        color: #ff6347;  /* Dark text */
        font-weight: bold;
        margin-bottom: 10px;
    }
    
    /* Button styling - Light background, dark text */
    div.stButton > button {
        color: #ff6347;  /* Dark text */
        background-color: #e0e0e0;  /* Light button background */
        border-radius: 12px;
        padding: 10px 20px;
        font-size: 16px;
        border: none;
    }
    
    div.stButton > button:hover {
        background-color: #cccccc;  /* Slightly darker on hover */
    }
    
    </style>
    """,
    unsafe_allow_html=True
)


def create_streamlit_app(llm, portfolio, clean_text):
    st.markdown("<h1 class='main-title'>📧 Cold E-mail Generator</h1>", unsafe_allow_html=True)
    st.markdown("<p class='subtitle'>Generate professional cold emails with ease.</p>", unsafe_allow_html=True)
    url_input = st.text_input("Enter job URL:", value="https://www.amazon.jobs/en/jobs/2698350/sr-data-scientist-search-data-science")
    submit_button = st.button("Generate Email")

    if submit_button:
        try:
            loader = WebBaseLoader([url_input])
            data = clean_text(loader.load().pop().page_content)
            portfolio.load_portfolio()
            jobs = llm.extract_jobs(data)
            for job in jobs:
                skills = job.get('skills', [])
                links = portfolio.query_links(skills)
                email = llm.write_mail(job, links)
                st.code(email, language='markdown')
        except Exception as e:
            st.error(f"An Error Occurred: {e}")


if __name__ == "__main__":

    chain = Chain()
    portfolio = Portfolio()
    #st.set_page_config(layout="wide", page_title="Cold Email Generator", page_icon="📧")
    create_streamlit_app(chain, portfolio, clean_text)
