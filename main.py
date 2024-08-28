import os
import streamlit as st
from google.oauth2 import service_account
from google.cloud import firestore
from dotenv import load_dotenv
from Login import login_page
from Register import register_page
from Chat_with_Gemini import chat_with_gemini_page
from gemini_with_bigquery import gemini_with_bigquery_page
from AutoML_Overview import automl
from AutoML_EDA_Insights import eda_analysis_page
from AutoML_EDA_Execution import eda_run_page
from AutoML_Data_Refinement import preprocess_page
from AutoML_Model_Crafting import model_page
from AutoML_Introduction import introduction_page
from upload_data_to_firestore import upload_data_to_firestore
from about_page import about_page
from upload_to_gcp import upload_to_gcp_page
from gcp_monitoring import gcp_monitoring_page
from rag_with_bigquery import rag_with_bigquery_page  # 新增导入

# 加载环境变量
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# 初始化 Firestore
credentials_file_path = './streamlit-gemini-ccf60-c9d613f7d829.json'
credentials = service_account.Credentials.from_service_account_file(credentials_file_path)
db = firestore.Client(credentials=credentials)

def main():
    # 初始化 session state
    if "default_description" not in st.session_state:
        st.session_state.default_description = (
            "私は工場エンジニアで、窒化アルミニウム（AIN）生産に関連するデータを持っています。"
            "このデータを使用して、目標変数（生産されたAIN粒子の密度）に対する予測モデルを構築したいと考えています。"
            "特征量はｍａｓｋｅｄされており、メタデータではありません。"
            "このプロジェクトでは、目標変数の分布をより正確に予測できる新しいアプローチを模索し、データの前処理、分析、およびモデリングを行います。"
        )
    if "eda_suggestions" not in st.session_state:
        st.session_state.eda_suggestions = " "
    if "preprocessing_suggestions" not in st.session_state:
        st.session_state.preprocessing_suggestions = " "
    if "model_results" not in st.session_state:
        st.session_state.model_results = " "
    if "gemini_feedback" not in st.session_state:
        st.session_state.gemini_feedback = " "

    # 创建侧边栏导航
    st.sidebar.title("Navigation")

    page = st.sidebar.radio("Go to", [
        "Login", 
        "Register", 
        "Chat with Gemini", 
        "Gemini with BigQuery", 
        "RAG with BigQuery",  # 新增选项
        "😊AutoML Introduction ", 
        "🌟AutoML: Overview", 
        "🔍AutoML: EDA Insights", 
        "🛠️AutoML: EDA Execution", 
        "⚙️AutoML: Data Refinement", 
        "🧠AutoML: Model Crafting", 
        "📊GCP Monitoring",  # 新增选项
        "Upload to GCP",     # 新增选项
        "About"
    ])

    if page == "Login":
        login_page(db)
    elif page == "Register":
        register_page(db)
    elif page == "Chat with Gemini":
        chat_with_gemini_page(db, GOOGLE_API_KEY)
    elif page == "Gemini with BigQuery":
        gemini_with_bigquery_page(db, GOOGLE_API_KEY, credentials_file_path)
    elif page == "RAG with BigQuery":
        rag_with_bigquery_page(db, GOOGLE_API_KEY, credentials_file_path)  # 调用 RAG with BigQuery 页面
    elif page == "😊AutoML Introduction ":
        introduction_page()
    elif page == "🌟AutoML: Overview":
        automl(db, GOOGLE_API_KEY)
    elif page == "🔍AutoML: EDA Insights":
        eda_analysis_page(db, GOOGLE_API_KEY, credentials_file_path)
    elif page == "🛠️AutoML: EDA Execution":
        eda_run_page(db, GOOGLE_API_KEY, credentials_file_path)
    elif page == "⚙️AutoML: Data Refinement":
        preprocess_page(db, GOOGLE_API_KEY, credentials_file_path)
    elif page == "🧠AutoML: Model Crafting":
        model_page()
    elif page == "📊GCP Monitoring":
        gcp_monitoring_page()  # 调用 GCP Monitoring 页面
    elif page == "Upload to GCP":
        upload_to_gcp_page()  # 调用 Upload to GCP 页面
    elif page == "About":
        about_page()

if __name__ == "__main__":
    main()
