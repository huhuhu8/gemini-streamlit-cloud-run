import pandas as pd
import streamlit as st
from google.cloud import bigquery
import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
import time

def gemini_with_bigquery_page(db, api_key, credentials_file_path):
    st.title("Gemini with BigQuery")

    # 检查并初始化 'user'
    if 'user' not in st.session_state:
        st.warning("Please log in to access this page.")
        return
    
    user_id = st.session_state['user']

    # 设置凭证文件路径
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = credentials_file_path

    # 设置项目 ID 和位置
    project_id = 'streamlit-gemini-ccf60'
    location = 'asia-northeast1'  # 指定数据集位置

    # 初始化 BigQuery 客户端，并指定位置
    client = bigquery.Client(project=project_id, location=location)

    # 获取数据集列表
    try:
        datasets = list(client.list_datasets())
        if not datasets:
            st.warning("No datasets found in BigQuery.")
            return
        
        dataset_id = st.selectbox("Select a dataset", [dataset.dataset_id for dataset in datasets])

        # 获取表格列表
        tables = list(client.list_tables(dataset_id))
        if not tables:
            st.warning(f"No tables found in dataset {dataset_id}.")
            return
        
        table_id = st.selectbox("Select a table", [table.table_id for table in tables])

        # 查询表数据
        table_ref = client.dataset(dataset_id).table(table_id)
        table = client.get_table(table_ref)
        rows = client.list_rows(table, max_results=1000)  # 获取最多1000行数据
        df = rows.to_dataframe()

        st.write(f"Preview of `{table_id}`:")
        st.write(df.head())  # 显示前5行数据

        # 使用表格数据作为输入的勾选框
        use_table_as_input = st.checkbox("Use this table as input for Data as String")

        data_string = ""
        if use_table_as_input:
            data_string = df.head().to_string(index=False)
            st.info("Table data is being used as input. File upload is disabled.")
        else:
            # 文件上传功能，用于多模态输入
            st.subheader("Upload Files for Multi-modal Input")
            uploaded_files = st.file_uploader("Upload files (CSV, images, videos, text)", accept_multiple_files=True)
            for uploaded_file in uploaded_files:
                file_type = uploaded_file.type
                if "csv" in file_type:
                    try:
                        df_uploaded = pd.read_csv(uploaded_file)
                        st.write("Uploaded CSV Data:")
                        st.write(df_uploaded)
                        data_string += df_uploaded.to_string(index=False)
                    except Exception as e:
                        st.error(f"Error reading CSV file: {e}")
                elif "image" in file_type or "video" in file_type or "text" in file_type:
                    st.write(f"Uploaded {file_type.split('/')[0].capitalize()} File: {uploaded_file.name}")
                    # 根据需要处理多模态文件内容

        st.text_area("Data as String", data_string, height=200)

        # 提示模板输入功能 - 完全替换为数据分析和问题解答
        prompt_input = st.text_area("Enter your prompt template:", 
                                    value="You are an expert data analyst. Answer the following question based on the data provided:\n\n{data}\n\nQuestion: {question}",
                                    height=150)  # 调整对话框的高度

        # 初始化 Gemini 模型
        model_choice = st.sidebar.selectbox("Choose a Gemini model:", ["Gemini 1.5 Pro", "Gemini 1.5 Flash"])
        temperature = st.sidebar.slider("Temperature", 0.0, 1.0, 0.2, 0.05)
        max_tokens = st.sidebar.number_input("Max Tokens", min_value=10, max_value=3000, value=500, step=10)
        top_p = st.sidebar.slider("Top-p (nucleus sampling)", 0.0, 1.0, 0.9, 0.05)
        frequency_penalty = st.sidebar.slider("Frequency Penalty", 0.0, 1.0, 0.0, 0.1)
        presence_penalty = st.sidebar.slider("Presence Penalty", 0.0, 1.0, 0.0, 0.1)

        llm = ChatGoogleGenerativeAI(
            model=model_choice.lower().replace(" ", "-"),
            google_api_key=api_key,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty
        )

        prompt_template = PromptTemplate.from_template(prompt_input)
        llm_chain = LLMChain(llm=llm, prompt=prompt_template, verbose=True)

        # 用户提问
        user_question = st.text_input("Ask a question:", "")
        combined_input = {"data": data_string, "question": user_question}

        if st.button("Generate"):
            if user_question or data_string:
                try:
                    # 创建占位符用于流式输出
                    output_placeholder = st.empty()

                    # 生成完整响应
                    response = llm_chain.run(combined_input)

                    # 模拟流式输出
                    for i in range(0, len(response), 50):  # 每次输出50个字符
                        output_placeholder.text(response[:i+50])
                        time.sleep(0.1)  # 模拟延迟
                        
                    # 最终输出完整内容
                    output_placeholder.text(response)

                    save_conversation_to_firestore(db, user_id, combined_input, response)

                except Exception as e:
                    st.error(f"Failed to generate response: {e}")
            else:
                st.warning("Please enter a question or upload a file!")

        if st.button("Show Conversation History"):
            history = get_conversation_history(db, user_id)
            st.write(history)

    except Exception as e:
        st.error(f"An error occurred: {e}")

def save_conversation_to_firestore(db, user_id, user_input, bot_response):
    try:
        conversation_ref = db.collection('conversations').document()
        conversation_ref.set({
            'user_id': user_id,
            'user_input': user_input,
            'bot_response': bot_response
        })
    except Exception as e:
        st.error(f"Failed to save conversation: {e}")

def get_conversation_history(db, user_id):
    try:
        conversations = db.collection('conversations').where('user_id', '==', user_id).stream()
        history = []
        for conv in conversations:
            conv_dict = conv.to_dict()
            history.append(f"User: {conv_dict['user_input']}\nBot: {conv_dict['bot_response']}")
        return "\n\n".join(history)
    except Exception as e:
        return f"Failed to retrieve conversation history: {e}"


# 使用 Streamlit 的文件上传功能来上传凭证文件
uploaded_file = st.file_uploader("Upload your Google Cloud credentials JSON file")

if uploaded_file is not None:
    credentials_file_path = "credentials.json"
    with open(credentials_file_path, "wb") as f:
        f.write(uploaded_file.read())
    gemini_with_bigquery_page(db=None, api_key="your-google-api-key", credentials_file_path=credentials_file_path)
else:
    st.info("Please upload your Google Cloud credentials JSON file.")
