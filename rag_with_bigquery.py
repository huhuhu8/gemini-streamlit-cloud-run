import pandas as pd
import streamlit as st
from google.cloud import bigquery
import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
import time
from google.oauth2 import service_account

def rag_with_bigquery_page(db, api_key, credentials_file_path):
    st.title("RAG with BigQuery")

    # 设置凭证文件路径
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = credentials_file_path

    # 设置项目 ID 和位置
    project_id = 'streamlit-gemini-ccf60'
    location = 'asia-northeast1'  # 指定数据集位置

    # 初始化 BigQuery 客户端，并指定位置
    credentials = service_account.Credentials.from_service_account_file(credentials_file_path)
    client = bigquery.Client(credentials=credentials, project=project_id, location=location)

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

        # 添加一个按钮来下载并打印此数据集中所有表的 schema
        all_schemas = []
        if st.button("Download and Print Schemas of All Tables in Dataset"):
            for table in tables:
                table_ref = client.dataset(dataset_id).table(table.table_id)
                schema = client.get_table(table_ref).schema
                schema_str = f"**Table: {table.table_id}**\n"
                for field in schema:
                    schema_str += f"* {field.name} ({field.field_type})\n"
                all_schemas.append(schema_str)
                st.write(schema_str)
            
            # 将所有表的 schema 合并为一个字符串，用作 RAG 的输入
            data_string = "\n".join(all_schemas)

        else:
            data_string = ""
        
        # 使用所有表的 schema 作为输入的勾选框
        use_schema_as_input = st.checkbox("Use schema of all tables as input for SQL generation")

        if use_schema_as_input:
            st.info("Schema of all tables is being used as input.")
            prompt_input = f"Schema:\n{data_string}\n\nQuestion: {{question}}"
        else:
            prompt_input = "Enter your retrieval and generation prompt template:"

        st.text_area("Data as String", data_string, height=200)

        # 提示模板输入功能 - 用于数据检索和生成
        prompt_template_input = st.text_area("Enter your retrieval and generation prompt template:", 
                                             value=prompt_input, height=150)  # 调整对话框的高度

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

        prompt_template = PromptTemplate.from_template(prompt_template_input)
        llm_chain = LLMChain(llm=llm, prompt=prompt_template, verbose=True)

        # 用户提问
        user_question = st.text_input("Ask a question for SQL generation:", "")
        combined_input = {"data": data_string, "question": user_question}

        # 添加按钮以打印当前 SQL 生成的 prompt
        if st.button("Print SQL Prompt"):
            if use_schema_as_input and user_question:
                sql_prompt = [
                    {'role': 'system', 'content': "You are a BigQuery SQL expert. Please help to generate a SQL query to answer the question. Your response should ONLY be based on the given context and follow the response guidelines and format instructions. ===Response Guidelines \n1. If the provided context is sufficient, please generate a valid SQL query without any explanations for the question. \n2. If the provided context is almost sufficient but requires knowledge of a specific string in a particular column, please generate an intermediate SQL query to find the distinct strings in that column. Prepend the query with a comment saying intermediate_sql \n3. If the provided context is insufficient, please explain why it can't be generated. \n4. Please use the most relevant table(s). \n5. If the question has been asked and answered before, please repeat the answer exactly as it was given before."},
                    {'role': 'user', 'content': f'Schema:\n{data_string}\n\nQuestion: {user_question}'}
                ]
                st.write(sql_prompt)
            else:
                st.warning("Please provide a question and ensure the schema is used as input.")

        # 生成 SQL 的按钮
        if st.button("Generate SQL based on Schema and Question"):
            if user_question and use_schema_as_input:
                try:
                    # 生成 SQL 响应
                    sql_response = llm_chain.run({"data": data_string, "question": user_question})
                    st.code(sql_response)

                    # 保存生成的 SQL 查询
                    st.session_state.generated_sql = sql_response

                except Exception as e:
                    st.error(f"Failed to generate SQL: {e}")
            else:
                st.warning("Please enter a question and ensure the schema is used as input!")

        # 执行生成的 SQL 查询按钮
        if 'generated_sql' in st.session_state and st.session_state.generated_sql:
            st.write("Generated SQL:")
            st.code(st.session_state.generated_sql)  # 打印生成的SQL查询

            if st.button("Execute Generated SQL"):
                try:
                    # 执行生成的 SQL 查询
                    query_job = client.query(st.session_state.generated_sql)
                    results = query_job.result().to_dataframe()
                    st.write("Query Results:")
                    st.write(results)

                    # 基于查询结果回答用户问题
                    result_description = f"The query returned {len(results)} rows."
                    if len(results) > 0:
                        result_description += f" Here are the first few rows:"
                        st.write(result_description)
                        st.write(results.head())
                    else:
                        st.write(result_description + " No data found.")

                except Exception as e:
                    st.error(f"Failed to execute SQL query: {e}")

        if st.button("Show RAG Conversation History"):
            history = get_conversation_history(db, user_id)
            st.write(history)

    except Exception as e:
        st.error(f"An error occurred: {e}")

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
