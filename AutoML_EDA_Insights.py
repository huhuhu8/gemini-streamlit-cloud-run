import os
import streamlit as st
import pandas as pd
import pyarrow.csv as pa_csv
import pyarrow as pa
from google.oauth2 import service_account
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from dotenv import load_dotenv
from io import StringIO

# 環境変数を読み込む
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Firestore の初期化（必要に応じて）
credentials_file_path = './streamlit-gemini-ccf60-c9d613f7d829.json'

credentials = service_account.Credentials.from_service_account_file(credentials_file_path)

# ローカルにアップロードされたファイルを保存するディレクトリ
UPLOAD_DIR = "./uploaded_files"
if not os.path.exists(UPLOAD_DIR):
    os.makedirs(UPLOAD_DIR)

def eda_analysis_page(db, GOOGLE_API_KEY, credentials_file_path):
    st.title("EDA インサイト")

    # ファイルアップロード機能を追加
    uploaded_file = st.file_uploader("CSV または Excel ファイルをアップロードしてください", type=["csv", "xlsx"])

    # ファイルを保存する処理
    if uploaded_file is not None:
        file_path = os.path.join(UPLOAD_DIR, uploaded_file.name)
        
        # ファイルを保存
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        st.success(f"ファイルがアップロードされ、保存されました: {uploaded_file.name}")

        # ファイルリストをセッションに保存
        if "uploaded_files" not in st.session_state:
            st.session_state.uploaded_files = []
        if uploaded_file.name not in st.session_state.uploaded_files:
            st.session_state.uploaded_files.append(uploaded_file.name)

    # 保存されたファイルのリストを表示し、選択させる
    if "uploaded_files" in st.session_state and st.session_state.uploaded_files:
        selected_file = st.selectbox("処理するファイルを選択してください", st.session_state.uploaded_files)

        # 選択されたファイルのパスを取得
        file_path = os.path.join(UPLOAD_DIR, selected_file)

        # ファイルを読み込む処理
        try:
            if selected_file.endswith(".csv"):
                table = pa_csv.read_csv(file_path)
                data = table.to_pandas()
            elif selected_file.endswith(".xlsx"):
                data = pd.read_excel(file_path)
                table = pa.Table.from_pandas(data)
                data = table.to_pandas()
            else:
                st.error("サポートされていないファイルタイプです！")
                return

            st.success(f"選択されたファイルを使用中: {selected_file}")
        except Exception as e:
            st.error(f"ファイルの読み込みエラー: {e}")
            return

        # 1. データセットの最初の10行と各列のデータ型を表示
        st.subheader("データセットの最初の10行と列のデータ型")
        first_10_rows = data.head(10).copy()
        first_10_rows.loc['Data Type'] = data.dtypes  # データ型を最後の行として追加
        st.dataframe(first_10_rows)

        # データ情報を追加で表示（行数と列数を含む）
        st.subheader("データセット情報")
        buffer = StringIO()
        data.info(buf=buffer)
        info_str = buffer.getvalue()
        st.text(info_str)

        # 2. プロジェクト説明を取得
        project_description = st.session_state.get("default_description", "")

        # プロジェクト説明とデータを結合
        description_with_data =f"{project_description}\n\nデータセットの最初の10行はこちらです:\n{data.head(10).to_string(index=False)}"

        # サイドバーに LLM パラメータ調整オプションを追加
        st.sidebar.title("LLM パラメータ")
        model_choice = st.sidebar.selectbox("Gemini モデル", ["Gemini 1.5 Flash", "Gemini 1.5 Pro"], index=1)
        temperature = st.sidebar.slider("温度 (Temperature)", 0.0, 1.0, 0.5, 0.05)
        max_tokens = st.sidebar.number_input("最大トークン数 (Max Tokens)", min_value=10, max_value=3000, value=500, step=10)
        top_p = st.sidebar.slider("Top-p (nucleus sampling)", 0.0, 1.0, 0.9, 0.05)
        frequency_penalty = st.sidebar.slider("頻度ペナルティ (Frequency Penalty)", 0.0, 1.0, 0.0, 0.1)
        presence_penalty = st.sidebar.slider("出現ペナルティ (Presence Penalty)", 0.0, 1.0, 0.0, 0.1)

        # 3. ボタンを追加してプロジェクト説明と最初の10行のデータに基づいたEDA分析方法を提案
        if st.button("プロジェクト説明とデータの最初の10行に基づいてEDA分析方法を提案"):
            llm = ChatGoogleGenerativeAI(
                model=model_choice.lower().replace(" ", "-"),
                google_api_key=GOOGLE_API_KEY,
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p,
                frequency_penalty=frequency_penalty,
                presence_penalty=presence_penalty
            )

            # Gemini との対話型プロンプトを作成
            prompt_template = PromptTemplate.from_template(
                "以下のプロジェクト説明とデータセットの最初の10行に基づいて、データがカテゴリ型か時系列型かを判断してください。データの種類を明確に述べた上で、"
                "このデータを分析するのに最も適した6つのEDA手法を推奨し、それぞれの手法がデータのどの側面を捉えるのに役立つかを簡潔に説明してください。"
                "推奨される手法は、データ分布のすべての側面（例えば、中心傾向、分散、相関性、パターンの有無、外れ値の特定など）を可能な限り網羅するものであるべきです。\n\n"
                "{description_with_data}\n\nデータの種類: [カテゴリ型/時系列型]\n\n推奨されるEDA手法:\n1. 手法1 - 理由\n2. 手法2 - 理由\n3. 手法3 - 理由\n4. 手法4 - 理由\n5. 手法5 - 理由\n6. 手法6 - 理由"
            )



            llm_chain = LLMChain(llm=llm, prompt=prompt_template, verbose=True)
            eda_suggestions = llm_chain.run({"description_with_data": description_with_data})

            # Gemini の分析提案を表示
            st.subheader("推奨されるEDA手法")
            st.write(eda_suggestions)

            # 更新 st.session_state 以便其他页面可以访问
            st.session_state.eda_suggestions = eda_suggestions  # 添加这一行以更新 st.session_state

            if st.button("結果をInsightsに保存する"):
                st.rerun()            
            
            # 提案を session_state に保存し、後で使用できるようにする
            #st.session_state.eda_suggestions = eda_suggestions.splitlines()


            # Save the project description and dataset info in session_state
            st.session_state['description_with_data'] = description_with_data
            st.session_state['dataset_info'] = info_str
            

            # Correct way to print the values stored in session_state
            st.header("入力したプロンプト")
            st.write("Description with Data:", st.session_state['description_with_data'])
            st.write("Dataset Info:", st.session_state['dataset_info'])
            
            

            


            

