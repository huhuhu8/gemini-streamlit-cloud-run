import os
import streamlit as st
from google.oauth2 import service_account
from google.cloud import firestore
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

# 環境変数を読み込む
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Firestoreの初期化
credentials_file_path = './streamlit-gemini-ccf60-c9d613f7d829.json'
credentials = service_account.Credentials.from_service_account_file(credentials_file_path)
db = firestore.Client(credentials=credentials)

# Ensure default_description is initialized in session state


def automl(db, api_key):

    if "default_description" not in st.session_state:
        st.session_state.default_description = (
            "私は工場エンジニアで、窒化アルミニウム（AIN）生産に関連するデータを持っています。"
            "このデータを使用して、目標変数（生産されたAIN粒子の密度）に対する予測モデルを構築したいと考えています。"
            "特征量はｍａｓｋｅｄされており、メタデータではありません。"
            "このプロジェクトでは、目標変数の分布をより正確に予測できる新しいアプローチを模索し、データの前処理、分析、およびモデリングを行います。"
        )
    st.title("AutoML プロジェクト概要入力")

    # プロジェクトの説明を表示して更新
    project_description = st.text_area("プロジェクトの説明を入力してください：", value=st.session_state.default_description, height=150)

    # デフォルト値でGeminiモデルを初期化
    model_choice = st.sidebar.selectbox("Geminiモデルを選択してください：", ["Gemini 1.5 Flash", "Gemini 1.5 Pro"], index=0)
    temperature = st.sidebar.slider("温度", 0.0, 1.0, 0.2, 0.05)
    max_tokens = st.sidebar.number_input("最大トークン数", min_value=10, max_value=3000, value=500, step=10)
    top_p = st.sidebar.slider("Top-p（nucleus sampling）", 0.0, 1.0, 0.4, 0.05)
    frequency_penalty = st.sidebar.slider("頻度ペナルティ", 0.0, 1.0, 0.0, 0.1)
    presence_penalty = st.sidebar.slider("出現ペナルティ", 0.0, 1.0, 0.0, 0.1)

    llm = ChatGoogleGenerativeAI(
        model=model_choice.lower().replace(" ", "-"),
        google_api_key=api_key,
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=top_p,
        frequency_penalty=frequency_penalty,
        presence_penalty=presence_penalty
    )

    # プロジェクトの説明を最適化
    if st.button("プロジェクトの説明を最適化"):
        prompt_template = PromptTemplate.from_template(
            "プロジェクト説明を簡潔で明確に最適化してください:\n\n{project_description}"
        )
        llm_chain = LLMChain(llm=llm, prompt=prompt_template, verbose=True)
        optimized_description = llm_chain.run({"project_description": project_description})

        # セッション状態に最適化された説明を更新
        st.session_state.default_description = optimized_description
        st.rerun()  # 更新された内容でテキストエリアを再描画

    # 最適化を受け入れてシステムプロンプトとして使用するオプション
    if st.button("最適化を受け入れてシステムプロンプトとして使用"):
        st.success("最適化された説明がシステムプロンプトとして選択されました。")

    # ユーザーが直接フィードバックを入力するための新しいテキストボックス
    improvement_points = st.text_area("説明をさらに洗練するためのフィードバックを入力してください：", height=100)

    # フィードバックを追加して再最適化するボタン
    if st.button("フィードバックを追加して再最適化"):
        refined_description = f"{st.session_state.default_description}\n\n追加のフィードバック：{improvement_points}"
        
        # フィードバックを追加した新しいプロンプトを作成
        refined_prompt_template = PromptTemplate.from_template(
            "あなたはプロジェクト管理の専門家です。追加されたフィードバックを用いて以下のプロジェクト説明をさらに最適化してください：\n\n{refined_description}"
        )
        refined_llm_chain = LLMChain(llm=llm, prompt=refined_prompt_template, verbose=True)
        reoptimized_description = refined_llm_chain.run({"refined_description": refined_description})
        
        # セッション状態に再最適化された説明を更新
        st.session_state.default_description = reoptimized_description
        st.rerun()  # 更新された内容でテキストエリアを再描画



if __name__ == "__main__":
    automl(db, GOOGLE_API_KEY)
