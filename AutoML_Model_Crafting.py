import os
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import shap
from streamlit_shap import st_shap
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

# 设置 matplotlib 显示字体为英文，避免乱码
plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['axes.unicode_minus'] = False

# 環境変数を読み込む
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

def model_page():
    st.title("モデル作成")

    # 使用相对路径读取数据
    data_path = "uploaded_files_copy/downloadable_data.csv"
    
    # 加载数据
    if os.path.exists(data_path):
        data = pd.read_csv(data_path)
    else:
        st.error("指定されたファイルが見つかりません。")
        return

    if data is not None:
        # 特徴量とターゲットの選択
        features = st.multiselect("特徴量を選択", options=data.columns.tolist(), default=data.columns.tolist()[:-1])
        target = st.selectbox("ターゲット変数を選択", options=data.columns.tolist(), index=len(data.columns) - 1)

        X = data[features]
        y = data[target]

        # データ分割 - 添加勾选框用于控制 shuffle
        test_size = st.slider("テストデータの割合を選択", 0.1, 0.5, 0.2)
        shuffle_data = st.checkbox("データをシャッフルする", value=True)
        
        # 使用 train_test_split 分割数据，依据勾选框是否选中控制 shuffle 功能
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42, shuffle=shuffle_data)

        # モデル辞書
        model_dict = {
            "Linear Regression": LinearRegression(),
            "Ridge Regression": Ridge(),
            "Lasso Regression": Lasso(),
            "Random Forest": RandomForestRegressor(),
            "Gradient Boosting": GradientBoostingRegressor(),
            "Support Vector Regression (SVR)": SVR(),
            "K-Nearest Neighbors (KNN)": KNeighborsRegressor(),
            "XGBoost": XGBRegressor(),
            "LightGBM": LGBMRegressor()
        }

        # 一键训练所有模型并打印结果
        if st.button("全てのモデルをトレーニングして結果を表示"):
            results = []
            model_predictions = {}  # 存储每个模型的预测结果
            for model_name, model in model_dict.items():
                # 根据模型类型进行缩放归一化
                if isinstance(model, (Ridge, Lasso, LinearRegression)):
                    scaler = StandardScaler()
                    X_train_scaled = scaler.fit_transform(X_train)
                    X_test_scaled = scaler.transform(X_test)
                    model.fit(X_train_scaled, y_train)
                    y_pred = model.predict(X_test_scaled)
                else:
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)

                # 计算模型评估指标
                r2 = r2_score(y_test, y_pred)
                rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                mae = mean_absolute_error(y_test, y_pred)

                # 存储结果
                results.append({
                    "モデル": model_name,
                    "MAE": mae,
                    "RMSE": rmse,
                    "R²": r2
                })
                
                # 存储预测结果
                model_predictions[model_name] = y_pred

            # トレーニング結果を表として表示
            results_df = pd.DataFrame(results)
            st.write("各モデルのトレーニング結果:")
            st.table(results_df.style.format({
                "MAE": "{:.4f}",
                "RMSE": "{:.4f}",
                "R²": "{:.4f}"
            }))

            # 将训练结果存储在 session_state 中以供 Gemini 使用
            st.session_state["model_results"] = results_df


            # 选取 RMSE 最低的三个模型
            top_3_models = results_df.nsmallest(3, 'RMSE')

            # 展示真实值与预测值对比的可视化图表
            st.subheader("RMSE 最低の3つのモデルの予測と実測値の比較")

            for index, row in top_3_models.iterrows():
                model_name = row['モデル']
                st.write(f"Model: {model_name}")

                # 绘制真实值与预测值对比图表
                y_pred = model_predictions[model_name]
                fig, ax = plt.subplots()
                ax.plot(y_test.values, label="Actual", color="blue")
                ax.plot(y_pred, label="Predicted", color="red")
                ax.set_title(f"{model_name} Prediction vs Actual")
                ax.legend()

                st.pyplot(fig)

            # SHAP分析：显示RMSE最低的模型的SHAP解释
            top_model_name = top_3_models.iloc[0]['モデル']

            st.subheader(f"{top_model_name} のSHAP解析")

            best_model = model_dict[top_model_name]
            if isinstance(best_model, (Ridge, Lasso, LinearRegression)):
                explainer = shap.Explainer(best_model, X_train_scaled)
                shap_values = explainer(X_test_scaled)
            else:
                explainer = shap.Explainer(best_model, X_train)
                shap_values = explainer(X_test)

            # 使用 matplotlib 创建一个自定义的图表大小
            plt.figure(figsize=(10, 8))  # 设置图表大小
            shap.summary_plot(shap_values, X_test, plot_type="dot", show=False)
            plt.tight_layout()
            st.pyplot(plt.gcf())  # 使用 plt.gcf() 获取当前的图表对象并显示


            # 绘制 SHAP Summary Plot，调整点大小和图表尺寸
            fig, ax = plt.subplots(figsize=(10, 8))  # 设置图表大小
            

            if st.button("結果をInsightsに保存する"):
                st.rerun()  

        # Gemini によるモデル提案ボタン
        if st.button("Geminiにfeedbackを依頼"):
            # 1. 获取项目描述和数据的最初10行
            project_description = st.session_state.get("default_description", "プロジェクトの説明がありません。")
            first_10_rows = data.head(10).copy()
            dataset_info = data.describe().to_string()

            # 2. 数据前処理の提案 (示例中的处理方法)
            data_processing_suggestions = """
            - 欠損値の処理: 平均値で補完
            - 特徴量のスケーリング: Min-Maxスケーリングを使用
            """

            # 3. 模型结果
            model_results = st.session_state.get("model_results", None)
            model_results_text = model_results.to_string() if model_results is not None else "モデルの結果がありません。"

            # 4. 生成 prompt
            prompt = f"""
            プロジェクト説明: {project_description}

            データの最初の10行:
            {first_10_rows}

            データセット情報:
            {dataset_info}

            データ前処理の提案:
            {data_processing_suggestions}

            全てのモデルのトレーニング結果:
            {model_results_text}

            評価:
            - あなたの評価はどうですか？
            - 改善するためには何ができますか？
            - お客様への発表方法を教えてください。
            """

            # 打印 prompt
            st.write("Gemini へのプロンプト:")
            st.code(prompt)

            # 5. 向 Gemini 发送请求并获取反馈
            model_choice = "Gemini 1.5 Pro"
            temperature = 0.5
            max_tokens = 1000
            top_p = 0.8

            llm = ChatGoogleGenerativeAI(
                model=model_choice.lower().replace(" ", "-"),
                google_api_key=GOOGLE_API_KEY,
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p
            )

            prompt_template = PromptTemplate.from_template("{prompt}")
            llm_chain = LLMChain(llm=llm, prompt=prompt_template, verbose=True)
            gemini_feedback = llm_chain.run({"prompt": prompt})

            # 显示 Gemini 的反馈
            st.subheader("Gemini のフィードバック")
            st.write(gemini_feedback)

            if st.button("結果をInsightsに保存する"):
                st.session_state.gemini_feedback=gemini_feedback
                st.write(st.session_state["gemini_feedback"])
                st.rerun() 

if __name__ == "__main__":
    model_page()
