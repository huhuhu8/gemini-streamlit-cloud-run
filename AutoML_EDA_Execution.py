import os
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf 
import numpy as np
from scipy.signal import find_peaks
from statsmodels.tsa.seasonal import seasonal_decompose



# データを保存するディレクトリのパス
UPLOAD_DIR = "./uploaded_files"

def eda_run_page(db, GOOGLE_API_KEY, credentials_file_path):
    st.title("EDA 分析の実行")

    # 以前にアップロードされたファイルを読み込む
    if "uploaded_files" in st.session_state and st.session_state.uploaded_files:
        selected_file = st.selectbox("処理するファイルを選択してください", st.session_state.uploaded_files)

        # ファイルのパスを取得してデータを読み込む
        file_path = os.path.join(UPLOAD_DIR, selected_file)
        if selected_file.endswith(".csv"):
            data = pd.read_csv(file_path)
        elif selected_file.endswith(".xlsx"):
            data = pd.read_excel(file_path)
        else:
            st.error("サポートされていないファイルタイプです！")
            return

        # 時系列データかバッチデータかを選択
        st.subheader("データタイプを選択してください")
        data_type = st.radio("データタイプ", ("時系列データ", "バッチデータ"))

        # サイドバーにマルチセレクトメニューを追加
        st.sidebar.title("EDA 分析選択")
        selected_methods = st.sidebar.multiselect("実行するEDA分析方法を選択してください", [
            "時系列図（Time Series Plot）",
            "移動平均線（Moving Average）",
            "差分（Differencing）",
            "自己相関図（ACF）",
            "偏自己相関図（PACF）",
            "季節性分解（Seasonal Decomposition）",
            "時系列分布図（Time Series Distribution Plot）",
            "移動標準偏差（Rolling Standard Deviation）",
            "周期図（Periodogram）",
            "ピークとトラフの検出（Peak and Trough Detection）",
            "ヒストグラム（Histogram）",
            "箱ひげ図（Box Plot）",
            "密度図（Density Plot）",
            "散布図（Scatter Plot）",
            "相関行列（Correlation Matrix）",
            "ペアプロット（Pair Plot）",
            "グループ統計（Group Statistics）",
            "⭐ 欠損値、外れ値、重複データ分析",
            "主成分分析（PCA）"
        ])

        # EDA分析結果を保存する変数
        eda_results = []
        
        numerical_columns = data.select_dtypes(include=['float64', 'int64']).columns
        categorical_columns = data.select_dtypes(include=['object']).columns

        # 時系列データ分析方法
        # 時系列図（Time Series Plot）
        if "時系列図（Time Series Plot）" in selected_methods:
            st.subheader("時系列図（Time Series Plot）")
            st.line_chart(data)

            # 時系列データの基本統計情報を計算
            mean_value = data.mean()
            std_value = data.std()
            trend_direction = "増加傾向" if data.iloc[-1].mean() > data.iloc[0].mean() else "減少傾向"

            eda_results.append(
                f"時系列図: データの基本的なトレンドを示しています。\n"
                f"- 平均値: {mean_value.values[0]:.2f}\n"
                f"- 標準偏差: {std_value.values[0]:.2f}\n"
                f"- トレンドの方向: {trend_direction}"
            )

        # 移動平均線（Moving Average）
        if "移動平均線（Moving Average）" in selected_methods:
            st.subheader("移動平均線（Moving Average）")
            window_size = st.slider("移動平均線のウィンドウサイズを選択してください", 1, 30, 7)
            moving_avg = data.rolling(window=window_size).mean()
            st.line_chart(moving_avg)

            # 平滑化されたデータの変化情報を提供
            avg_mean = moving_avg.mean()
            eda_results.append(
                f"移動平均線: ウィンドウサイズ {window_size} を使用してデータを平滑化しました。\n"
                f"- 平滑化後の平均値: {avg_mean.values[0]:.2f}\n"
                f"- 元のデータ平均値との変化: {avg_mean.values[0] - mean_value.values[0]:.2f}"
            )

        # 差分（Differencing）
        if "差分（Differencing）" in selected_methods:
            st.subheader("差分（Differencing）")
            differencing_order = st.slider("差分の次数を選択してください", 1, 3, 1)
            differenced_data = data.diff(periods=differencing_order).dropna()
            st.line_chart(differenced_data)

            diff_mean = differenced_data.mean()
            eda_results.append(
                f"差分: {differencing_order} 次の差分でデータのトレンドと季節性を除去しました。\n"
                f"- 差分後の平均値: {diff_mean.values[0]:.2f}\n"
                f"- 差分前後の平均値の変化: {diff_mean.values[0] - mean_value.values[0]:.2f}"
            )

        # 自己相関図（ACF）
        if "自己相関図（ACF）" in selected_methods:
            st.subheader("自己相関図（ACF）")
            fig, ax = plt.subplots()
            plot_acf(data.dropna(), ax=ax)
            st.pyplot(fig)

            # 有効な情報を抽出
            acf_values = plot_acf(data.dropna(), alpha=0.05).acf
            significant_lags = np.where(np.abs(acf_values) > 0.2)[0]
            eda_results.append(
                "自己相関図 (ACF): データの自己相関性を分析します。\n"
                f"- 顕著なラグ: {significant_lags}\n"
                f"- 最大自己相関値: {np.max(acf_values):.2f}"
            )

        # 偏自己相関図（PACF）
        if "偏自己相関図（PACF）" in selected_methods:
            st.subheader("偏自己相関図（PACF）")
            fig, ax = plt.subplots()
            plot_pacf(data.dropna(), ax=ax)
            st.pyplot(fig)

            # 有効な情報を抽出
            pacf_values = plot_pacf(data.dropna(), alpha=0.05).pacf
            significant_lags = np.where(np.abs(pacf_values) > 0.2)[0]
            eda_results.append(
                "偏自己相関図 (PACF): ARIMAモデルのパラメータを特定するために使用されます。\n"
                f"- 顕著なラグ: {significant_lags}\n"
                f"- 最大偏自己相関値: {np.max(pacf_values):.2f}"
            )

        # 季節性分解（Seasonal Decomposition）
        if "季節性分解（Seasonal Decomposition）" in selected_methods:
            st.subheader("季節性分解（Seasonal Decomposition）")
            decomposition = seasonal_decompose(data.dropna(), model='additive', period=12)
            st.write("トレンド部分")
            st.line_chart(decomposition.trend.dropna())
            st.write("季節性部分")
            st.line_chart(decomposition.seasonal.dropna())
            st.write("残差部分")
            st.line_chart(decomposition.resid.dropna())
            eda_results.append(
                "季節性分解: データをトレンド、季節性、および残差に分解しました。\n"
                "- トレンド: 長期的な変化を示しています。\n"
                "- 季節性: データの周期的な変動を示しています。\n"
                "- 残差: トレンドと季節性で説明できないランダムな変動を示しています。"
            )

        # 時系列分布図（Time Series Distribution Plot）
        if "時系列分布図（Time Series Distribution Plot）" in selected_methods:
            st.subheader("時系列分布図（Time Series Distribution Plot）")
            fig, ax = plt.subplots()
            sns.histplot(data.dropna(), kde=True, ax=ax)
            st.pyplot(fig)

            # 有効な情報を抽出
            data_mean = data.mean()
            data_std = data.std()
            eda_results.append(
                f"時系列分布図（Time Series Distribution Plot）:"
                f"- データの平均値: {data_mean.values}\n"
                f"- データの標準偏差: {data_std.values}"
            )

        # 移動標準偏差（Rolling Standard Deviation）
        if "移動標準偏差（Rolling Standard Deviation）" in selected_methods:
            st.subheader("移動標準偏差（Rolling Standard Deviation）")
            window_size = st.slider("移動標準偏差のウィンドウサイズを選択してください", 1, 30, 7)
            rolling_std = data.rolling(window=window_size).std()
            st.line_chart(rolling_std)

            avg_rolling_std = rolling_std.mean()
            eda_results.append(
                f"移動標準偏差: ウィンドウサイズ {window_size} を使用してデータの標準偏差を計算しました。\n"
                f"- 移動標準偏差の平均値: {avg_rolling_std.values[0]:.2f}"
            )

        # 周期図（Periodogram）
        if "周期図（Periodogram）" in selected_methods:
            st.subheader("周期図（Periodogram）")
            freqs = np.fft.fftfreq(len(data.dropna()))
            periodogram = np.abs(np.fft.fft(data.dropna()))**2
            fig, ax = plt.subplots()
            ax.plot(freqs, periodogram)
            ax.set_xlim([0, 0.5])
            ax.set_title("周期図（Periodogram）")
            st.pyplot(fig)

            # 有効な情報を抽出
            dominant_freq = freqs[np.argmax(periodogram)]
            eda_results.append(
                "周期図: 時系列データの周期性を分析します。\n"
                f"- 顕著な周波数: {dominant_freq:.2f}\n"
                f"- 対応する周期: {1/dominant_freq:.2f} 時間単位"
            )

        # ピークとトラフの検出（Peak and Trough Detection）
        if "ピークとトラフの検出（Peak and Trough Detection）" in selected_methods:
            st.subheader("ピークとトラフの検出（Peak and Trough Detection）")
            peaks, _ = find_peaks(data.squeeze())
            troughs, _ = find_peaks(-data.squeeze())
            fig, ax = plt.subplots()
            ax.plot(data, label="Data")
            ax.plot(data.index[peaks], data.iloc[peaks], "x", label="Peaks")
            ax.plot(data.index[troughs], data.iloc[troughs], "o", label="Troughs")
            ax.legend()
            st.pyplot(fig)
            eda_results.append(
                "ピークとトラフの検出: 時系列データ内の局所的なピークとトラフを識別してマークしました。\n"
                f"- ピーク数: {len(peaks)}\n"
                f"- トラフ数: {len(troughs)}"
            )

        # バッチデータ分析方法
        elif data_type == "バッチデータ":

            # ヒストグラム（Histogram）
            if "ヒストグラム（Histogram）" in selected_methods:
                st.subheader("ヒストグラム（Histogram）")
                for col in numerical_columns:
                    st.write(f"{col} のヒストグラム")
                    fig, ax = plt.subplots()
                    data[col].hist(ax=ax, bins=20)
                    st.pyplot(fig)

                    # 統計情報を追加
                    mean_val = data[col].mean()
                    median_val = data[col].median()
                    std_val = data[col].std()
                    eda_results.append(
                        f"{col} のヒストグラム:\n"
                        f"- 平均値: {mean_val:.2f}\n"
                        f"- 中央値: {median_val:.2f}\n"
                        f"- 標準偏差: {std_val:.2f}"
                    )

            # 箱ひげ図（Box Plot）
            if "箱ひげ図（Box Plot）" in selected_methods:
                st.subheader("箱ひげ図（Box Plot）")
                for col in numerical_columns:
                    st.write(f"{col} の箱ひげ図")
                    fig, ax = plt.subplots()
                    sns.boxplot(x=data[col], ax=ax)
                    st.pyplot(fig)

                    # 統計情報を追加
                    iqr = data[col].quantile(0.75) - data[col].quantile(0.25)
                    eda_results.append(
                        f"{col} の箱ひげ図:\n"
                        f"- 四分位範囲: {iqr:.2f}\n"
                        f"- 最大値: {data[col].max():.2f}\n"
                        f"- 最小値: {data[col].min():.2f}\n"
                        f"- 中央値: {data[col].median():.2f}"
                    )

            # 密度図（Density Plot）
            if "密度図（Density Plot）" in selected_methods:
                st.subheader("密度図（Density Plot）")
                for col in numerical_columns:
                    st.write(f"{col} の密度図")
                    fig, ax = plt.subplots()
                    sns.kdeplot(data[col], ax=ax)
                    st.pyplot(fig)

                    # 統計情報を追加
                    mode_val = data[col].mode().values[0] if not data[col].mode().empty else "N/A"
                    eda_results.append(
                        f"{col} の密度図:\n"
                        f"- 最頻値: {mode_val}\n"
                        f"- 平均値: {data[col].mean():.2f}\n"
                        f"- 標準偏差: {data[col].std():.2f}"
                    )

            # 散布図（Scatter Plot）
            if "散布図（Scatter Plot）" in selected_methods:
                st.subheader("散布図（Scatter Plot）")
                x_col = st.selectbox("X 軸の変数を選択してください", numerical_columns)
                y_col = st.selectbox("Y 軸の変数を選択してください", numerical_columns, index=1)
                st.write(f"{x_col} vs {y_col} の散布図")
                fig, ax = plt.subplots()
                ax.scatter(data[x_col], data[y_col])
                ax.set_xlabel(x_col)
                ax.set_ylabel(y_col)
                st.pyplot(fig)

                # 相関性情報を追加
                correlation = data[x_col].corr(data[y_col])
                eda_results.append(
                    f"{x_col} vs {y_col} の散布図:\n"
                    f"- 相関係数: {correlation:.2f}\n"
                    f"- X 軸の平均値: {data[x_col].mean():.2f}\n"
                    f"- Y 軸の平均値: {data[y_col].mean():.2f}"
                )

            # 相関行列（Correlation Matrix）
            if "相関行列（Correlation Matrix）" in selected_methods:
                st.subheader("相関行列（Correlation Matrix）")
                correlation_matrix = data[numerical_columns].corr()
                st.write("相関行列")
                st.dataframe(correlation_matrix)

                # ヒートマップを表示
                fig, ax = plt.subplots()
                sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
                st.pyplot(fig)

                # 有効な情報を抽出
                max_corr = correlation_matrix.abs().unstack().sort_values(ascending=False).drop_duplicates()
                eda_results.append(
                    "相関行列: 数値列間の相関性を示しています。\n"
                    f"- 最大相関係数: {max_corr.iloc[1]:.2f}、"
                    f"{max_corr.index[1][0]} と {max_corr.index[1][1]} の間で発生"
                )

            # ペアプロット（Pair Plot）
            if "ペアプロット（Pair Plot）" in selected_methods:
                st.subheader("ペアプロット（Pair Plot）")
                selected_pairplot_columns = st.multiselect("ペアプロットを描く変数を選択してください", numerical_columns, default=numerical_columns[:3])
                if len(selected_pairplot_columns) > 1:
                    fig = sns.pairplot(data[selected_pairplot_columns])
                    st.pyplot(fig)
                    eda_results.append(f"ペアプロット: {', '.join(selected_pairplot_columns)} 間のペア関係を示しました。")

            # グループ統計（Group Statistics）
            if "グループ統計（Group Statistics）" in selected_methods:
                st.subheader("グループ統計（Group Statistics）")
                if len(categorical_columns) > 0:
                    category_column = st.selectbox("グループカテゴリを選択してください", categorical_columns)
                    group_stats = data.groupby(category_column).mean()
                    st.write(f"{category_column} のグループ統計")
                    st.dataframe(group_stats)
                    eda_results.append(
                        f"{category_column} のグループ統計: 各グループの平均値およびその他の統計情報を示します。\n"
                        f"- グループ数: {data[category_column].nunique()}\n"
                        f"- 各グループのサンプル数: {data[category_column].value_counts().to_dict()}"
                    )
                else:
                    st.warning("グループ統計を行うためのカテゴリ列が利用できません。")

            # ⭐ 欠損値、外れ値、重複データ分析（Missing Value Analysis）
            if "⭐ 欠損値、外れ値、重複データ分析" in selected_methods:
                st.subheader("⭐ 欠損値、外れ値、重複データ分析")

                # 欠損値分析
                missing_values = data.isnull().sum()
                missing_values = missing_values[missing_values > 0]

                if not missing_values.empty:
                    st.write("データセット内の欠損値")
                    st.dataframe(missing_values)
                    missing_percentage = (missing_values / len(data)) * 100
                    eda_results.append(
                        "欠損値分析: データセット内の欠損値の分布と数を示します。\n"
                        f"- 総欠損値: {missing_values.sum()}\n"
                        f"- 欠損値の割合: {missing_percentage.to_dict()}"
                    )
                else:
                    st.write("データセット内に欠損値はありません")
                    eda_results.append("欠損値分析: データセット内に欠損値はありません。")

                # 外れ値分析
                st.write("外れ値分析")
                for col in numerical_columns:
                    if len(data[col].unique()) > 10:  # 数値列のみを対象に外れ値分析を行う
                        q1 = data[col].quantile(0.25)
                        q3 = data[col].quantile(0.75)
                        iqr = q3 - q1
                        lower_bound = q1 - 1.5 * iqr
                        upper_bound = q3 + 1.5 * iqr
                        outliers = data[(data[col] < lower_bound) | (data[col] > upper_bound)]
                        num_outliers = len(outliers)
                        if num_outliers > 0:
                            st.write(f"{col} 列で {num_outliers} 個の外れ値が見つかりました")
                            eda_results.append(
                                f"外れ値分析: {col} 列で {num_outliers} 個の外れ値が見つかりました。\n"
                                f"下限値: {lower_bound}, 上限値: {upper_bound}。\n"
                            )
                        else:
                            st.write(f"{col} 列に外れ値はありません")
                            eda_results.append(f"外れ値分析: {col} 列に外れ値はありません。\n")

                # 重複データ分析
                st.write("重複データ分析")
                duplicate_rows = data.duplicated().sum()
                if duplicate_rows > 0:
                    st.write(f"データセット内に {duplicate_rows} 行の重複データが見つかりました")
                    eda_results.append(f"重複データ分析: データセット内に {duplicate_rows} 行の重複データが見つかりました。")
                else:
                    st.write("データセット内に重複データはありません")
                    eda_results.append("重複データ分析: データセット内に重複データはありません。")

            # 主成分分析（PCA）
            if "主成分分析（PCA）" in selected_methods:
                st.subheader("主成分分析（PCA）")
                st.write("主成分分析（PCA）")
                pca = PCA(n_components=2)
                pca_result = pca.fit_transform(data[numerical_columns].dropna())
                fig, ax = plt.subplots()
                ax.scatter(pca_result[:, 0], pca_result[:, 1])
                ax.set_xlabel("主成分 1")
                ax.set_ylabel("主成分 2")
                st.pyplot(fig)

                explained_variance = pca.explained_variance_ratio_
                eda_results.append(
                    "主成分分析: データを2つの主成分に次元削減し、主成分間の関係を表示します。\n"
                    f"- 主成分 1 が説明する分散割合: {explained_variance[0]:.2f}\n"
                    f"- 主成分 2 が説明する分散割合: {explained_variance[1]:.2f}"
                )

        # サイドバーのボタン: 上記のEDA分析結果をGeminiに導入し、データ前処理の提案を行います
        # サイドバーのボタン: 上記のEDA分析結果をGeminiに導入し、データ前処理の提案を行います
        if st.sidebar.button("上記のEDA分析結果をGeminiに導入してデータ前処理の提案を受ける"):

            # session_stateにEDA分析結果を保存する
            if "eda_results" not in st.session_state:
                st.session_state.eda_results = []
            
            # 現在のEDA結果をsession_stateに保存
            st.session_state.eda_results = eda_results
            
            # 結果をレポートとしてまとめる
            eda_report = "\n\n".join(st.session_state.eda_results)
            st.text_area("EDA 分析レポート", eda_report, height=200)

            # Step 1: 先把 eda_report 单独输入 Gemini 进行整理
            st.subheader("Gemini による EDA 分析結果の整理")
            llm = ChatGoogleGenerativeAI(
                model="gemini-1.5-pro",
                google_api_key=GOOGLE_API_KEY,
                temperature=0.5,
                max_tokens=1000,
                top_p=0.9,
                frequency_penalty=0,
                presence_penalty=0
            )

            # Prompt to organize the EDA report
            eda_report_template = PromptTemplate.from_template(
                "以下は初期のEDA分析結果です。すべての有効な情報を整理し、漏れがないように簡潔でわかりやすい形式で出力してください。内容が論理的な順序に沿って整然と並べられるように注意してください。\n\nEDA分析結果:\n{eda_report}"
            )

            # 组织 EDA 报告的 prompt
            llm_chain_report = LLMChain(llm=llm, prompt=eda_report_template, verbose=True)
            organized_eda_report = llm_chain_report.run({"eda_report": eda_report})

            # 打印整理后的 EDA 分析报告
            st.text_area("整理されたEDA 分析レポート", organized_eda_report, height=200)

            # 将整理后的 EDA 报告保存到 session_state 中
            st.session_state.eda_report = organized_eda_report

            # Step 2: 使用整理后的 EDA 报告生成新的 prompt
            prompt_template = PromptTemplate.from_template(
                "以下はプロジェクトの説明、データセット情報、および整理されたEDA分析結果です。"
                "これに基づいて、包括的なデータ前処理の提案をお願いします。以下のポイントを最低限含めてくださいが、"
                "それ以外にも適切な前処理の提案があれば追加してください:\n\n"
                "1. 異常値処理: 異常値を検出するための具体的な閾値の設定。\n"
                "2. 欠損値処理: 欠損値を均値、中央値、もしくは他の方法で補完するための最適なアプローチ。\n"
                "3. 特徴エンジニアリング: 無関係または冗長な特徴量の削除、または新たな特徴量の作成に関する提案。\n"
                "4. 特徴量の変換: 対数変換、平方根変換、Box-Cox変換など、特定の特徴量に対して適用すべき変換の提案。\n"
                "5. 特徴交互作用: 有意な特徴量間の交互作用を提案し、それを追加することで予測モデルの性能が向上するかどうかの見解。\n"
                "6. 標準化または正規化: データの標準化や正規化が必要かどうかの判断とその実施方法。\n"
                "7. その他: 上記に含まれていないが、データの品質やモデル性能向上のために重要な前処理についての追加提案。\n\n"
                "プロジェクト説明:\n{description_with_data}\n\n"
                "データセット情報:\n{dataset_info}\n\n"
                "EDA分析結果:\n{eda_report}"
            )


            # 使用新的 prompt 进行最终请求
            llm_chain = LLMChain(llm=llm, prompt=prompt_template, verbose=True)
            preprocessing_suggestions = llm_chain.run({
                "description_with_data": st.session_state.get("description_with_data", ""),
                "dataset_info": st.session_state.get("dataset_info", ""),
                "eda_report": organized_eda_report  # 使用整理后的 EDA 报告
            })

            # デバッグ用に生成されたプロンプトを表示
            st.subheader("生成されたプロンプトの内容")
            generated_prompt = prompt_template.format(
                description_with_data=st.session_state.get("description_with_data", ""),
                dataset_info=st.session_state.get("dataset_info", ""),
                eda_report=organized_eda_report
            )
            st.text_area("Prompt Content", generated_prompt, height=300)

            st.session_state.preprocessing_suggestions = preprocessing_suggestions

           

            # Geminiの提案を表示
            st.subheader("データ前処理の提案")
            st.write(preprocessing_suggestions)
            
            # 生成并存储数据前处理的提案
            if st.button("結果をInsightsに保存する"):
                st.session_state.gemini_feedback=preprocessing_suggestions
                st.rerun()
