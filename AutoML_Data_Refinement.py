import os
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
import numpy as np

# 分析データのパスをインポート
UPLOAD_DIR = "./uploaded_files"
COPY_DIR = "./uploaded_files_copy"

def preprocess_page(db, GOOGLE_API_KEY, credentials_file_path):
    st.title("データ前処理")

    # データをロード
    data = load_data()
    if data is None:
        return

    if st.button("副本を作成/削除（再度クリックで元の副本を削除）"):
        if os.path.exists(COPY_DIR):
            # 副本が存在する場合、削除
            for file in os.listdir(COPY_DIR):
                os.remove(os.path.join(COPY_DIR, file))
            os.rmdir(COPY_DIR)
            st.success("以前の副本が削除されました。")
        else:
            # 新しい副本を作成
            os.makedirs(COPY_DIR, exist_ok=True)
            selected_file = st.session_state["selected_file"]
            copy_file_path = os.path.join(COPY_DIR, os.path.basename(selected_file))
            data.to_csv(copy_file_path, index=False)
            st.success(f"ファイルの副本が作成されました：{copy_file_path}")

    # 副本が存在するか確認し、存在する場合はそれを使用
    data = load_copy_data() if os.path.exists(COPY_DIR) else data

    # `current_tab`をセッションステートに初期化
    if 'current_tab' not in st.session_state:
        st.session_state['current_tab'] = 'プレビュー'
        
    # タブを作成
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["プレビュー", "クリーニング", "特徴エンジニアリング", "標準化と正規化", "データの分割", "保存とエクスポート"])

    # Sidebarの設定項目をタブに応じて表示
    with st.sidebar:
        st.subheader("クリーニングの設定")
        threshold = st.slider(
            "異常値のしきい値を設定", 
            min_value=0.0,  # 合理的最小值
            max_value=3.0,  # 合理的最大值
            value=1.5,      # 默认值设为中间值
            step=0.1,       # 步进值设为 0.1
            key="threshold_slider1"  # 指定唯一的 key
        )
        
        fill_option = st.selectbox(
            "欠損値の補完方法を選択", 
            ["補完しない", "平均値で補完", "中央値で補完", "前方補完", "後方補完"],
            key="fill_option_selectbox"  # 指定唯一的 key
        )

        st.subheader("標準化と正規化の設定")
        scale_option = st.selectbox(
            "標準化方法を選択", 
            ["Z-score 標準化"],
            key="scale_option_selectbox"  # 指定唯一的 key
        )
        
        range_min = st.slider(
            "正規化の最小値を選択", 
            0.0, 1.0, 0.0,
            key="range_min_slider"  # 指定唯一的 key
        )
        
        range_max = st.slider(
            "正規化の最大値を選択", 
            0.0, 1.0, 1.0,
            key="range_max_slider"  # 指定唯一的 key
        )

    with tab1:
        st.header("プレビュー")
        st.write("データのプレビュー:")
        st.dataframe(data.head(10))
        st.session_state['current_tab'] = 'プレビュー'

    with tab2:
        st.header("クリーニング")
        st.session_state['current_tab'] = 'クリーニング'

        # 缺损值的显示与填补
        st.subheader("欠損値の表示と補完")

        if st.button("欠損値を補完"):
            if fill_option == "補完しない":
                st.warning("欠損値は補完されていません。")
            else:
                for col in data.columns:
                    if data[col].isnull().sum() > 0:  # 如果列中存在缺失值
                        if fill_option == "平均値で補完":
                            imputer = SimpleImputer(strategy='mean')
                        elif fill_option == "中央値で補完":
                            imputer = SimpleImputer(strategy='median')
                        elif fill_option == "前方補完":
                            data.fillna(method='ffill', inplace=True)
                            continue
                        elif fill_option == "後方補完":
                            data.fillna(method='bfill', inplace=True)
                            continue
                        
                        # 对数值列应用插补策略
                        if fill_option in ["平均値で補完", "中央値で補完"]:
                            data[col] = imputer.fit_transform(data[[col]])

                # 保存填补后的数据到副本文件
                save_path = os.path.join(COPY_DIR, os.path.basename(st.session_state["selected_file"]))
                data.to_csv(save_path, index=False)
            
                st.success(f"全ての欠損値が{fill_option}で補完され、データが保存されました。")

        # 異常値と欠損値処理を一括実行
        st.subheader("異常値と欠損値の処理")

        # 提醒用户不要轻易删除异常值
        st.warning("異常値の削除は慎重に行ってください。過度の削除はデータの損失につながる可能性があります。")

        # 选择缺失值的填补方法
        fill_option = st.sidebar.selectbox(
            "欠損値の補完方法を選択", 
            ["補完しない", "平均値で補完", "中央値で補完", "前方補完", "後方補完"],
            key="fill_option_tab2"  # 指定唯一的 key
        )

        threshold = st.sidebar.slider(
            "異常値のしきい値を設定", 
            min_value=0.0,  # 合理的最小值
            max_value=3.0,  # 合理的最大值
            value=1.5,      # 默认值设为中间值
            step=0.1,       # 步进值设为 0.1
            key="threshold_slider2"  # 指定唯一的 key
        )

        # 执行按钮
        if st.button("異常値を削除し、欠損値を補完"):
            # 清除异常值
            for col in data.select_dtypes(include=['float64', 'int64']).columns:
                Q1 = data[col].quantile(0.25)
                Q3 = data[col].quantile(0.75)
                IQR = Q3 - Q1
                
                # 使用用户指定的阈值，而不是固定值
                threshold_value = st.session_state['threshold_slider']  # 从用户设置中获取阈值
                
                # 打印每列被删除的数据比例
                before_len = len(data)
                data = data[~((data[col] < (Q1 - threshold_value * IQR)) | (data[col] > (Q3 + threshold_value * IQR)))]
                after_len = len(data)
                st.write(f"{col} 列中删除了 {(before_len - after_len) / before_len * 100:.2f}% のデータ")

            # 填补缺失值
            if fill_option != "補完しない":
                for col in data.columns:
                    if data[col].isnull().sum() > 0:  # 如果列中存在缺失值
                        if fill_option == "平均値で補完":
                            imputer = SimpleImputer(strategy='mean')
                        elif fill_option == "中央値で補完":
                            imputer = SimpleImputer(strategy='median')
                        elif fill_option == "前方補完":
                            data.fillna(method='ffill', inplace=True)
                            continue
                        elif fill_option == "後方補完":
                            data.fillna(method='bfill', inplace=True)
                            continue
                        
                        # 对数值列应用插补策略
                        if fill_option in ["平均値で補完", "中央値で補完"]:
                            data[col] = imputer.fit_transform(data[[col]])

                st.success(f"{fill_option}が完了しました。")

            # 保存修改后的数据到副本文件
            save_path = os.path.join(COPY_DIR, os.path.basename(st.session_state["selected_file"]))
            data.to_csv(save_path, index=False)
            st.success("全ての異常値が削除され、欠損値が補完され、副本ファイルに保存されました。")

            # 重新读取副本文件以更新数据状态
            data = load_copy_data()




    with tab3:
        st.header("特徴エンジニアリング")
        st.session_state['current_tab'] = '特徴エンジニアリング'

        # 无关特征删除
        st.subheader("無関係な特徴量の削除")

        # 自动检测分散为0的特征
        low_variance_features = data.columns[data.var() == 0].tolist()

        # 设置默认值，确保更换文件时不会出错
        default_irrelevant_features = ['data18', 'data22', 'data40', 'data42', 'LotNO']
        irrelevant_features = st.multiselect(
            "削除する無関係な特徴量を選択してください", 
            options=data.columns.tolist(), 
            default=[f for f in default_irrelevant_features if f in data.columns] + low_variance_features
        )

        if st.button("無関係な特徴量を削除"):
            before_cols = data.columns.tolist()
            data = data.drop(columns=irrelevant_features, errors='ignore')
            after_cols = data.columns.tolist()
            removed_features = set(before_cols) - set(after_cols)
            st.success(f"次の無関係な特徴量が削除されました: {', '.join(removed_features)}")

            # 保存修改后的数据到副本文件
            save_path = os.path.join(COPY_DIR, os.path.basename(st.session_state["selected_file"]))
            data.to_csv(save_path, index=False)
            st.success(f"無関係な特徴量を削除後のデータが保存されました：{save_path}")

        # 特征变换
        st.subheader("特徴量の変換")

        # 设置默认值
        default_skewed_features = ['data2', 'data5', 'data6', 'data7', 'data12', 'data26', 'data36', 'data41', 'data43']
        skewed_features = st.multiselect(
            "変換する特徴量を選択してください", 
            options=data.columns.tolist(), 
            default=[f for f in default_skewed_features if f in data.columns]
        )
        transform_option = st.selectbox("変換方法を選択", ["対数変換", "平方根変換", "Box-Cox変換"])

        if st.button("特徴量の変換を実行"):
            for feature in skewed_features:
                if feature in data.columns:
                    if transform_option == "対数変換":
                        data[feature] = np.log1p(data[feature])
                    elif transform_option == "平方根変換":
                        data[feature] = np.sqrt(data[feature])
                    elif transform_option == "Box-Cox変換":
                        data[feature], _ = stats.boxcox(data[feature] + 1)  # Box-Cox变换需要正值

            st.success(f"{transform_option}が選択された特徴量に適用されました。")

            # 保存修改后的数据到副本文件
            save_path = os.path.join(COPY_DIR, os.path.basename(st.session_state["selected_file"]))
            data.to_csv(save_path, index=False)
            st.success(f"特徴量の変換後のデータが保存されました：{save_path}")

        # 特征交互作用
        st.subheader("特徴交互作用")
        features = data.columns.tolist()  # `features` は既に定義された列リストと仮定
        if len(features) > 1:  # 少なくとも2つの特徴が选択できることを确认
            feature_1 = st.selectbox("最初の特徴を選択", options=features, key="interaction_feature_1")
            feature_2 = st.selectbox("2つ目の特徴を選択", options=features, index=1, key="interaction_feature_2")
            if st.button(f"{feature_1} と {feature_2} の交互作用を作成"):
                interaction_term = data[feature_1] * data[feature_2]
                interaction_column_name = f"{feature_1}*{feature_2}"
                data[interaction_column_name] = interaction_term
                st.success(f"{feature_1} と {feature_2} の交互作用 '{interaction_column_name}' が作成されました。")
                st.write(data[[feature_1, feature_2, interaction_column_name]].head())

                # 保存修改后的数据到副本文件
                save_path = os.path.join(COPY_DIR, os.path.basename(st.session_state["selected_file"]))
                data.to_csv(save_path, index=False)
                st.success(f"交互作用後のデータが保存されました：{save_path}")
        else:
            st.warning("交互作用を作成するには、少なくとも2つの特徴を選択してください。")


    with tab4:
        st.header("標準化と正規化")
        st.session_state['current_tab'] = '標準化と正規化'

        # 標準化と正規化の説明
        st.markdown("""
        ### 標準化 (Standardization)
        標準化とは、データの平均を0、分散を1にするスケーリング手法です。これにより、異なるスケールの特徴量を統一し、モデルの学習を安定させることができます。
        
        ### 正規化 (Normalization)
        正規化は、データを指定した範囲 (通常は0から1) にスケーリングする方法です。これは、特徴量の値を一定の範囲に収めることで、モデルが特定の特徴量に過度に依存するのを防ぐために使用されます。
        """)

        # 標準化
        st.subheader("標準化")
        if st.button("標準化を実行"):
            scaler = StandardScaler()
            numerical_columns = data.select_dtypes(include=['float64', 'int64']).columns
            data[numerical_columns] = scaler.fit_transform(data[numerical_columns])
            st.success("標準化が適用されました")

            # 標準化後のデータ統計情報を表示
            st.write("標準化後のデータ統計:")
            st.dataframe(data.describe())

            # 保存副本文件
            save_path = os.path.join(COPY_DIR, os.path.basename(st.session_state["selected_file"]))
            data.to_csv(save_path, index=False)
            st.success(f"標準化後のデータが保存されました: {save_path}")

        # 正規化
        st.subheader("正規化")
        range_min = st.slider("正規化の最小値を選択", 0.0, 1.0, 0.0)
        range_max = st.slider("正規化の最大値を選択", 0.0, 1.0, 1.0)
        
        if st.button("正規化を実行"):
            scaler = MinMaxScaler(feature_range=(range_min, range_max))
            numerical_columns = data.select_dtypes(include=['float64', 'int64']).columns
            data[numerical_columns] = scaler.fit_transform(data[numerical_columns])
            st.success("正規化が適用されました")

            # 正規化後のデータ統計情報を表示
            st.write("正規化後のデータ統計:")
            st.dataframe(data.describe())

            # 保存副本文件
            save_path = os.path.join(COPY_DIR, os.path.basename(st.session_state["selected_file"]))
            data.to_csv(save_path, index=False)
            st.success(f"正規化後のデータが保存されました: {save_path}")

    with tab5:
        st.header("データ分割")
        st.session_state['current_tab'] = 'データの分割'

        # データ分割
        test_size = st.slider("テストセットの割合を選択", min_value=0.1, max_value=0.5, value=0.2)
        if st.button("データ分割を実行"):
            X = data.drop(columns=[target_variable])
            y = data[target_variable]
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)
            st.success(f"トレーニングデータのサンプル数: {len(X_train)}, テストデータのサンプル数: {len(X_test)}")

    with tab6:
        st.header("保存とエクスポート")
        st.session_state['current_tab'] = '保存とエクスポート'

        # 処理後のデータを保存
        st.subheader("処理後のデータを保存")
        if st.button("データを保存"):
            save_path = os.path.join(COPY_DIR, "processed_data.csv")
            data.to_csv(save_path, index=False)
            st.success(f"データが保存されました: {save_path}")
            st.write(f"保存先のパス: {save_path}")

            # ファイルダウンロードボタンを追加
            with open(save_path, "rb") as file:
                st.download_button(
                    label="処理後のデータをダウンロード",
                    data=file,
                    file_name="processed_data.csv",
                    mime="text/csv"
                )

def load_data():
    # アップロードされたデータファイルを読み込む
    if "uploaded_files" in st.session_state and st.session_state.uploaded_files:
        selected_file = st.selectbox("処理するファイルを選択してください", st.session_state.uploaded_files)
        file_path = os.path.join(UPLOAD_DIR, selected_file)
        st.session_state["selected_file"] = file_path  # 選択したファイルをセッションに保存
        if selected_file.endswith(".csv"):
            return pd.read_csv(file_path)
        elif selected_file.endswith(".xlsx"):
            return pd.read_excel(file_path)
    return None

def load_copy_data():
    # 副本を読み込む
    if "selected_file" in st.session_state:
        copy_file_path = os.path.join(COPY_DIR, os.path.basename(st.session_state["selected_file"]))
        if os.path.exists(copy_file_path):
            return pd.read_csv(copy_file_path)
    return None

def main():
    st.sidebar.title("データ前処理アプリ")
    db = None  # データベース関連のオブジェクト (必要に応じて)
    GOOGLE_API_KEY = "YOUR_API_KEY"
    credentials_file_path = "path_to_credentials_file"

    preprocess_page(db, GOOGLE_API_KEY, credentials_file_path)

if __name__ == "__main__":
    main()
       
