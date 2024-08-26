import streamlit as st

def introduction_page():
    st.markdown(
        """
        <style>
        .title-container {
            display: flex;
            align-items: center;
            justify-content: center;
            margin-top: -50px;
        }
        .title {
            font-size: 2.5em;
            font-weight: bold;
            color: #34495e;
            margin-right: 10px;
        }
        .section-title {
            font-size: 1.8em;
            color: #16a085;
            font-weight: bold;
            margin-top: 40px;
            text-align: left;
        }
        .section-content {
            font-size: 1.1em;
            margin-top: 15px;
            color: #2c3e50;
            line-height: 1.6;
        }
        .divider {
            border-top: 2px solid #bdc3c7;
            margin: 30px 0;
        }
        .animation {
            width: 50px;
            height: 50px;
            cursor: pointer;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # 标题和小人动画
    st.markdown(
        """
        <div class="title-container">
            <div class="title">😊AutoML イントロダクション </div>
            <img src="https://media.giphy.com/media/3o6ZtpxSZbQRRnwCKQ/giphy.gif" class="animation" title="クリックしてハイタッチ！" onclick="alert('ハイタッチ！データ探索の旅を始めましょう！')">
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    # AutoML工具的整体介绍
    st.markdown(
        """
        <div class="section-title">🚀 AutoMLツールについて</div>
        <div class="section-content">
        AutoMLは、初心者にも優しい自動化された機械学習ツールです。Geminiとの対話を通じて、データ分析やモデルの作成をサポートし、必要に応じて完全にGeminiの提案に従うこともできます。
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    # 页面内容
    st.markdown(
        """
        <div class="section-title">🌟 AutoML プロジェクト概要入力</div>
        <div class="section-content">
        <strong>機能:</strong> プロジェクトの紹介と最適化のためのインターフェースを提供します。<br>
        <strong>詳細:</strong> ユーザーはプロジェクト説明を入力または編集し、Gemini モデルを使用してそれらを最適化できます。
        </div>

        <div class="divider"></div>

        <div class="section-title">🔍 AutoML EDA インサイト</div>
        <div class="section-content">
        <strong>機能:</strong> プロジェクト説明および初期データに基づいた探索的データ分析（EDA）を提案します。<br>
        <strong>詳細:</strong> ユーザーはデータファイルをアップロードでき、システムが適切な EDA 手法を生成し、データプレビューと詳細情報を表示します。
        </div>

        <div class="divider"></div>

        <div class="section-title">🛠️ AutoML EDA 分析の実行</div>
        <div class="section-content">
        <strong>機能:</strong> EDA を実行し、ユーザーがデータをよりよく理解し、処理できるようにするためのデータ前処理の提案を生成します。<br>
        <strong>詳細:</strong> 選択された手法に基づいてグラフやレポートを生成し、詳細なデータ前処理の提案を提供します。
        </div>

        <div class="divider"></div>

        <div class="section-title">⚙️ AutoML データ処理</div>
        <div class="section-content">
        <strong>機能:</strong> データクリーニング、特徴エンジニアリング、標準化、正規化、およびデータ分割を行います。<br>
        <strong>詳細:</strong> モデリング段階に入る前に、データが適切にクリーニングおよび最適化されていることを確認します。
        </div>

        <div class="divider"></div>

        <div class="section-title">🧠 AutoML モデル作成</div>
        <div class="section-content">
        <strong>機能:</strong> 回帰モデルを選択およびトレーニングし、モデル結果を分析します。<br>
        <strong>詳細:</strong> ユーザーは特徴とターゲット変数を選択し、複数の回帰モデルをトレーニングし、パフォーマンスメトリックを比較し、SHAP グラフを生成して Gemini のフィードバックを得ることができます。
        </div>

        <div class="divider"></div>

        <div class="section-title">💡 Sidebar Popover機能</div>
        <div class="section-content">
        Sidebarの「Insights」セクションでは、メモを記録したり、プロジェクトの詳細情報を確認したり、Geminiの提案を管理したりできます。メモや分析結果をクラウドにアップロードして管理するには、ログインが必要です。
        </div>

        <div class="divider"></div>

        <div class="section-content" style="text-align: center; margin-top: 30px;">
        左側のナビゲーションバーを使用して、これらの機能を探索してください。このツールがあなたの機械学習の旅をより効率的で成功したものにすることを願っています！
        </div>
        """,
        unsafe_allow_html=True,
    )

# 调用页面函数
if __name__ == "__main__":
    introduction_page()
