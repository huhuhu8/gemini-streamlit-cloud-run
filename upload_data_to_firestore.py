import streamlit as st

def upload_data_to_firestore(db):
    """将所有信息上传到 Firebase"""
    try:
        # 获取用户信息
        user_id = st.session_state.get("user", "anonymous")  # 如果没有用户信息，默认为 'anonymous'
        
        # 将 DataFrame 转换为字典列表
        model_results_dict = st.session_state.model_results.to_dict(orient="records")

        # 上传数据到 Firestore
        doc_ref = db.collection("project_insights").document()
        doc_ref.set({
            "user_id": user_id,
            "project_description": st.session_state.default_description,
            "eda_suggestions": st.session_state.eda_suggestions,
            "preprocessing_suggestions": st.session_state.preprocessing_suggestions,
            "model_results": model_results_dict,  # 转换后的字典列表
            "gemini_feedback": st.session_state.gemini_feedback,
            "memo_text": st.session_state.get("memo_text", "")
        })
        st.success("データが正常にクラウドにアップロードされました。")
    except Exception as e:
        st.error(f"クラウドへのアップロード中にエラーが発生しました: {e}")
