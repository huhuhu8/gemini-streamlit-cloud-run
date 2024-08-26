import streamlit as st
import firebase_admin
from firebase_admin import auth, credentials, firestore

# 初始化 Firebase 应用
if not firebase_admin._apps:
    cred = credentials.Certificate('/workspaces/lanngchain_streamlit_private_pleasecommit/streamlit-gemini-ccf60-c9d613f7d829.json')
    firebase_admin.initialize_app(cred)

def register_page(db):
    st.title("Register")
    
    email = st.text_input("Enter your email:", key="register_email")
    password = st.text_input("Enter your password:", type="password", key="register_password")
    
    if st.button("Register"):
        try:
            # 使用 Firebase Authentication 注册用户
            user = auth.create_user(
                email=email,
                password=password
            )
            
            # 在 Firestore 中创建用户文档
            db.collection('users').document(user.uid).set({
                'email': email,
                'uid': user.uid
            })
            
            st.success("Registration successful! Please login.")
        except Exception as e:
            st.error(f"Registration failed! Error: {e}")

