import streamlit as st
import firebase_admin
from firebase_admin import auth, credentials
import requests

# 初始化 Firebase 应用
if not firebase_admin._apps:
    cred = credentials.Certificate('/workspaces/lanngchain_streamlit_private_pleasecommit/streamlit-gemini-ccf60-c9d613f7d829.json')
    firebase_admin.initialize_app(cred)

def login_page(db):
    st.title("Login")

    # 如果用户已经登录
    if 'user' in st.session_state:
        st.success(f"Logged in as {st.session_state['user']}")
        return

    # 输入框用于传统的Email/Password登录
    email = st.text_input("Enter your email:")
    password = st.text_input("Enter your password:", type="password")

    if st.button("Login"):
        try:
            # 使用 Firebase Authentication 验证用户
            user = auth.get_user_by_email(email)
            # 验证密码
            if verify_password(email, password):
                st.session_state['user'] = user.email
                st.success("Login successful!")
            else:
                st.error("Invalid credentials. Please check your email and password.")
        except firebase_admin.exceptions.FirebaseError as e:
            st.error(f"Login failed! {e}")
        except Exception as e:
            st.error(f"Login failed! Error: {e}")

    # Google 登录回调处理
    if 'oauth_code' in st.query_params:
        oauth_code = st.query_params['oauth_code'][0]
        credentials = get_google_credentials(oauth_code)
        user_info = get_user_info(credentials)
        st.session_state['user'] = user_info['email']
        st.success(f"Logged in as {user_info['email']}")

def verify_password(email, password):
    api_key = "AIzaSyBUmHIbT4azEVKhAh5iyrlzDWXc0GKneYc"  # 将其替换为你的实际 API 密钥
    url = f"https://identitytoolkit.googleapis.com/v1/accounts:signInWithPassword?key={api_key}"
    payload = {
        "email": email,
        "password": password,
        "returnSecureToken": True
    }
    response = requests.post(url, json=payload)
    if response.status_code == 200:
        return True
    else:
        return False

def get_google_credentials(auth_code):
    flow = google.auth.transport.requests.Request()
    credentials = flow.fetch_token(code=auth_code)
    return credentials

def get_user_info(credentials):
    id_info = google.oauth2.id_token.verify_oauth2_token(credentials, Request())
    return id_info
