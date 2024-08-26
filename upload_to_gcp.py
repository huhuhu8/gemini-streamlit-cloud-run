import streamlit as st
from google.cloud import storage
from google.oauth2 import service_account

def upload_to_gcp_page():
    st.title("Upload Files to GCP")

    # 读取 GCP credentials
    credentials = service_account.Credentials.from_service_account_file(
        '/workspaces/lanngchain_streamlit_private_pleasecommit/streamlit-gemini-ccf60-c9d613f7d829.json'
    )
    
    # 初始化 GCP Storage 客户端
    client = storage.Client(credentials=credentials)

    # 设置上传文件控件
    uploaded_file = st.file_uploader("Choose a file to upload", type=["csv", "txt", "xlsx", "jpg", "png", "pdf"])

    # 从 session state 中获取或初始化 bucket_name
    if "bucket_name" not in st.session_state:
        st.session_state.bucket_name = ""

    # 提示输入 GCP 存储桶名称，并使用 session state 作为默认值
    bucket_name = st.text_input("Enter the name of your GCP bucket", value=st.session_state.bucket_name)

    if st.button("Upload"):
        if bucket_name:
            try:
                # 保存 bucket_name 到 session state
                st.session_state.bucket_name = bucket_name

                # 获取 GCP 存储桶
                bucket = client.get_bucket(bucket_name)

                # 上传文件到指定存储桶
                blob = bucket.blob(uploaded_file.name)
                blob.upload_from_file(uploaded_file)

                st.success(f"File '{uploaded_file.name}' successfully uploaded to '{bucket_name}'!")
            except Exception as e:
                st.error(f"Error uploading file: {e}")
        else:
            st.warning("Please enter the bucket name.")

    # 显示 GCP 存储桶中的文件列表
    if bucket_name:
        try:
            bucket = client.get_bucket(bucket_name)
            blobs = bucket.list_blobs()

            st.subheader(f"Files in bucket '{bucket_name}':")
            files = [blob.name for blob in blobs]

            if files:
                st.write(files)
            else:
                st.write("No files found in this bucket.")
        except Exception as e:
            st.error(f"Error accessing bucket: {e}")

