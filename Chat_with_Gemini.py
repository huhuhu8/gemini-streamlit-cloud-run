import pandas as pd
import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

def chat_with_gemini_page(db, api_key):
    st.title("Chat with Gemini")
    
    if 'user' not in st.session_state:
        st.warning("Please log in to access this page.")
        return
    
    user_id = st.session_state['user']

    # 添加模型选择器
    model_choice = st.sidebar.selectbox("Choose a Gemini model:", ["Gemini 1.5 Pro", "Gemini 1.5 Flash"])

    # 添加模型参数选项
    temperature = st.sidebar.slider("Temperature", 0.0, 1.0, 0.7, 0.05)
    max_tokens = st.sidebar.number_input("Max Tokens", min_value=10, max_value=1000, value=100, step=10)
    top_p = st.sidebar.slider("Top-p (nucleus sampling)", 0.0, 1.0, 0.9, 0.05)
    frequency_penalty = st.sidebar.slider("Frequency Penalty", 0.0, 1.0, 0.0, 0.1)
    presence_penalty = st.sidebar.slider("Presence Penalty", 0.0, 1.0, 0.0, 0.1)

    # 初始化模型
    llm = ChatGoogleGenerativeAI(
        model=model_choice.lower().replace(" ", "-"),
        google_api_key=api_key,
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=top_p,
        frequency_penalty=frequency_penalty,
        presence_penalty=presence_penalty
    )

    # 创建新的 LLM 链
    prompt_template = PromptTemplate.from_template("Analyze the following data and write a summary tweet about {topic}:")
    llm_chain = LLMChain(llm=llm, prompt=prompt_template, verbose=True)

    # 文件上传功能
    uploaded_file = st.file_uploader("Upload a CSV file", type="csv")
    data_string = ""
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.write("Uploaded CSV Data:")
            st.write(df)
            data_string = df.to_string(index=False)
            st.text_area("Data as String", data_string, height=200)
        except Exception as e:
            st.error(f"Error reading CSV file: {e}")

    topic = st.text_input("Enter a topic:", "")
    combined_input = f"{data_string}\nTopic: {topic}"

    if st.button("Generate"):
        if topic or data_string:
            try:
                response = llm_chain.run(combined_input)
                st.write(response)
                save_conversation_to_firestore(db, user_id, combined_input, response)
            except Exception as e:
                st.error(f"Failed to generate response: {e}")
        else:
            st.warning("Please enter a topic or upload a file!")

    if st.button("Show Conversation History"):
        history = get_conversation_history(db, user_id)
        st.write(history)

def save_conversation_to_firestore(db, user_id, user_input, bot_response):
    try:
        conversation_ref = db.collection('conversations').document()
        conversation_ref.set({
            'user_id': user_id,
            'user_input': user_input,
            'bot_response': bot_response
        })
    except Exception as e:
        st.error(f"Failed to save conversation: {e}")

def get_conversation_history(db, user_id):
    try:
        conversations = db.collection('conversations').where('user_id', '==', user_id).stream()
        history = []
        for conv in conversations:
            conv_dict = conv.to_dict()
            history.append(f"User: {conv_dict['user_input']}\nBot: {conv_dict['bot_response']}")
        return "\n\n".join(history)
    except Exception as e:
        return f"Failed to retrieve conversation history: {e}"
