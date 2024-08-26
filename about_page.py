import streamlit as st

def about_page():
    # 设置页面标题和描述，并加入动画效果的CSS
    st.markdown(
        """
        <style>
        @keyframes fadeIn {
            0% { opacity: 0; transform: translateY(-20px); }
            100% { opacity: 1; transform: translateY(0); }
        }
        .big-font {
            font-size: 70px !important;
            text-align: center;
            color: #FF5733;
            animation: fadeIn 2s ease-in-out;
        }
        .small-font {
            font-size: 20px !important;
            text-align: center;
            color: #333;
        }
        </style>
        """, unsafe_allow_html=True
    )

    # 显示图片路径
    image_path = '/workspaces/lanngchain_streamlit_private_pleasecommit/22final2/images/PixPin_2024-08-23_11-50-16.png'
    
    # 使用 st.image() 显示图片
    st.image(image_path)

    # 显示带动画效果的大字体文本
    st.markdown('<p class="big-font">MADE BY SEIKU</p>', unsafe_allow_html=True)

    # 添加GitHub链接
    st.markdown(
        """
        <div style='text-align: center;'>
            <a href='https://github.com/huhuhu8' target='_blank'>
                <img src='https://github.githubassets.com/images/modules/logos_page/GitHub-Mark.png' width='40'>
            </a>
        </div>
        <p class="small-font">Visit my GitHub profile for more projects and code!</p>
        """, unsafe_allow_html=True
    )

# 调用页面函数
if __name__ == "__main__":
    about_page()
