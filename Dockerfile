# 使用官方的 Python 基础镜像
FROM python:3.9-slim

# 安装必要的系统依赖，包括 libgomp1
RUN apt-get update && apt-get install -y libgomp1 && apt-get clean

# 设置工作目录
WORKDIR /app

# 复制项目文件到容器中
COPY . /app

# 安装 Python 依赖
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# 设置环境变量，禁止 Streamlit 的 telemetry
ENV STREAMLIT_TELEMETRY=False

# 暴露端口
EXPOSE 8080

# 启动 Streamlit 应用
CMD ["streamlit", "run", "main.py", "--server.port=8080", "--server.address=0.0.0.0"]
