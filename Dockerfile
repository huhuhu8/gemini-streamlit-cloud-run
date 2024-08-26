# 使用官方的 Python 3.9 slim 版本作为基础镜像
FROM python:3.9-slim

# 安装必要的系统依赖，包括 libgomp1 和其他可能需要的库
RUN apt-get update && \
    apt-get install -y libgomp1 && \
    apt-get install -y build-essential && \
    apt-get clean

# 设置工作目录
WORKDIR /app

# 将当前目录中的所有文件复制到容器的 /app 目录中
COPY . /app

# 升级 pip 并安装 Python 依赖
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# 设置环境变量，禁止 Streamlit 的 telemetry
ENV STREAMLIT_TELEMETRY=False

# 暴露应用将运行的端口
EXPOSE 8080

# 运行 Streamlit 应用
CMD ["streamlit", "run", "main.py", "--server.port=8080", "--server.address=0.0.0.0"]
