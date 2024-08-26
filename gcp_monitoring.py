import streamlit as st
from google.cloud import monitoring_v3, bigquery, storage
from google.oauth2 import service_account

def gcp_monitoring_page():
    st.title("Google Cloud Monitoring")

    # 读取 GCP credentials
    credentials = service_account.Credentials.from_service_account_file(
        './streamlit-gemini-ccf60-c9d613f7d829.json'
    )
    
    # 初始化 Monitoring 客户端
    monitoring_client = monitoring_v3.MetricServiceClient(credentials=credentials)
    project_id = "your-project-id"  # 替换为你的项目 ID
    project_name = f"projects/{project_id}"

    # 显示 BigQuery 使用情况
    st.subheader("BigQuery Usage")

    bigquery_client = bigquery.Client(credentials=credentials)
    datasets = list(bigquery_client.list_datasets())
    
    if datasets:
        dataset_names = [dataset.dataset_id for dataset in datasets]
        st.write(f"Datasets in BigQuery: {', '.join(dataset_names)}")
    else:
        st.write("No datasets found in BigQuery.")
    
    # 显示 Cloud Storage 使用情况
    st.subheader("Cloud Storage Usage")

    storage_client = storage.Client(credentials=credentials)
    buckets = list(storage_client.list_buckets())
    
    if buckets:
        bucket_names = [bucket.name for bucket in buckets]
        st.write(f"Buckets in Cloud Storage: {', '.join(bucket_names)}")
    else:
        st.write("No buckets found in Cloud Storage.")
    
    # 获取 GCP Metric 列表
    st.subheader("Available Metrics")
    try:
        metrics = monitoring_client.list_metric_descriptors(name=project_name)
        metric_names = [metric.type for metric in metrics]
        selected_metric = st.selectbox("Select a Metric", metric_names)

        # 查询指定 Metric 的数据
        if selected_metric:
            interval = monitoring_v3.TimeInterval(
                {
                    "end_time": {"seconds": int(st.time.time())},
                    "start_time": {"seconds": int(st.time.time()) - 3600},  # 最近1小时的数据
                }
            )

            results = monitoring_client.list_time_series(
                request={
                    "name": project_name,
                    "filter": f'metric.type = "{selected_metric}"',
                    "interval": interval,
                    "view": monitoring_v3.ListTimeSeriesRequest.TimeSeriesView.FULL,
                }
            )

            # 显示 Metric 数据
            st.subheader("Metric Data")
            for result in results:
                st.write(f"{result.metric.labels}: {result.points[0].value}")
    except Exception as e:
        st.error(f"Failed to retrieve metrics: {e}. Please ensure that the API is enabled and the service account has the appropriate permissions.")
