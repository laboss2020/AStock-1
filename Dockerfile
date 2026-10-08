# A股大盘选股 — Docker 运行
#
# 构建：  docker build -t astock .
# 选股：  docker run --rm --env-file .env -v ./reports:/app/reports -v ./data_cache:/app/data_cache astock
# Web UI: docker run --rm --env-file .env -p 8501:8501 -v ./reports:/app/reports astock \
#           python -m streamlit run web/app.py --server.port 8501 --server.address 0.0.0.0
#
# 国内网络构建慢时换源：docker build --build-arg PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple -t astock .
FROM python:3.12-slim

WORKDIR /app

ARG PIP_INDEX_URL=""

COPY pyproject.toml README.md ./
COPY astock ./astock
COPY main.py ./
RUN pip install --no-cache-dir ${PIP_INDEX_URL:+--index-url "$PIP_INDEX_URL"} ".[web]"

# 控制台编码 + 日志实时输出（容器里也常跑中文）
ENV PYTHONIOENCODING=utf-8 \
    PYTHONUNBUFFERED=1

# K线/清单缓存与报告目录（挂载宿主目录可跨容器复用缓存）
VOLUME ["/app/data_cache", "/app/reports"]

CMD ["python", "main.py"]
