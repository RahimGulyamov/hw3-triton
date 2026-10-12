FROM nvcr.io/nvidia/tritonserver:23.12-py3 AS server

ENV PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
WORKDIR /workspace
COPY requirements.txt /workspace/requirements.txt
RUN python3 -m pip install --no-cache-dir -r requirements.txt
COPY model_repository /models
COPY scripts /workspace/scripts
COPY clients /workspace/clients
COPY profiling /workspace/profiling
RUN python3 scripts/export_model.py
EXPOSE 8000 8001 8002
HEALTHCHECK --interval=5s --timeout=3s --start-period=60s --retries=12 \
  CMD python3 -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/v2/models/mobilenet/ready', timeout=2)"
ENTRYPOINT ["tritonserver"]
CMD ["--model-repository=/models", "--strict-model-config=true", "--disable-auto-complete-config", "--allow-gpu-metrics=false", "--exit-on-error=true"]

FROM nvcr.io/nvidia/tritonserver:23.12-py3-sdk AS sdk

FROM server AS analyzer
COPY --from=sdk /workspace/install /opt/triton-sdk
COPY requirements-analyzer.txt /workspace/requirements-analyzer.txt
RUN apt-get update && apt-get install -y --no-install-recommends wkhtmltopdf libb64-0d \
    && rm -rf /var/lib/apt/lists/* \
    && python3 -m pip install --no-cache-dir -r requirements-analyzer.txt
ENV QT_QPA_PLATFORM=offscreen LD_LIBRARY_PATH=/opt/triton-sdk/lib:${LD_LIBRARY_PATH}
RUN /opt/triton-sdk/bin/perf_analyzer --help > /dev/null
ENTRYPOINT ["bash", "/workspace/scripts/model_analyzer.sh"]

# `docker build .` produces the inference server by default.
FROM server AS runtime
