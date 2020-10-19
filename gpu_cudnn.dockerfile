FROM datamachines/cudnn_tensorflow_opencv:10.2_2.3.0_4.4.0-20200803

COPY requirements_docker.txt /tmp/requirements.txt
RUN pip --no-cache-dir install -r /tmp/requirements.txt

EXPOSE 8888
CMD jupyter-notebook --ip=0.0.0.0 --port=8888 --no-browser --allow-root