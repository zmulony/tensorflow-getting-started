FROM tensorflow/tensorflow

RUN             apt-get update \
             && apt-get install -y --no-install-recommends \
                    ca-certificates \
                    build-essential \
                    git \
                    python \
                    python-pip \
                    python-setuptools \
             && git clone https://github.com/tensorflow/models.git \
             && cd models \
             && pip install --user -r official/requirements.txt \
             && export PYTHONPATH=$PYTHONPATH:`pwd`

ENV PYTHONPATH=$PYTHONPATH:/notebooks/models:/notebooks/models/official

COPY mnist-simple.py ./

CMD ["python", "mnist-simple.py"]