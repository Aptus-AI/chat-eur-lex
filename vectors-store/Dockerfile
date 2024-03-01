FROM python:3.8

WORKDIR /app

COPY requirements.txt .

COPY scripts/data_vectorizer.py .
COPY scripts/preprocessing_utils.py .
RUN pip install -r requirements.txt

CMD ["sh", "-c", "python -m nltk.downloader averaged_perceptron_tagger && python -m nltk.downloader punkt && python /app/data_vectorizer.py --config_path /app/config.yaml"]
