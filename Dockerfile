FROM nvidia/cuda:12.1.0-base
RUN apt-get update && apt-get install -y git python3-pip

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 7860

CMD ["python", "rag_app.py"]