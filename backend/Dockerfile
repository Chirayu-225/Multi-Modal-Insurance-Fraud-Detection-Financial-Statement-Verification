FROM python:3.11-slim

# tesseract-ocr binary is required by pytesseract (app/services/ocr.py::TesseractOCR).
# libgl1 is a common Pillow/opencv-adjacent runtime dependency on slim images.
RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN mkdir -p uploads app/ml/artifacts data

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
