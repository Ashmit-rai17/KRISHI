# Use an official PyTorch runtime
FROM python:3.10-slim

# Install dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy project files
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Expose port
EXPOSE 5000

# Start with Gunicorn (more production ready than flask run)
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "serve:app"]
