FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    git \
    poppler-utils \
    tesseract-ocr \
    libreoffice \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file
COPY requirements-docker.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements-docker.txt

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p files
RUN mkdir -p vector_db
RUN mkdir -p chat_history

# Expose port for Gradio
EXPOSE 7860

# Set environment variables
ENV PYTHONPATH=/app
ENV GRADIO_SERVER_NAME=0.0.0.0
ENV GRADIO_SERVER_PORT=7860

# Create a script to run the application
RUN echo '#!/bin/bash\n\
python -m interface.research_agent_interface "$@"' > /app/run.sh && \
chmod +x /app/run.sh

# Set the entrypoint
ENTRYPOINT ["/app/run.sh"]

# Default command - can be overridden at runtime
CMD ["--platform", "groq", "--model", "llama-3.1-8b-instant", "--max_searches", "3"]
