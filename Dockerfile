# Use official Python 3.10 image as base
FROM python:3.10.14-slim

# Set working directory
WORKDIR /app

# Copy requirements.txt to the working directory
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy all source code to the working directory
COPY . .

# Use environment variable for port
ARG DEFAULT_PORT=8501
ENV PORT=${DEFAULT_PORT}

# Enable permissions for correct file handling in container
ENV STREAMLIT_SERVER_ENABLE_CORS=false

RUN mkdir -p ~/.streamlit
RUN echo "\
[server]\n\
headless = true\n\
enableCORS=false\n\
enableXsrfProtection=false\n\
port = ${PORT}\n\
maxUploadSize=1028\n\
" > ~/.streamlit/config.toml

# Expose port 8501 for the Streamlit app
EXPOSE 8501

# Command to run the Streamlit app
CMD ["streamlit", "run", "streamlit_measurements.py", "--server.port=8501", "--server.address=0.0.0.0"]
