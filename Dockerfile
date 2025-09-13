FROM python:3.11

# Instala dependencias del sistema necesarias para face_recognition
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    libopenblas-dev \
    liblapack-dev \
    libjpeg-dev \
    libpng-dev \
    libtiff-dev \
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev \
    libv4l-dev \
    libxvidcore-dev \
    libx264-dev \
    libgtk-3-dev \
    && rm -rf /var/lib/apt/lists/*

# Copia los archivos del proyecto
WORKDIR /app
COPY . /app
# Instala las dependencias de Python
RUN pip install --upgrade pip
RUN pip install -r requirements.txt
# Expone el puerto 3000
EXPOSE 3000

# Comando para iniciar el servicio
CMD ["uvicorn", "face_service:app", "--host", "0.0.0.0", "--port", "3000"]