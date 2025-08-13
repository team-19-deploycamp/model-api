# Gunakan image Python resmi sebagai base image
FROM python:3.10.2-slim

# Instal dependensi sistem yang dibutuhkan untuk mengkompilasi paket C++
# Perintah ini akan menginstal build-essential (GCC, G++) dan python3-dev
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    python3-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Tetapkan direktori kerja di dalam container
WORKDIR /deploycamp-capstone

# Salin file requirements.txt ke direktori kerja
COPY requirements.txt .

# Instal semua dependensi Python
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Salin seluruh folder 'code' ke direktori kerja di dalam container
COPY code/ ./code/

COPY dataset/ ./dataset/

# Ekspos port yang akan digunakan oleh aplikasi
EXPOSE 8000

# Tentukan perintah default untuk menjalankan aplikasi
CMD ["/bin/sh", "-c", "uvicorn code.api.app:app --host 0.0.0.0 --port 8000"]