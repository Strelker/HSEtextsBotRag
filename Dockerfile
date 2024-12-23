# Используем официальный образ Python
FROM python:3.11-slim

RUN apt update && apt install -y python3-tk

# Устанавливаем рабочую директорию внутри контейнера
WORKDIR /app

# Копируем файл зависимостей в контейнер
COPY src/requirements.txt /app/

# Устанавливаем зависимости из файла requirements.txt
RUN pip install --no-cache-dir -r requirements.txt && python3 -m spacy download en_core_web_sm

# Копируем все остальные файлы в контейнер
COPY src /app/

# Указываем команду для запуска бота
CMD ["python", "app.py"]