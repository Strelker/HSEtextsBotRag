#!/bin/bash

# Создание виртуального окружения
python3 -m venv venv

# Активация виртуального окружения
source venv/bin/activate

# Установка зависимостей
pip install --upgrade pip
pip install -r requirements.txt

# Запуск приложения
# python3 app.py
python3 app.py

# Деактивация виртуального окружения (опционально)
deactivate