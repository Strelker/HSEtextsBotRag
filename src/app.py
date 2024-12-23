import os
import io
import logging
import requests
import nltk
import chromadb
import chardet
from docx import Document
from dotenv import load_dotenv
from telebot import TeleBot
from sentence_transformers import SentenceTransformer
import spacy
from nltk.tokenize import sent_tokenize

# Инициализация переменных окружения
load_dotenv()

# Загрузка модулей NLTK, необходимых для обработки текста
nltk_dependencies = ['punkt', 'punkt_tab', 'averaged_perceptron_tagger', 'maxent_ne_chunker', 'words']
for dependency in nltk_dependencies:
    nltk.download(dependency)

# Настройка системы логирования
logging.basicConfig(
    format='%(asctime)s | %(levelname)s | %(message)s',
    level=logging.INFO
)
logger = logging.getLogger("FileProcessingBot")

# Создание подключения к ChromaDB и коллекции для хранения данных
chroma_client = chromadb.Client()
document_storage = chroma_client.create_collection(name="text_documents")

# Инициализация моделей для обработки текста
embedding_model = SentenceTransformer('distiluse-base-multilingual-cased-v1')
entity_recognizer = spacy.load("en_core_web_sm")

def extract_file_content(file_bytes, file_name):
    """Получает текстовое содержимое файла в зависимости от его формата."""
    try:
        if file_name.lower().endswith('.txt'):
            encoding = chardet.detect(file_bytes)['encoding']
            return file_bytes.decode(encoding)
        elif file_name.lower().endswith('.docx'):
            doc = Document(io.BytesIO(file_bytes))
            return "\n".join([paragraph.text for paragraph in doc.paragraphs])
        else:
            logger.warning(f"Формат файла {file_name} не поддерживается.")
            return None
    except Exception as error:
        logger.error(f"Ошибка при обработке файла {file_name}: {error}")
        return None

def populate_database(file_content, file_name):
    """Создаёт базу данных на основе предложений из текста."""
    текст = extract_file_content(file_content, file_name)
    if not текст:
        return "Не удалось обработать содержимое файла."

    предложения = sent_tokenize(текст)
    logger.info(f"Файл {file_name} разделён на {len(предложения)} предложений.")

    for index, предложение in enumerate(предложения):
        try:
            вектор = embedding_model.encode(предложение)
            document_storage.add(
                ids=[str(index)],
                embeddings=[вектор],
                documents=[предложение]
            )
            logger.debug(f"Добавлено предложение {index}: {предложение[:50]}...")
        except Exception as error:
            logger.error(f"Не удалось добавить предложение {index}: {error}")
            return f"Ошибка при добавлении данных из файла {file_name}."

    return "Данные успешно сохранены!"

def download_and_process_file(file_id, bot, file_name):
    """Загружает файл с Telegram и передаёт его для обработки."""
    try:
        file_info = bot.get_file(file_id)
        download_url = f"https://api.telegram.org/file/bot{bot.token}/{file_info.file_path}"
        response = requests.get(download_url)
        if response.status_code == 200:
            logger.info(f"Файл {file_name} успешно загружен.")
            return populate_database(response.content, file_name)
        else:
            logger.error(f"Ошибка при скачивании файла {file_name}: {response.status_code}")
            return "Не удалось скачать файл."
    except Exception as error:
        logger.error(f"Ошибка при обработке файла {file_id}: {error}")
        return "Произошла ошибка при обработке загруженного файла."

def extract_named_entities(text):
    """Извлекает именованные сущности из текста."""
    parsed_text = entity_recognizer(text)
    return [(entity.text, entity.label_) for entity in parsed_text.ents]

def search_similar_content(query, collection, top_n=5):
    """Выполняет поиск наиболее похожих текстов в базе данных."""
    try:
        query_vector = embedding_model.encode(query)
        results = collection.query(
            query_embeddings=[query_vector],
            n_results=top_n
        )

        matched_documents = results.get('documents', [])
        if not matched_documents:
            return "Совпадений не найдено. Попробуйте уточнить запрос."

        ответ = "Результаты поиска:\n"
        for i, document in enumerate(matched_documents[0]):
            ответ += f"{i+1}. {document}\n"
        return ответ
    except Exception as error:
        logger.error(f"Ошибка во время поиска: {error}")
        return "Во время поиска произошла ошибка."

def send_welcome_message(message, bot):
    """Отправляет приветственное сообщение пользователю."""
    try:
        приветствие = (
            "Добро пожаловать! \ud83d\udc4b\n"
            "Я бот, который поможет вам искать информацию по текстовым файлам. \ud83d\udd0d\n"
            "Отправьте мне файл (.txt или .docx), и я обработаю его содержимое. \ud83d\udd04"
        )
        bot.send_message(message.chat.id, приветствие)
        logger.info(f"Приветственное сообщение отправлено пользователю {message.chat.id}.")
    except Exception as error:
        logger.error(f"Ошибка при отправке приветственного сообщения: {error}")

def handle_text_query(message, bot):
    """Обрабатывает текстовый запрос пользователя."""
    try:
        query = message.text
        logger.info(f"Получен запрос от пользователя {message.chat.id}: {query}")
        response = search_similar_content(query, document_storage)
        bot.send_message(message.chat.id, response)
    except Exception as error:
        logger.error(f"Ошибка при обработке запроса: {error}")
        bot.send_message(message.chat.id, "Не удалось обработать запрос.")

def handle_file_upload(message, bot):
    """Обрабатывает загруженный файл пользователя."""
    try:
        uploaded_file = message.document
        file_name = uploaded_file.file_name
        if file_name.lower().endswith(('.txt', '.docx')):
            result = download_and_process_file(uploaded_file.file_id, bot, file_name)
            bot.send_message(message.chat.id, result)
        else:
            bot.send_message(message.chat.id, "Пожалуйста, загрузите файл формата .txt или .docx.")
    except Exception as error:
        logger.error(f"Ошибка при загрузке файла: {error}")
        bot.send_message(message.chat.id, "Не удалось обработать загруженный файл.")

# Инициализация бота и регистрация обработчиков
telegram_token = os.getenv("BOT_TOKEN")
bot = TeleBot(telegram_token)

@bot.message_handler(commands=['start'])
def handle_start_command(message):
    send_welcome_message(message, bot)

@bot.message_handler(content_types=['document'])
def handle_document_upload(message):
    handle_file_upload(message, bot)

@bot.message_handler(func=lambda msg: True, content_types=['text'])
def handle_text_message(message):
    handle_text_query(message, bot)

if __name__ == "__main__":
    try:
        logger.info("Бот запущен и готов к работе.")
        bot.infinity_polling()
    except Exception as error:
        logger.critical(f"Критическая ошибка при запуске бота: {error}")
