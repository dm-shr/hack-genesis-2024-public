import os
import json
import logging


# Общие функции
def save_file(file_dir, file):
  with open(file_dir, 'w') as f:
    f.write(file)

def save_json(file_dir, file):
  with open(file_dir, 'w') as f:
    json.dump(file, f, ensure_ascii=False, indent=4)

# workarouund for dev environment
if os.getenv("environment") != "production":
    from dotenv import load_dotenv
    load_dotenv("./.env")

logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO
)

logger = logging.getLogger(__name__)

# Авторизация в сервисе GigaChat
AUTH_DATA = os.getenv("AUTH_DATA")
API_KEY = os.getenv("API_KEY")
FOLDER_ID = os.getenv("FOLDER_ID")

# Заголовки для Киберленинки
headers = {
    'content-type': 'application/json',
    'sec-fetch-dest': 'empty',
    'sec-fetch-mode': 'cors',
    'sec-fetch-site': 'same-origin'
}

# параметры для ЛЛМ
MODEL = os.getenv("MODEL", "GigaChat-Pro")
SCOPE = os.getenv("SCOPE", "GIGACHAT_API_PERS")
TEMPERATURE = float(os.getenv("TEMPERATURE", 0.1))
TIMEOUT = int(os.getenv("TIMEOUT", 600))

PROMPT_PATH = os.path.join('.', 'prompts')

# Промпт для нашего агента
system_prompt = """
Ты ИИ ассистент по финансовой деятельности, специализирующийся на помощи работникам по извлечению данных из финансовой отчетности. 
Ты должен помочь пользователю выбрать для ответа структурированные данные и предоставить необходимую информацию по ним. 

Если ты не можешь найти необходимые данные, используй предоставленные тебе инструменты (tools)

У тебя есть следующие инструменты (tools).

# unstructured_tool 

Выполняет поиск ответа на вопрос пользователя в неструктурированной базе данных финансовых отчётов.
Примеры таких вопросов:
- Группа считает признаками дефолта следующие виды событий?
- Из чего состоят арендные платежи?
- В каких случаях группа переоценивает обязательство по аренде?
- Что такое «LTIP»?
- Что такое амортизированная стоимость?
- Когда прекращается признание финансовых обязательств?
И другие вопросы...

# structured_tool

Выполняет поиск ответа на вопрос пользователя в структурированной базе данных финансовых отчётов.
  Примеры таких вопросов:
  - Размер нераспределенной прибыли Компании на 31.12.2022
  - Итого обязательства компании на 31 декабря 2022 г.?
  - Итого дефицит капитала компании по состоянию на 31 декабря 2022 года
  - Чему раны прочие нефинансовые активы на 31 декабря 2022? 
  - Итого прибыль за 2020 год?
  - Итого займы выданные по состоянию на 31 декабря 2021 года
  И другие вопросы...
"""

prompt_postfix = "Если тебя спрашивают вопрос, требующий нового обращения к инструментам (tools) сделай это, а не придумывай данные."