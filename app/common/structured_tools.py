import os

from typing import Type
from langchain.tools import BaseTool
from langchain_community.vectorstores import FAISS
from langchain.pydantic_v1 import BaseModel, Field
from langchain_community.chat_models import GigaChat
from langchain.prompts import load_prompt
from pdf2image import convert_from_path
from yandex_chain import YandexEmbeddings

from common import logger, FOLDER_ID, API_KEY, AUTH_DATA, PROMPT_PATH, SCOPE, MODEL
from common.knowledge_graphs import is_company_match


faiss_vdb = "faiss_structured_v0.0.1"

db = FAISS.load_local(
    os.path.join("..", "data", "structured_vdb", faiss_vdb),
    YandexEmbeddings(folder_id=FOLDER_ID, api_key=API_KEY),
    allow_dangerous_deserialization = True
)

class StructuredToolInput(BaseModel):
    user_query: str = Field(
        query="Поисковый вопрос пользователя, направленный на cтруктурированные данные"
    )
    company_name: str = Field(
        query="Название компании, может быть пустым"
    )

class StructuredTool(BaseTool):
    name = "structured_tool"
    description = """
    Выполняет поиск ответа на вопрос пользователя в структурированной базе данных финансовых отчётов.
    Примеры таких вопросов:
    - Размер нераспределенной прибыли Компании на 31.12.2022
    - Итого обязательства компании на 31 декабря 2022 г.?
    - Итого дефицит капитала компании по состоянию на 31 декабря 2022 года
    - Чему раны прочие нефинансовые активы на 31 декабря 2022? 
    - Итого прибыль за 2020 год?
    - На сколько снизилась экономика 
    - Итого займы выданные по состоянию на 31 декабря 2021 года
    И другие вопросы...
    """
    args_schema: Type[BaseModel] = StructuredToolInput
    return_direct: bool = True
    giga = GigaChat(credentials=AUTH_DATA, verify_ssl_certs=False, scope=SCOPE, model=MODEL)
    prompt = load_prompt(os.path.join(PROMPT_PATH, "extract.yaml"))
    chain = prompt | giga

    def _run(
        self,
        user_query: str="",
        company_name: str="",
        run_manager=None,
    ) -> str:
        logger.info(f"Metadata: {user_query}")

        docs = db.similarity_search(user_query, k=25)

        if company_name: # filter docs by company name
            docs_filtered = [d for d in docs if is_company_match(d.metadata['source'].split('/')[-2] + ".pdf", company_name)]
            if len(docs_filtered) > 0:
                logger.info(f"Matching docs {docs_filtered}")
                docs = docs_filtered

        page = docs[0].metadata["source"].split("/")[-1].split("_")[-1].split(".")[0].replace("page", "")
        file_name = docs[0].metadata['source'].split('/')[-2] + ".pdf"

        try:
            image = convert_from_path(os.path.join("..", "data", "pdf", file_name), first_page=int(page) + 1, last_page=int(page) + 1)[0]
        except Exception as e:
            logger.error(str(e))
            logger.info(os.path.join("..", "data", "pdf", file_name))
            image = None

        result = self.chain.invoke(
            {
                "source_text": user_query,
                "source_data": docs[0].page_content
            }
        ).content

        logger.info(f"Output: {result}")

        result_markdown = f"{result}\n\n**Источник:**\n\n- Имя файла: {file_name}\n- Страница: {page}"

        return {
           "result": result_markdown, #docs[0].page_content,
           "metadata": {"file_name": file_name, "page": page},
           "image": image
        }