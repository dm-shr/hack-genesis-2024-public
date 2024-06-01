import os

from typing import Type
from langchain.tools import BaseTool
from langchain_community.vectorstores import FAISS
from langchain.pydantic_v1 import BaseModel, Field
from langchain_community.chat_models import GigaChat
from langchain.prompts import load_prompt

from yandex_chain import YandexEmbeddings

from common import logger, FOLDER_ID, API_KEY, AUTH_DATA, PROMPT_PATH, SCOPE, MODEL
from common.knowledge_graphs import is_company_match


faiss_vdb = "faiss_unstructured_v0.0.2"

db = FAISS.load_local(
    os.path.join("..", "data", "unstructured_vdb", faiss_vdb),
    YandexEmbeddings(folder_id=FOLDER_ID, api_key=API_KEY),
    allow_dangerous_deserialization = True
)

class UnstructuredToolInput(BaseModel):
    user_query: str = Field(
        query="Поисковый вопрос пользователя, направленный на неструктурированные данные"
    )
    company_name: str = Field(
        query="Название компании, может быть пустым"
    )

class UnstructuredTool(BaseTool):
    name = "unstructured_tool"
    description = """
    Выполняет поиск ответа на вопрос пользователя в неструктурированной базе данных финансовых отчётов.
    Примеры таких вопросов:
    - Группа считает признаками дефолта следующие виды событий?
    - Из чего состоят арендные платежи?
    - В каких случаях группа переоценивает обязательство по аренде?
    - Что такое «LTIP»?
    - Что такое амортизированная стоимость?
    - Когда прекращается признание финансовых обязательств?
    И другие вопросы...
    """
    args_schema: Type[BaseModel] = UnstructuredToolInput
    return_direct: bool = True
    giga = GigaChat(credentials=AUTH_DATA, verify_ssl_certs=False, scope=SCOPE, model=MODEL)
    prompt = load_prompt(os.path.join(PROMPT_PATH, "rephrase.yaml"))
    chain = prompt | giga

    def _run(
        self,
        user_query: str="",
        company_name: str="",
        run_manager=None,
    ) -> str:
        logger.info(f"Metadata: {user_query} {company_name}")

        docs = db.similarity_search(user_query, k=10)

        if company_name: # filter docs by company name
            docs_filtered = [d for d in docs if is_company_match(d.metadata['source'].split('/')[-2] + ".pdf", company_name)]
            if len(docs_filtered) > 0:
                logger.info(f"Matching docs {docs_filtered}")
                docs = docs_filtered

        page = docs[0].metadata["source"].split("/")[-1].split("_")[-1].split(".")[0]
        file_name = docs[0].metadata['source'].split('/')[-2] + ".pdf"

        result = self.chain.invoke(
            {
                "source_text": docs[0].page_content
            }
        ).content

        logger.info(f"Output: {result}")

        result_markdown = f"{result}\n\n**Источник:**\n\n- Имя файла: {file_name}\n- Страница: {page}"

        return {
           "result": result_markdown, #docs[0].page_content,
           "metadata": {"file_name": file_name, "page": page}
        }