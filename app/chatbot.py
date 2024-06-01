import streamlit as st
from common import streamlit_texts as TEXTS, system_prompt, prompt_postfix
from common.unstructured_tools import UnstructuredTool
from common.structured_tools import StructuredTool
from common import AUTH_DATA, MODEL, SCOPE, TEMPERATURE, TIMEOUT, logger
from langchain.agents import AgentExecutor
from common.utils import create_gigachat_functions_agent

from langchain_community.chat_models import GigaChat
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage


# Инициализируем модель
llm = GigaChat(
    credentials=AUTH_DATA,
    verify_ssl_certs=False,
    timeout=TIMEOUT,
    model=MODEL,
    scope=SCOPE,
    temperature=TEMPERATURE
)

# Собираем агента
tools = [UnstructuredTool(), StructuredTool()]
agent_runnable = create_gigachat_functions_agent(llm, tools)

agent_executor = AgentExecutor(
    agent=agent_runnable, tools=tools, verbose=True, return_intermediate_steps=True
)

st.set_page_config(page_title=TEXTS.PAGE_TITLE)
st.title(TEXTS.TITLE)

st.markdown(TEXTS.HINT)


TEXTS.HR

st.markdown(TEXTS.SIDEBAR_STYLE, unsafe_allow_html=True)

with st.sidebar:
    # st.image(os.path.join("resources", "img", "logo.jpeg"))
    st.markdown(TEXTS.COMMAND_EXAMPLES)
    st.code(TEXTS.ECONOMY_DECREASE)
    st.code(TEXTS.LOAN_VOLUME)
    st.code(TEXTS.LTIP)
    st.code(TEXTS.DEFAULT)
    st.code(TEXTS.RENT_PAYMENTS)
    # st.markdown(TEXTS.EXAMPLE_PAPER)

if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Привет, я ИИ Ассистент!"}]

if "chat_history" not in st.session_state:
    st.session_state.chat_history = [SystemMessage(content=system_prompt)]

if "metadata" not in st.session_state:
    st.session_state.metadata = None

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"], unsafe_allow_html=True)

if prompt := st.chat_input("Обратитесь ко мне..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt, unsafe_allow_html=True)

    with st.spinner(TEXTS.WAITING):
        with st.chat_message("assistant"):
            # Запуск агента
            logger.info(str(st.session_state.chat_history))
            try:
                result = agent_executor.invoke(
                    {
                        "chat_history": st.session_state.chat_history,
                        "input": prompt + prompt_postfix,
                    }
                )
            except Exception as e:
                logger.error(str(e))
                result = {"output": TEXTS.SORRY}

            image=None
            # Handling complex outputs
            if type(result["output"]) == dict:
                response = result["output"]["result"]
                if "image" in result["output"].keys():
                    image = result["output"]["image"]
                # st.session_state.metadata = result["output"]["metadata"]
                # st.session_state.chat_history.append(HumanMessage(content=str(st.session_state.metadata)))
            else:
                response = result["output"]

            # обновление истории диалога
            st.session_state.chat_history.append(HumanMessage(content=prompt))
            # st.session_state.chat_history += result["output"]["intermediate_steps"]
            st.session_state.chat_history.append(AIMessage(content=response))
            # don't vanish the system prompt

            # рисуем ответ
            st.markdown(response, unsafe_allow_html=True)      
            if image:
                st.image(result["output"]["image"])

    st.session_state.messages.append({"role": "assistant", "content": response})