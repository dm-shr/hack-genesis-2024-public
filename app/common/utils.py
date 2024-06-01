import json
from langchain_core.runnables import Runnable, RunnablePassthrough
from langchain_core.tools import BaseTool
from langchain_core.language_models import BaseLanguageModel
from langchain_community.utils.openai_functions import FunctionDescription
from langchain_core.utils.json_schema import dereference_refs
from langchain_core.pydantic_v1 import BaseModel
from typing import Sequence
from typing import (
    Optional,
    Type,
)
from langchain_core.prompts.chat import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
)
from typing import List, Union, Tuple

from langchain_core.agents import AgentAction, AgentActionMessageLog, AgentFinish
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, Generation

from gigachat.models.function_call import FunctionCall
from langchain.agents.agent import AgentOutputParser
from langchain_core.messages import AIMessage, BaseMessage, FunctionMessage


def _create_function_message(
    agent_action: AgentAction, observation: str
) -> FunctionMessage:
    """Convert agent action and observation into a function message.
    Args:
        agent_action: the tool invocation request from the agent
        observation: the result of the tool invocation
    Returns:
        FunctionMessage that corresponds to the original tool invocation
    """
    if not isinstance(observation, str):
        try:
            content = json.dumps(observation, ensure_ascii=False)
        except Exception:
            content = str(observation)
    else:
        content = observation
    return FunctionMessage(
        name=agent_action.tool,
        content=content,
    )

def _convert_agent_action_to_messages(
    agent_action: AgentAction, observation: str
) -> List[BaseMessage]:
    """Convert an agent action to a message.

    This code is used to reconstruct the original AI message from the agent action.

    Args:
        agent_action: Agent action to convert.

    Returns:
        AIMessage that corresponds to the original tool invocation.
    """
    if isinstance(agent_action, AgentActionMessageLog):
        return list(agent_action.message_log) + [
            _create_function_message(agent_action, observation)
        ]
    else:
        return [AIMessage(content=agent_action.log)]

def format_to_gigachat_function_messages(
    intermediate_steps: Sequence[Tuple[AgentAction, str]],
) -> List[BaseMessage]:
    """Convert (AgentAction, tool output) tuples into FunctionMessages.

    Args:
        intermediate_steps: Steps the LLM has taken to date, along with observations

    Returns:
        list of messages to send to the LLM for the next prediction

    """
    messages = []

    for agent_action, observation in intermediate_steps:
        messages.extend(_convert_agent_action_to_messages(agent_action, observation))

    return messages


class GigaChatFunctionsAgentOutputParser(AgentOutputParser):
    """Parses a message into agent action/finish.

    Is meant to be used with GigaChat models, as it relies on the specific
    function_call parameter from GigaChat to convey what tools to use.

    If a function_call parameter is passed, then that is used to get
    the tool and tool input.

    If one is not passed, then the AIMessage is assumed to be the final output.
    """

    @property
    def _type(self) -> str:
        return "gigachat-functions-agent"

    @staticmethod
    def _parse_ai_message(message: BaseMessage) -> Union[AgentAction, AgentFinish]:
        """Parse an AI message."""
        if not isinstance(message, AIMessage):
            raise TypeError(f"Expected an AI message got {type(message)}")

        function_call = message.additional_kwargs.get("function_call", None)

        if isinstance(function_call, dict):
            function_call = FunctionCall(**function_call)

        if function_call:
            if not isinstance(function_call, FunctionCall):
                raise TypeError(f"Expected a FunctionCall message got {type(message)}")

            function_name = function_call.name
            _tool_input = function_call.arguments or {}

            # Legacy to support old code for OpenAI
            # HACK HACK HACK:
            # The code that encodes tool input into Open AI uses a special variable
            # name called `__arg1` to handle old style tools that do not expose a
            # schema and expect a single string argument as an input.
            # We unpack the argument here if it exists.
            # Open AI does not support passing in a JSON array as an argument.
            if "__arg1" in _tool_input:
                tool_input = _tool_input["__arg1"]
            else:
                tool_input = _tool_input

            content_msg = f"responded: {message.content}\n" if message.content else "\n"
            log = f"\nInvoking: `{function_name}` with `{tool_input}`\n{content_msg}\n"
            return AgentActionMessageLog(
                tool=function_name,
                tool_input=tool_input,
                log=log,
                message_log=[message],
            )

        return AgentFinish(
            return_values={"output": message.content}, log=str(message.content)
        )

    def parse_result(
        self, result: List[Generation], *, partial: bool = False
    ) -> Union[AgentAction, AgentFinish]:
        if not isinstance(result[0], ChatGeneration):
            raise ValueError("This output parser only works on ChatGeneration output")
        message = result[0].message
        return self._parse_ai_message(message)

    def parse(self, text: str) -> Union[AgentAction, AgentFinish]:
        raise ValueError("Can only parse messages")

def convert_pydantic_to_gigachat_function(
    model: Type[BaseModel],
    *,
    name: Optional[str] = None,
    description: Optional[str] = None,
) -> FunctionDescription:
    """Converts a Pydantic model to a function description for the GigaChat API."""
    schema = dereference_refs(model.schema())
    schema.pop("definitions", None)
    if "properties" in schema:
        for key in schema["properties"]:
            if "type" not in schema["properties"][key]:
                schema["properties"][key]["type"] = "object"
            if "description" not in schema["properties"][key]:
                schema["properties"][key]["description"] = ""
    return {
        "name": name or schema["title"],
        "description": description or schema["description"],
        "parameters": schema,
    }

def format_tool_to_gigachat_function(tool: BaseTool) -> FunctionDescription:
    """Format tool into the OpenAI function API."""
    if tool.args_schema:
        return convert_pydantic_to_gigachat_function(
            tool.args_schema, name=tool.name, description=tool.description
        )
    else:
        return {
            "name": tool.name,
            "description": tool.description,
            "parameters": {
                # This is a hack to get around the fact that some tools
                # do not expose an args_schema, and expect an argument
                # which is a string.
                # And Open AI does not support an array type for the
                # parameters.
                "properties": {
                    "__arg1": {"title": "__arg1", "type": "string"},
                },
                "required": ["__arg1"],
                "type": "object",
            },
        }

def create_gigachat_agent_prompt() -> ChatPromptTemplate:
    messages = [
        MessagesPlaceholder(variable_name="chat_history", optional=True),
        HumanMessagePromptTemplate.from_template("{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ]
    return ChatPromptTemplate.from_messages(messages=messages)

def create_gigachat_functions_agent(
    llm: BaseLanguageModel,
    tools: Sequence[BaseTool],
    prompt: ChatPromptTemplate = create_gigachat_agent_prompt(),
) -> Runnable:
    """Create an agent that uses GigaChat function calling.

    Args:
        llm: LLM to use as the agent. Should work with OpenAI function calling,
            so either be an GigaChat model that supports that or a wrapper of
            a different model that adds in equivalent support.
        tools: Tools this agent has access to.
        prompt: The prompt to use, must have input key `agent_scratchpad`, which will
            contain agent action and tool output messages.

    Returns:
        A Runnable sequence representing an agent. It takes as input all the same input
        variables as the prompt passed in does. It returns as output either an
        AgentAction or AgentFinish.

    TODO: Support gigachat
    Example (old version from openAI):

        Creating an agent with no memory

        .. code-block:: python

            from langchain_community.chat_models import ChatOpenAI
            from langchain.agents import AgentExecutor, create_openai_functions_agent
            from langchain import hub

            prompt = hub.pull("hwchase17/openai-functions-agent")
            model = ChatOpenAI()
            tools = ...

            agent = create_gigachat_functions_agent(model, tools, prompt)
            agent_executor = AgentExecutor(agent=agent, tools=tools)

            agent_executor.invoke({"input": "hi"})

            # Using with chat history
            from langchain_core.messages import AIMessage, HumanMessage
            agent_executor.invoke(
                {
                    "input": "what's my name?",
                    "chat_history": [
                        HumanMessage(content="hi! my name is bob"),
                        AIMessage(content="Hello Bob! How can I assist you today?"),
                    ],
                }
            )

    Creating prompt example:

        .. code-block:: python

            from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

            prompt = ChatPromptTemplate.from_messages(
                [
                    ("system", "You are a helpful assistant"),
                    MessagesPlaceholder("chat_history", optional=True),
                    ("human", "{input}"),
                    MessagesPlaceholder("agent_scratchpad"),
                ]
            )
    """
    if "agent_scratchpad" not in prompt.input_variables:
        raise ValueError(
            "Prompt must have input variable `agent_scratchpad`, but wasn't found. "
            f"Found {prompt.input_variables} instead."
        )
    llm_with_tools = llm.bind(
        functions=[format_tool_to_gigachat_function(t) for t in tools]
    )
    agent = (
        RunnablePassthrough.assign(
            agent_scratchpad=lambda x: format_to_gigachat_function_messages(
                x["intermediate_steps"]
            )
        )
        | prompt
        | llm_with_tools
        | GigaChatFunctionsAgentOutputParser()
    )
    return agent