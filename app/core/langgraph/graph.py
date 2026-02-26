"""This file contains the LangGraph Agent/workflow and interactions with the LLM."""

import asyncio
from typing import AsyncGenerator, Optional
from urllib.parse import quote_plus

from asgiref.sync import sync_to_async
from langchain_core.messages import (
    BaseMessage,
    ToolMessage,
    convert_to_openai_messages,
)
from langfuse.langchain import CallbackHandler
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.graph import END, StateGraph
from langgraph.graph.state import Command, CompiledStateGraph
from langgraph.types import RunnableConfig, StateSnapshot
from psycopg_pool import AsyncConnectionPool

from app.core.config import Environment, settings
from app.core.langgraph.tools import tools
from app.core.logging import logger
from app.core.metrics import llm_inference_duration_seconds
from app.core.prompts import load_system_prompt
from app.schemas import GraphState, Message
from app.services.llm import llm_service
from app.utils import dump_messages, prepare_messages, process_llm_response


class AtlasAgent:
    """LangGraph powered agent with safe fallback to direct LLM."""

    def __init__(self):

        self.llm_service = llm_service
        self.llm_service.bind_tools(tools)

        self.tools_by_name = {tool.name: tool for tool in tools}

        self._connection_pool: Optional[AsyncConnectionPool] = None
        self._graph: Optional[CompiledStateGraph] = None

        logger.info(
            "atlas_agent_initialized",
            model=settings.DEFAULT_LLM_MODEL,
            environment=settings.ENVIRONMENT.value,
        )

    async def _get_connection_pool(self):

        if self._connection_pool is None:

            connection_url = (
                "postgresql://"
                f"{quote_plus(settings.POSTGRES_USER)}:"
                f"{quote_plus(settings.POSTGRES_PASSWORD)}@"
                f"{settings.POSTGRES_HOST}:"
                f"{settings.POSTGRES_PORT}/"
                f"{settings.POSTGRES_DB}"
            )

            self._connection_pool = AsyncConnectionPool(
                connection_url,
                open=False,
                max_size=settings.POSTGRES_POOL_SIZE,
                kwargs={
                    "autocommit": True,
                    "connect_timeout": 5,
                    "prepare_threshold": None,
                },
            )

            await self._connection_pool.open()

            logger.info("connection_pool_created")

        return self._connection_pool

    async def _get_relevant_memory(self, user_id: str, query: str):

        return ""

    async def _update_long_term_memory(self, user_id: str, messages, metadata=None):

        pass

    async def _chat(self, state: GraphState, config: RunnableConfig):

        current_llm = self.llm_service.get_llm()

        model_name = getattr(
            current_llm,
            "model_name",
            settings.DEFAULT_LLM_MODEL,
        )

        SYSTEM_PROMPT = load_system_prompt(
            long_term_memory=state.long_term_memory
        )

        messages = prepare_messages(
            state.messages,
            current_llm,
            SYSTEM_PROMPT,
        )

        try:

            with llm_inference_duration_seconds.labels(
                model=model_name
            ).time():

                response_message = await self.llm_service.call(
                    dump_messages(messages)
                )

            response_message = process_llm_response(response_message)

            logger.info(
                "llm_response_generated",
                session_id=config["configurable"]["thread_id"],
                model=model_name,
            )

            if response_message.tool_calls:
                goto = "tool_call"
            else:
                goto = END

            return Command(
                update={"messages": [response_message]},
                goto=goto,
            )

        except Exception as e:

            logger.error("llm_call_failed", error=str(e))

            raise e

    async def _tool_call(self, state: GraphState):

        outputs = []

        for tool_call in state.messages[-1].tool_calls:

            tool_result = await self.tools_by_name[
                tool_call["name"]
            ].ainvoke(tool_call["args"])

            outputs.append(
                ToolMessage(
                    content=str(tool_result),
                    name=tool_call["name"],
                    tool_call_id=tool_call["id"],
                )
            )

        return Command(update={"messages": outputs}, goto="chat")

    async def create_graph(self):

        if self._graph is None:

            graph_builder = StateGraph(GraphState)

            graph_builder.add_node(
                "chat",
                self._chat,
                ends=["tool_call", END],
            )

            graph_builder.add_node(
                "tool_call",
                self._tool_call,
                ends=["chat"],
            )

            graph_builder.set_entry_point("chat")
            graph_builder.set_finish_point("chat")

            pool = await self._get_connection_pool()

            checkpointer = AsyncPostgresSaver(pool)

            await checkpointer.setup()

            self._graph = graph_builder.compile(
                checkpointer=checkpointer,
                name="AtlasAgent",
            )

            logger.info("graph_created")

        return self._graph

    async def get_response(
        self,
        messages: list[Message],
        session_id: str,
        user_id: Optional[str] = None,
    ):

        if self._graph is None:
            await self.create_graph()

        config = {
            "configurable": {"thread_id": session_id},
            "callbacks": [CallbackHandler()],
            "metadata": {
                "user_id": user_id,
                "session_id": session_id,
            },
        }

        relevant_memory = (
            await self._get_relevant_memory(
                user_id,
                messages[-1].content,
            )
            or ""
        )

        try:

            response = await self._graph.ainvoke(
                {
                    "messages": dump_messages(messages),
                    "long_term_memory": relevant_memory,
                },
                config=config,
            )

            processed = self.__process_messages(
                response["messages"]
            )

            if processed:
                return processed

        except Exception as e:

            logger.error(
                "langgraph_failed_using_fallback",
                error=str(e),
            )

        try:

            llm = self.llm_service.get_llm()

            result = await llm.ainvoke(
                messages[-1].content
            )

            text = (
                result.content
                if hasattr(result, "content")
                else str(result)
            )

            return [
                Message(
                    role="assistant",
                    content=text,
                )
            ]

        except Exception as e:

            logger.error("fallback_failed", error=str(e))

            return [
                Message(
                    role="assistant",
                    content="System error occurred.",
                )
            ]

    async def get_stream_response(
        self,
        messages,
        session_id,
        user_id=None,
    ) -> AsyncGenerator[str, None]:

        if self._graph is None:
            await self.create_graph()

        config = {
            "configurable": {"thread_id": session_id},
        }

        async for token, _ in self._graph.astream(
            {"messages": dump_messages(messages)},
            config,
            stream_mode="messages",
        ):

            if token.content:
                yield token.content

    async def get_chat_history(self, session_id):

        if self._graph is None:
            await self.create_graph()

        state: StateSnapshot = await sync_to_async(
            self._graph.get_state
        )(
            config={
                "configurable": {
                    "thread_id": session_id
                }
            }
        )

        if not state.values:
            return []

        return self.__process_messages(
            state.values["messages"]
        )

    def __process_messages(self, messages):

        openai_messages = convert_to_openai_messages(
            messages
        )

        return [
            Message(
                role=m["role"],
                content=str(m["content"]),
            )
            for m in openai_messages
            if m["role"] in ["assistant", "user"]
            and m["content"]
        ]

    async def clear_chat_history(self, session_id):

        pool = await self._get_connection_pool()

        async with pool.connection() as conn:

            for table in settings.CHECKPOINT_TABLES:

                await conn.execute(
                    f"DELETE FROM {table} WHERE thread_id = %s",
                    (session_id,),
                )

        logger.info("chat_history_cleared")
