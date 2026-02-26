"""LLM service for managing LLM calls with retries and fallback mechanisms.
Supports local Ollama models (free) and OpenAI models (optional).
"""

from typing import Any, Dict, List, Optional

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage

# OpenAI (optional)
from langchain_openai import ChatOpenAI
from openai import APIError, APITimeoutError, OpenAIError, RateLimitError

# Ollama (FREE local)
from langchain_ollama import ChatOllama

from tenacity import (
    before_sleep_log,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from app.core.config import Environment, settings
from app.core.logging import logger


def create_llm(model_name: str) -> BaseChatModel:
    """
    Factory function to create LLM safely depending on provider.
    """

    # FREE LOCAL MODELS
    if model_name.startswith("llama") or model_name.startswith("mistral") or model_name.startswith("phi"):
        logger.info("creating_ollama_llm", model=model_name)

        return ChatOllama(
            model=model_name,
            temperature=settings.DEFAULT_LLM_TEMPERATURE,
        )

    # OPENAI MODELS (optional)
    if model_name.startswith("gpt"):

        if not settings.OPENAI_API_KEY or settings.OPENAI_API_KEY == "not_required":
            raise RuntimeError("OpenAI API key not configured")

        logger.info("creating_openai_llm", model=model_name)

        return ChatOpenAI(
            model=model_name,
            temperature=settings.DEFAULT_LLM_TEMPERATURE,
            api_key=settings.OPENAI_API_KEY,
            max_tokens=settings.MAX_TOKENS,
            top_p=0.95 if settings.ENVIRONMENT == Environment.PRODUCTION else 0.8,
        )

    raise ValueError(f"Unsupported model: {model_name}")


class LLMRegistry:
    """
    Registry of available models.
    Order defines fallback priority.
    """

    LLMS: List[Dict[str, Any]] = [
        {
            "name": "llama3",
            "llm": create_llm("llama3"),
        },

        # Optional OpenAI models (only work if API key exists)
        {
            "name": "gpt-4o-mini",
            "llm": None,
        },
        {
            "name": "gpt-4o",
            "llm": None,
        },
    ]

    @classmethod
    def get(cls, model_name: str, **kwargs) -> BaseChatModel:

        model_entry = None

        for entry in cls.LLMS:
            if entry["name"] == model_name:
                model_entry = entry
                break

        if not model_entry:
            available = [entry["name"] for entry in cls.LLMS]
            raise ValueError(
                f"model '{model_name}' not found. available: {available}"
            )

        # Create dynamically if needed
        if model_entry["llm"] is None:
            model_entry["llm"] = create_llm(model_name)

        # Override settings if kwargs provided
        if kwargs:

            logger.debug(
                "creating_llm_with_custom_args",
                model=model_name,
                args=list(kwargs.keys()),
            )

            if model_name.startswith("llama"):
                return ChatOllama(
                    model=model_name,
                    **kwargs,
                )

            return ChatOpenAI(
                model=model_name,
                api_key=settings.OPENAI_API_KEY,
                **kwargs,
            )

        logger.debug("using_default_llm_instance", model=model_name)

        return model_entry["llm"]

    @classmethod
    def get_all_names(cls) -> List[str]:

        return [entry["name"] for entry in cls.LLMS]

    @classmethod
    def get_model_at_index(cls, index: int) -> Dict[str, Any]:

        if 0 <= index < len(cls.LLMS):
            return cls.LLMS[index]

        return cls.LLMS[0]


class LLMService:
    """
    Handles LLM calls with retry and automatic fallback.
    """

    def __init__(self):

        self._llm: Optional[BaseChatModel] = None
        self._current_model_index: int = 0

        all_names = LLMRegistry.get_all_names()

        try:

            self._current_model_index = all_names.index(
                settings.DEFAULT_LLM_MODEL
            )

            self._llm = LLMRegistry.get(settings.DEFAULT_LLM_MODEL)

            logger.info(
                "llm_initialized",
                model=settings.DEFAULT_LLM_MODEL,
            )

        except Exception as e:

            logger.warning(
                "default_model_failed_using_llama3",
                error=str(e),
            )

            self._current_model_index = 0
            self._llm = LLMRegistry.get("llama3")

    def _get_next_model_index(self):

        total = len(LLMRegistry.LLMS)

        return (self._current_model_index + 1) % total

    def _switch_to_next_model(self):

        try:

            next_index = self._get_next_model_index()

            model_name = LLMRegistry.LLMS[next_index]["name"]

            logger.warning(
                "switching_model",
                to=model_name,
            )

            self._llm = LLMRegistry.get(model_name)

            self._current_model_index = next_index

            return True

        except Exception as e:

            logger.error(
                "model_switch_failed",
                error=str(e),
            )

            return False

    @retry(
        stop=stop_after_attempt(settings.MAX_LLM_CALL_RETRIES),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type(
            (RateLimitError, APITimeoutError, APIError, Exception)
        ),
        before_sleep=before_sleep_log(logger, "WARNING"),
        reraise=True,
    )
    async def _call_llm_with_retry(
        self,
        messages: List[BaseMessage],
    ) -> BaseMessage:

        if not self._llm:
            raise RuntimeError("LLM not initialized")

        response = await self._llm.ainvoke(messages)

        return response

    async def call(
        self,
        messages: List[BaseMessage],
        model_name: Optional[str] = None,
        **kwargs,
    ) -> BaseMessage:

        if model_name:

            self._llm = LLMRegistry.get(model_name, **kwargs)

        total_models = len(LLMRegistry.LLMS)

        tried = 0

        last_error = None

        while tried < total_models:

            try:

                return await self._call_llm_with_retry(messages)

            except Exception as e:

                last_error = e

                tried += 1

                logger.error(
                    "model_failed",
                    model=LLMRegistry.LLMS[self._current_model_index]["name"],
                    error=str(e),
                )

                if not self._switch_to_next_model():
                    break

        raise RuntimeError(
            f"All models failed. Last error: {last_error}"
        )

    def get_llm(self):

        return self._llm

    def bind_tools(self, tools: List):

        if self._llm:

            self._llm = self._llm.bind_tools(tools)

        return self


# global instance
llm_service = LLMService()
