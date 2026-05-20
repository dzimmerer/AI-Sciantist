import asyncio
import json
import os
import sys
import time
from contextlib import AsyncExitStack
from pathlib import Path
from typing import Any

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from openai import OpenAI
from openai.types import Model

# CONFIGURATION
# API_BASE = "https://api.helmholtz-blablador.fz-juelich.de/"
API_BASE = "https://openrouter.ai/api/v1"
# API_BASE = "http://127.0.0.1:8595/"

# API_KEY = os.getenv("BLABLADOR_KEY")
# API_KEY = "sk-1234"
API_KEY = os.getenv("OPENROUTER_KEY")  # Allow override with OPENROUTER_KEY if set
# MODEL_NAME = "2 - Qwen3.5 122B, a new multimodal model from Feb 2026 with long context size and vision encoders"
# MODEL_NAME = "8 GLM-4.7-Flash"
# MODEL_NAME = "stepfun/step-3.5-flash:free"
# MODEL_NAME = "qwen3.5"
# MODEL_NAME = "qwen3.5-122b"
MODEL_NAME = "qwen/qwen3.6-plus-preview:free"


MAX_ROUNDS = 25  # Maximum number of tool call rounds before giving up. Each round allows the model to call one tool and receive results.

# 2. Web Search MCP Server Configuration
# We use 'uvx' to run the server directly from GitHub.
# Ensure you have the 'uv' tool installed in your environment.
WEB_SEARCH_MCP_SERVER_PARAMS = StdioServerParameters(
    command="uvx",
    args=["--from", "git+https://github.com/sydasif/web-search-mcp.git", "web-search-mcp"],
)

FILE_READER_MCP_SCRIPT = Path(__file__).with_name("file_reader_mcp.py")
WANDB_MCP_SCRIPT = Path(__file__).with_name("wandb_mcp.py")
GIT_MCP_SCRIPT = Path(__file__).with_name("git_mcp.py")

DEFAULT_SYSTEM_PROMPT = (
    "You are a helpful assistant. Use tools when needed. "
    "Never output <think> tags or internal reasoning. "
    "Return only the final user-facing answer."
)


def ensure_openai_base_url(base_url: str) -> str:
    """Ensure base URL points at an OpenAI-compatible API root."""
    normalized = base_url.rstrip("/")
    if normalized.endswith("/v1"):
        return normalized
    return f"{normalized}/v1"


def parse_chat_response(response: Any) -> dict[str, Any]:
    """Normalize chat responses from OpenAI-compatible backends."""
    if isinstance(response, str):
        stripped = response.strip().lower()
        if stripped.startswith("<html") or "<title>" in stripped:
            raise RuntimeError(
                "Received HTML instead of a chat completion JSON response. "
                "Check API_BASE (should usually end with /v1), API key, and model name."
            )
        return {"role": "assistant", "content": response, "tool_calls": []}

    if isinstance(response, dict):
        choices = response.get("choices") or []
        if choices and isinstance(choices[0], dict):
            message = choices[0].get("message") or {}
            return {
                "role": message.get("role", "assistant"),
                "content": message.get("content"),
                "tool_calls": message.get("tool_calls") or [],
            }

        content = response.get("content")
        return {
            "role": "assistant",
            "content": content if isinstance(content, str) else json.dumps(response, ensure_ascii=False),
            "tool_calls": [],
        }

    if hasattr(response, "choices") and response.choices:
        message = response.choices[0].message
        return {
            "role": getattr(message, "role", "assistant"),
            "content": getattr(message, "content", None),
            "tool_calls": getattr(message, "tool_calls", None) or [],
        }

    return {"role": "assistant", "content": str(response), "tool_calls": []}


def normalize_tool_calls(tool_calls: list[Any]) -> list[dict[str, Any]]:
    """Convert tool call objects to a consistent dictionary structure."""
    normalized: list[dict[str, Any]] = []

    for tool_call in tool_calls:
        if isinstance(tool_call, dict):
            fn = tool_call.get("function") or {}
            normalized.append(
                {
                    "id": tool_call.get("id"),
                    "type": tool_call.get("type", "function"),
                    "function": {
                        "name": fn.get("name"),
                        "arguments": fn.get("arguments", "{}"),
                    },
                }
            )
            continue

        fn_obj = getattr(tool_call, "function", None)
        normalized.append(
            {
                "id": getattr(tool_call, "id", None),
                "type": getattr(tool_call, "type", "function"),
                "function": {
                    "name": getattr(fn_obj, "name", None),
                    "arguments": getattr(fn_obj, "arguments", "{}"),
                },
            }
        )

    return normalized


def strip_think_blocks(text: str) -> str:
    """Remove visible chain-of-thought style tags from model output."""
    cleaned = text

    # Some providers leak traces as a prefix ending with </think> without an opening tag.
    if "</think>" in cleaned:
        cleaned = cleaned.split("</think>", 1)[1]

    # Remove any well-formed think blocks that may still appear.
    while "<think>" in cleaned and "</think>" in cleaned:
        start = cleaned.find("<think>")
        end = cleaned.find("</think>", start)
        if end == -1:
            break
        cleaned = cleaned[:start] + cleaned[end + len("</think>") :]

    # If an opening tag remains without a closer, drop that trailing segment.
    dangling_start = cleaned.find("<think>")
    if dangling_start != -1:
        cleaned = cleaned[:dangling_start]

    # Remove any dangling closing tags that may remain.
    cleaned = cleaned.replace("</think>", "")
    return cleaned.strip()


def mcp_content_to_text(content: list[Any]) -> str:
    """Convert mixed MCP content items to a single text payload."""
    rendered: list[str] = []
    for item in content:
        text_value = getattr(item, "text", None)
        if isinstance(text_value, str) and text_value.strip():
            rendered.append(text_value)
            continue

        model_dump = getattr(item, "model_dump", None)
        if callable(model_dump):
            rendered.append(json.dumps(model_dump(), ensure_ascii=False))
            continue

        rendered.append(str(item))
    return "\n".join(rendered)


def make_file_reader_mcp_server_params(mcp_base_dir: str | None = None) -> StdioServerParameters:
    """Build stdio parameters for the local sandboxed file-reader MCP server."""
    env = dict(os.environ)
    if mcp_base_dir:
        env["MCP_BASE_DIR"] = mcp_base_dir

    return StdioServerParameters(
        command=sys.executable,
        args=[str(FILE_READER_MCP_SCRIPT)],
        env=env,
    )


def make_wandb_mcp_server_params(
    wandb_entity: str | None = None,
    wandb_mcp_download_dir: str | None = None,
) -> StdioServerParameters:
    """Build stdio parameters for the local W&B MCP server."""
    env = dict(os.environ)
    if wandb_entity:
        env["WANDB_ENTITY"] = wandb_entity
    if wandb_mcp_download_dir:
        env["WANDB_MCP_DOWNLOAD_DIR"] = wandb_mcp_download_dir

    return StdioServerParameters(
        command=sys.executable,
        args=[str(WANDB_MCP_SCRIPT)],
        env=env,
    )


def make_git_mcp_server_params(git_repo_dir: str | None = None) -> StdioServerParameters:
    """Build stdio parameters for the local Git MCP server."""
    env = dict(os.environ)
    if git_repo_dir:
        env["GIT_MCP_REPO_DIR"] = git_repo_dir

    return StdioServerParameters(
        command=sys.executable,
        args=[str(GIT_MCP_SCRIPT)],
        env=env,
    )


class MiniMaxMCPClient:
    """Client wrapper for regular and MCP-tool-augmented queries."""

    def __init__(
        self,
        base_url: str,
        api_key: str,
        model_name: str,
        max_rounds: int = MAX_ROUNDS,
        mcp_base_dir: str | None = None,
        git_repo_dir: str | None = None,
        wandb_entity: str | None = None,
        wandb_mcp_download_dir: str | None = None,
        resolve_model_name: bool = False,
    ) -> None:
        """Initialize client configuration."""
        self.base_url = ensure_openai_base_url(base_url)
        self.api_key = api_key
        self.model_name = model_name
        self.max_rounds = max_rounds
        self.mcp_base_dir = str(Path(mcp_base_dir).resolve()) if mcp_base_dir else None
        self.git_repo_dir = (
            str(Path(git_repo_dir).resolve())
            if isinstance(git_repo_dir, str) and git_repo_dir.strip()
            else self.mcp_base_dir
        )
        self.wandb_entity = wandb_entity.strip() if isinstance(wandb_entity, str) and wandb_entity.strip() else None
        self.wandb_mcp_download_dir = (
            str(Path(wandb_mcp_download_dir).resolve())
            if isinstance(wandb_mcp_download_dir, str) and wandb_mcp_download_dir.strip()
            else None
        )
        # Generous timeout: slow/remote endpoints and streaming responses can take a while.
        self.client = OpenAI(api_key=api_key, base_url=self.base_url, timeout=300.0)
        self.model_name = model_name
        if resolve_model_name:
            self.model_name = self._resolve_model_name(model_name)

    def _list_available_model_ids(self) -> list[str]:
        """Return available model IDs from the configured endpoint."""
        try:
            models_response = self.client.models.list()
        except Exception as exc:
            print(f"⚠️ Could not fetch /v1/models: {exc}")
            return []

        data = getattr(models_response, "data", None)
        if not isinstance(data, list):
            return []

        model_ids: list[str] = []
        for model_item in data:
            if isinstance(model_item, Model):
                model_id = model_item.id
            else:
                model_id = getattr(model_item, "id", None)
            if isinstance(model_id, str) and model_id:
                model_ids.append(model_id)
        return model_ids

    def _resolve_model_name(self, preferred_model: str) -> str:
        """Ensure the selected model exists on the endpoint, else fallback safely."""
        available_model_ids = self._list_available_model_ids()
        if not available_model_ids:
            print(f"⚠️ Using configured model without validation: {preferred_model}")
            return preferred_model

        if preferred_model in available_model_ids:
            return preferred_model

        fallback_model = available_model_ids[0]
        print(
            "⚠️ Configured model is not available on this endpoint. "
            f"Configured='{preferred_model}', fallback='{fallback_model}'."
        )
        return fallback_model

    @staticmethod
    def _collect_stream(stream: Any) -> dict[str, Any]:
        """Accumulate a streaming chat completion into a message dict."""
        content_parts: list[str] = []
        tool_call_chunks: dict[int, dict[str, Any]] = {}

        for chunk in stream:
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta
            if delta.content:
                content_parts.append(delta.content)
            if delta.tool_calls:
                for tc in delta.tool_calls:
                    idx = tc.index
                    if idx not in tool_call_chunks:
                        tool_call_chunks[idx] = {
                            "id": "",
                            "type": "function",
                            "function": {"name": "", "arguments": ""},
                        }
                    if tc.id:
                        tool_call_chunks[idx]["id"] = tc.id
                    if tc.function:
                        if tc.function.name:
                            tool_call_chunks[idx]["function"]["name"] += tc.function.name
                        if tc.function.arguments:
                            tool_call_chunks[idx]["function"]["arguments"] += tc.function.arguments

        return {
            "role": "assistant",
            "content": "".join(content_parts) or None,
            "tool_calls": list(tool_call_chunks.values()),
        }

    def _create_chat_message(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """Create one assistant message, falling back to non-streaming on bad SSE chunks."""
        request: dict[str, Any] = {
            "model": self.model_name,
            "messages": messages,
        }
        if tools is not None:
            request["tools"] = tools

        try:
            stream = self.client.chat.completions.create(**request, stream=True)
            return self._collect_stream(stream)
        except json.JSONDecodeError:
            # Some OpenAI-compatible gateways occasionally emit empty/non-JSON SSE lines.
            response = self.client.chat.completions.create(**request, stream=False)
            return parse_chat_response(response)

    def make_query(self, query: str, system_prompt: str = DEFAULT_SYSTEM_PROMPT) -> str:
        """Run a standard LLM query without external tools (streaming)."""
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query},
        ]

        message = self._create_chat_message(messages)
        content = message.get("content")
        if isinstance(content, str):
            return strip_think_blocks(content)
        return ""

    async def _make_query_with_mcp_servers(
        self,
        query: str,
        mcp_servers: list[StdioServerParameters],
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    ) -> str:
        """Run a query with one or more MCP servers available as tool providers."""
        async with AsyncExitStack() as stack:
            sessions: list[ClientSession] = []
            tool_to_session: dict[str, ClientSession] = {}
            openai_tools: list[dict[str, Any]] = []

            for server_params in mcp_servers:
                with open(os.devnull, "w", encoding="utf-8") as silent_errlog:
                    read_stream, write_stream = await stack.enter_async_context(
                        stdio_client(server_params, errlog=silent_errlog)
                    )

                session = await stack.enter_async_context(ClientSession(read_stream, write_stream))
                await session.initialize()
                sessions.append(session)

                result = await session.list_tools()
                for tool in result.tools:
                    if tool.name in tool_to_session:
                        raise RuntimeError(f"Duplicate MCP tool name detected: '{tool.name}'.")
                    tool_to_session[tool.name] = session
                    openai_tools.append(
                        {
                            "type": "function",
                            "function": {
                                "name": tool.name,
                                "description": tool.description,
                                "parameters": tool.inputSchema,
                            },
                        }
                    )

            if not sessions:
                raise RuntimeError("No MCP servers configured for this query.")

            messages: list[dict[str, Any]] = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": query},
            ]

            message: dict[str, Any] = {"content": None}
            for round_idx in range(self.max_rounds):
                if round_idx == self.max_rounds - 1:
                    messages.append(
                        {
                            "role": "user",
                            "content": "Maximum tool call rounds reached. Provide the final concise answer using the information gathered so far.",
                        }
                    )
                    openai_tools = []

                message = self._create_chat_message(messages=messages, tools=openai_tools)
                tool_calls = normalize_tool_calls(message.get("tool_calls") or [])

                if tool_calls:
                    messages.append(
                        {
                            "role": "assistant",
                            "content": message.get("content"),
                            "tool_calls": tool_calls,
                        }
                    )

                    for idx, tool_call in enumerate(tool_calls):
                        tool_call_id = tool_call.get("id") or f"tool_call_{round_idx}_{idx}"
                        fn_name = tool_call["function"].get("name")
                        raw_args = tool_call["function"].get("arguments")

                        if not fn_name or fn_name not in tool_to_session:
                            tool_output = f"Error: Unknown tool requested: {fn_name!r}."
                        else:
                            try:
                                fn_args = json.loads(raw_args) if isinstance(raw_args, str) else (raw_args or {})
                            except json.JSONDecodeError as exc:
                                fn_args = {}
                                tool_output = f"Error: Invalid JSON arguments for tool '{fn_name}': {exc}"
                            else:
                                try:
                                    session = tool_to_session[fn_name]
                                    mcp_result = await session.call_tool(fn_name, arguments=fn_args)
                                    tool_output = mcp_content_to_text(mcp_result.content)
                                except Exception as exc:
                                    tool_output = f"Error: Tool '{fn_name}' failed: {exc}"

                        messages.append(
                            {
                                "role": "tool",
                                "tool_call_id": tool_call_id,
                                "content": tool_output,
                            }
                        )
                    continue

                content = message.get("content")
                final_text = strip_think_blocks(content) if isinstance(content, str) else ""
                if final_text:
                    return final_text

                if round_idx < self.max_rounds - 1:
                    messages.append(
                        {
                            "role": "user",
                            "content": "Use the gathered tool results and provide the final concise answer now.",
                        }
                    )

            fallback = message.get("content")
            return strip_think_blocks(fallback) if isinstance(fallback, str) else ""

    def make_query_with_websearch(
        self,
        query: str,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    ) -> str:
        """Run an LLM query with MCP web-search tool calling support."""
        return asyncio.run(
            self._make_query_with_mcp_servers(
                query=query,
                mcp_servers=[WEB_SEARCH_MCP_SERVER_PARAMS],
                system_prompt=system_prompt,
            )
        )

    def make_query_with_fileaccess(
        self,
        query: str,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    ) -> str:
        """Run an LLM query with sandboxed file-access MCP tools only."""
        return asyncio.run(
            self._make_query_with_mcp_servers(
                query=query,
                mcp_servers=[make_file_reader_mcp_server_params(self.mcp_base_dir)],
                system_prompt=system_prompt,
            )
        )

    def make_query_with_wandb(
        self,
        query: str,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    ) -> str:
        """Run an LLM query with W&B MCP tools only."""
        return asyncio.run(
            self._make_query_with_mcp_servers(
                query=query,
                mcp_servers=[
                    make_wandb_mcp_server_params(
                        wandb_entity=self.wandb_entity,
                        wandb_mcp_download_dir=self.wandb_mcp_download_dir,
                    )
                ],
                system_prompt=system_prompt,
            )
        )

    def make_query_with_git(
        self,
        query: str,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    ) -> str:
        """Run an LLM query with Git MCP tools only."""
        return asyncio.run(
            self._make_query_with_mcp_servers(
                query=query,
                mcp_servers=[make_git_mcp_server_params(self.git_repo_dir)],
                system_prompt=system_prompt,
            )
        )

    def make_query_with_alltools(
        self,
        query: str,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    ) -> str:
        """Run an LLM query with web-search, file-access, W&B, and Git MCP tools."""
        return asyncio.run(
            self._make_query_with_mcp_servers(
                query=query,
                mcp_servers=[
                    WEB_SEARCH_MCP_SERVER_PARAMS,
                    make_file_reader_mcp_server_params(self.mcp_base_dir),
                    make_wandb_mcp_server_params(
                        wandb_entity=self.wandb_entity,
                        wandb_mcp_download_dir=self.wandb_mcp_download_dir,
                    ),
                    make_git_mcp_server_params(self.git_repo_dir),
                ],
                system_prompt=system_prompt,
            )
        )


def main():
    if API_KEY is None:
        raise RuntimeError("BLABLADOR_KEY is not set.")

    query = "What is the weather like today in Heidelberg?"
    llm_client = MiniMaxMCPClient(
        base_url=API_BASE,
        api_key=API_KEY,
        model_name=MODEL_NAME,
        mcp_base_dir="/home/zimmerer/ws/chexclip",
        wandb_entity="dzimmererdkfz-dkfz-german-cancer-research-center",
    )

    print(f"\n🤖 Asking MiniMax: '{query}'...")
    print(f"🌐 Using API base: {llm_client.base_url}")

    print("🔎 Running query without web search...")
    answer = llm_client.make_query(query)

    if answer:
        print("\n📝 Final Answer without web search:")
        print(answer)

    time.sleep(2)  # Pause before the next query for clarity

    print("🔎 Running query with web search...")
    answer = llm_client.make_query_with_websearch(query)

    if answer:
        print("\n📝 Final Answer with web search:")
        print(answer)
        # return

    time.sleep(2)  # Pause before the next query for clarity

    print("🔎 Running query with file access...")
    answer = llm_client.make_query_with_fileaccess("What is the pretraining of the image_encoder in the config yaml?")

    if answer:
        print("\n📝 Final Answer with file access:")
        print(answer)
        # return

    time.sleep(2)  # Pause before the next query for clarity

    print("🔎 Running query with W&B tools...")
    wandb_query = (
        "Use the W&B MCP tools to inspect this run and report only the minimum validation loss value and the step it occurred at. "
        "Project: chexclip-sciantistD05, Run ID: 48175890. "
    )
    answer = llm_client.make_query_with_wandb(wandb_query)

    if answer:
        print("\n📝 Final Answer with W&B tools:")
        print(answer)
        # return

    time.sleep(2)  # Pause before the next query for clarity

    print("🔎 Running query with Git MCP tools...")
    answer = llm_client.make_query_with_git("What was the last change in the config yaml?")

    if answer:
        print("\n📝 Final Answer with Git MCP tools:")
        print(answer)
        # return

    # print("\n⚠️ No final answer produced.")


if __name__ == "__main__":
    if not API_KEY:
        print("❌ Error: Please set BLABLADOR_KEY environment variable.")
    else:
        main()
