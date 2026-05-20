"""Tests for llmclient.minimax_client module."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest


class TestEnsureOpenaiBaseUrl:
    """Test ensure_openai_base_url function."""

    def test_already_has_v1(self) -> None:
        from llmclient.minimax_client import ensure_openai_base_url
        result = ensure_openai_base_url("https://api.example.com/v1")
        assert result == "https://api.example.com/v1"

    def test_adds_v1_suffix(self) -> None:
        from llmclient.minimax_client import ensure_openai_base_url
        result = ensure_openai_base_url("https://api.example.com")
        assert result == "https://api.example.com/v1"

    def test_strips_trailing_slash(self) -> None:
        from llmclient.minimax_client import ensure_openai_base_url
        result = ensure_openai_base_url("https://api.example.com/")
        assert result == "https://api.example.com/v1"


class TestParseChatResponse:
    """Test parse_chat_response function."""

    def test_string_response_returns_assistant_message(self) -> None:
        from llmclient.minimax_client import parse_chat_response
        result = parse_chat_response("Hello world")
        assert result["role"] == "assistant"
        assert result["content"] == "Hello world"
        assert result["tool_calls"] == []

    def test_html_response_raises_error(self) -> None:
        from llmclient.minimax_client import parse_chat_response
        with pytest.raises(RuntimeError, match="HTML instead"):
            parse_chat_response("<html><body>Error</body></html>")

    def test_html_title_raises_error(self) -> None:
        from llmclient.minimax_client import parse_chat_response
        with pytest.raises(RuntimeError, match="HTML instead"):
            parse_chat_response("<html><title>Error</title></html>")

    def test_dict_with_choices(self) -> None:
        from llmclient.minimax_client import parse_chat_response
        response = {
            "choices": [
                {"message": {"role": "user", "content": "Hello"}}
            ]
        }
        result = parse_chat_response(response)
        assert result["role"] == "user"
        assert result["content"] == "Hello"

    def test_dict_with_content_only(self) -> None:
        from llmclient.minimax_client import parse_chat_response
        response = {"content": "Hello"}
        result = parse_chat_response(response)
        assert result["content"] == "Hello"

    def test_object_with_choices_attribute(self) -> None:
        from llmclient.minimax_client import parse_chat_response
        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content="Hello", role="assistant", tool_calls=None))]
        result = parse_chat_response(mock_response)
        assert result["content"] == "Hello"

    def test_fallback_to_str(self) -> None:
        from llmclient.minimax_client import parse_chat_response
        result = parse_chat_response(12345)
        assert result["content"] == "12345"


class TestNormalizeToolCalls:
    """Test normalize_tool_calls function."""

    def test_empty_list(self) -> None:
        from llmclient.minimax_client import normalize_tool_calls
        result = normalize_tool_calls([])
        assert result == []

    def test_dict_tool_call(self) -> None:
        from llmclient.minimax_client import normalize_tool_calls
        tool_calls = [
            {
                "id": "call_123",
                "type": "function",
                "function": {"name": "test", "arguments": "{}"}
            }
        ]
        result = normalize_tool_calls(tool_calls)
        assert len(result) == 1
        assert result[0]["id"] == "call_123"
        assert result[0]["function"]["name"] == "test"

    def test_object_tool_call(self) -> None:
        from llmclient.minimax_client import normalize_tool_calls
        mock_tool = MagicMock()
        mock_tool.id = "call_123"
        mock_tool.type = "function"
        mock_tool.function.name = "test"
        mock_tool.function.arguments = "{}"
        result = normalize_tool_calls([mock_tool])
        assert len(result) == 1
        assert result[0]["id"] == "call_123"


class TestStripThinkBlocks:
    """Test strip_think_blocks function."""

    def test_no_think_blocks(self) -> None:
        from llmclient.minimax_client import strip_think_blocks
        result = strip_think_blocks("Hello world")
        assert result == "Hello world"

    def test_removes_think_block(self) -> None:
        from llmclient.minimax_client import strip_think_blocks
        result = strip_think_blocks("Some text<think>think content</think> more text")
        assert "think content" not in result
        assert "more text" in result

    def test_removes_prefix_think_without_opening(self) -> None:
        from llmclient.minimax_client import strip_think_blocks
        result = strip_think_blocks("Some text</think>only closing")
        assert "only closing" in result
        assert "think content" not in result

    def test_removes_dangling_opening_tag(self) -> None:
        from llmclient.minimax_client import strip_think_blocks
        result = strip_think_blocks("Hello<think> world")
        assert result == "Hello"

    def test_removes_dangling_closing_tag(self) -> None:
        from llmclient.minimax_client import strip_think_blocks
        result = strip_think_blocks("Hello</think> world")
        assert "world" in result


class TestMcpContentToText:
    """Test mcp_content_to_text function."""

    def test_empty_list(self) -> None:
        from llmclient.minimax_client import mcp_content_to_text
        result = mcp_content_to_text([])
        assert result == ""

    def test_extracts_text_attribute(self) -> None:
        from llmclient.minimax_client import mcp_content_to_text
        mock_item = MagicMock()
        mock_item.text = "Hello world"
        result = mcp_content_to_text([mock_item])
        assert result == "Hello world"

    def test_uses_model_dump_json(self) -> None:
        from llmclient.minimax_client import mcp_content_to_text
        mock_item = MagicMock()
        mock_item.text = None
        mock_item.model_dump.return_value = {"key": "value"}
        result = mcp_content_to_text([mock_item])
        assert "key" in result

    def test_falls_back_to_str(self) -> None:
        from llmclient.minimax_client import mcp_content_to_text
        mock_item = MagicMock()
        mock_item.text = None
        mock_item.model_dump = None
        result = mcp_content_to_text([mock_item])
        assert result == str(mock_item)

    def test_multiple_items_concatenated(self) -> None:
        from llmclient.minimax_client import mcp_content_to_text

        class MockItem:
            def __init__(self, text):
                self.text = text

        result = mcp_content_to_text([MockItem("Hello"), MockItem(" World")])
        assert result == "Hello\n World"


class TestMakeFileReaderMcpServerParams:
    """Test make_file_reader_mcp_server_params function."""

    def test_returns_stdio_params(self) -> None:
        from llmclient.minimax_client import make_file_reader_mcp_server_params
        result = make_file_reader_mcp_server_params()
        assert result.command is not None
        assert any("file_reader_mcp.py" in str(a) for a in result.args)

    def test_sets_mcp_base_dir(self) -> None:
        from llmclient.minimax_client import make_file_reader_mcp_server_params
        result = make_file_reader_mcp_server_params(mcp_base_dir="/tmp/test")
        assert result.env is not None
        assert result.env.get("MCP_BASE_DIR") == "/tmp/test"


class TestMakeWandbMcpServerParams:
    """Test make_wandb_mcp_server_params function."""

    def test_returns_stdio_params(self) -> None:
        from llmclient.minimax_client import make_wandb_mcp_server_params
        result = make_wandb_mcp_server_params()
        assert result.command is not None
        assert any("wandb_mcp.py" in str(a) for a in result.args)

    def test_sets_wandb_entity(self) -> None:
        from llmclient.minimax_client import make_wandb_mcp_server_params
        result = make_wandb_mcp_server_params(wandb_entity="test_entity")
        assert result.env.get("WANDB_ENTITY") == "test_entity"

    def test_sets_download_dir(self) -> None:
        from llmclient.minimax_client import make_wandb_mcp_server_params
        result = make_wandb_mcp_server_params(wandb_mcp_download_dir="/tmp/downloads")
        assert result.env.get("WANDB_MCP_DOWNLOAD_DIR") == "/tmp/downloads"


class TestMakeGitMcpServerParams:
    """Test make_git_mcp_server_params function."""

    def test_returns_stdio_params(self) -> None:
        from llmclient.minimax_client import make_git_mcp_server_params
        result = make_git_mcp_server_params()
        assert result.command is not None
        assert any("git_mcp.py" in str(a) for a in result.args)

    def test_sets_git_repo_dir(self) -> None:
        from llmclient.minimax_client import make_git_mcp_server_params
        result = make_git_mcp_server_params(git_repo_dir="/tmp/repo")
        assert result.env.get("GIT_MCP_REPO_DIR") == "/tmp/repo"


class TestMiniMaxMCPClient:
    """Test MiniMaxMCPClient class."""

    def test_initialization(self) -> None:
        from llmclient.minimax_client import MiniMaxMCPClient
        client = MiniMaxMCPClient(
            base_url="https://api.minimax.chat",
            api_key="test_key",
            model_name="test-model"
        )
        assert client.base_url.endswith("/v1")
        assert client.api_key == "test_key"
        assert client.model_name == "test-model"

    def test_initialization_with_trailing_slash(self) -> None:
        from llmclient.minimax_client import MiniMaxMCPClient
        client = MiniMaxMCPClient(
            base_url="https://api.minimax.chat/",
            api_key="test_key",
            model_name="test-model"
        )
        assert client.base_url == "https://api.minimax.chat/v1"

    def test_initialization_preserves_mcp_base_dir(self) -> None:
        from llmclient.minimax_client import MiniMaxMCPClient
        client = MiniMaxMCPClient(
            base_url="https://api.minimax.chat",
            api_key="test_key",
            model_name="test-model",
            mcp_base_dir="/tmp/mcp"
        )
        assert client.mcp_base_dir == "/tmp/mcp"

    def test_git_repo_dir_defaults_to_mcp_base_dir(self) -> None:
        from llmclient.minimax_client import MiniMaxMCPClient
        client = MiniMaxMCPClient(
            base_url="https://api.minimax.chat",
            api_key="test_key",
            model_name="test-model",
            mcp_base_dir="/tmp/mcp"
        )
        assert client.git_repo_dir == "/tmp/mcp"

    def test_git_repo_dir_can_be_overridden(self) -> None:
        from llmclient.minimax_client import MiniMaxMCPClient
        client = MiniMaxMCPClient(
            base_url="https://api.minimax.chat",
            api_key="test_key",
            model_name="test-model",
            mcp_base_dir="/tmp/mcp",
            git_repo_dir="/tmp/git"
        )
        assert client.git_repo_dir == "/tmp/git"

    def test_wandb_entity_stripped(self) -> None:
        from llmclient.minimax_client import MiniMaxMCPClient
        client = MiniMaxMCPClient(
            base_url="https://api.minimax.chat",
            api_key="test_key",
            model_name="test-model",
            wandb_entity="  test_entity  "
        )
        assert client.wandb_entity == "test_entity"

    def test_empty_wandb_entity_stored_as_none(self) -> None:
        from llmclient.minimax_client import MiniMaxMCPClient
        client = MiniMaxMCPClient(
            base_url="https://api.minimax.chat",
            api_key="test_key",
            model_name="test-model",
            wandb_entity="   "
        )
        assert client.wandb_entity is None


class TestMiniMaxMCPClientListAvailableModels:
    """Test _list_available_model_ids method."""

    def test_returns_empty_on_exception(self) -> None:
        from llmclient.minimax_client import MiniMaxMCPClient
        client = MiniMaxMCPClient(
            base_url="https://api.minimax.chat",
            api_key="test_key",
            model_name="test-model"
        )
        with patch.object(client.client.models, "list", side_effect=Exception("API Error")):
            result = client._list_available_model_ids()
            assert result == []

    def test_returns_model_ids(self) -> None:
        from llmclient.minimax_client import MiniMaxMCPClient, Model
        client = MiniMaxMCPClient(
            base_url="https://api.minimax.chat",
            api_key="test_key",
            model_name="test-model"
        )
        mock_model = MagicMock(spec=Model)
        mock_model.id = "test-model"
        mock_response = MagicMock()
        mock_response.data = [mock_model]
        with patch.object(client.client.models, "list", return_value=mock_response):
            result = client._list_available_model_ids()
            assert "test-model" in result


class TestMiniMaxMCPClientResolveModelName:
    """Test _resolve_model_name method."""

    def test_returns_preferred_if_available(self) -> None:
        from llmclient.minimax_client import MiniMaxMCPClient
        client = MiniMaxMCPClient(
            base_url="https://api.minimax.chat",
            api_key="test_key",
            model_name="test-model",
            resolve_model_name=True
        )
        with patch.object(client, "_list_available_model_ids", return_value=["test-model", "other-model"]):
            result = client._resolve_model_name("test-model")
            assert result == "test-model"

    def test_falls_back_to_first_available(self) -> None:
        from llmclient.minimax_client import MiniMaxMCPClient
        client = MiniMaxMCPClient(
            base_url="https://api.minimax.chat",
            api_key="test_key",
            model_name="test-model",
            resolve_model_name=True
        )
        with patch.object(client, "_list_available_model_ids", return_value=["available-model"]):
            result = client._resolve_model_name("unavailable-model")
            assert result == "available-model"