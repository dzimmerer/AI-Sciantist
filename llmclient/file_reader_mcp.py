import os
from pathlib import Path

from mcp.server.fastmcp import FastMCP

# Resolve once for strict, absolute path comparisons.
BASE_DIR = Path(os.getenv("MCP_BASE_DIR", ".")).resolve()

mcp = FastMCP("Sandboxed File Reader")


def get_safe_path(requested_path: str) -> Path:
    """Resolve a path under BASE_DIR and deny traversal outside the sandbox."""
    clean_path = requested_path.lstrip(os.sep)
    target_path = (BASE_DIR / clean_path).resolve()

    if not target_path.is_relative_to(BASE_DIR):
        raise ValueError("Security Error: Access denied to path outside of base directory.")

    return target_path


@mcp.tool()
def list_directory(path: str = "") -> list[str]:
    """List files and folders relative to the base directory."""
    try:
        safe_path = get_safe_path(path)
        if not safe_path.is_dir():
            return [f"Error: '{path}' is not a valid directory."]

        return [str(p.relative_to(BASE_DIR)) for p in safe_path.iterdir()]
    except Exception as exc:
        return [f"Error: {exc}"]


@mcp.tool()
def read_file(path: str) -> str:
    """Read a UTF-8 text file relative to the base directory."""
    try:
        safe_path = get_safe_path(path)
        if not safe_path.is_file():
            return f"Error: '{path}' is not a valid file."

        with open(safe_path, "r", encoding="utf-8") as file_handle:
            return file_handle.read()
    except Exception as exc:
        return f"Error: {exc}"


if __name__ == "__main__":
    mcp.run()
