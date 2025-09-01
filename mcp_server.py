from mcp.server.fastmcp import FastMCP

# Create an MCP server
mcp = FastMCP("Demo", port=8001)


# Add an addition tool
@mcp.tool()
def add(a: int, b: int) -> int:
    """Add two numbers"""
    return a + b


@mcp.resource("user://{user_id}")
def get_user(user_id: str) -> dict:
    """提供用户信息资源"""
    # 模拟数据库返回
    users = {
        "1001": {"name": "Alice", "age": 25},
        "1002": {"name": "Bob", "age": 30},
    }
    return users.get(user_id, {"name": "Guest", "age": 0})


@mcp.prompt("welcome_message")
def welcome_prompt(user_name: str) -> str:
    """生成欢迎语模板"""
    return f"请生成一段热情的欢迎语，欢迎用户 {user_name} 加入我们的平台。"


if __name__ == "__main__":
    mcp.run(transport="streamable-http")
