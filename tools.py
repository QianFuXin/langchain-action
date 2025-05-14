"""
工具集
"""
import io
import sys

from langchain_core.tools import tool


@tool
def get_user_info_by_id(user_id: str) -> str:
    """根据用户id查询用户信息

    Args:
        user_id: 用户id
    Returns:
        用户信息
    """
    return "姓名：张三\n性别：男\n工资：30001"


@tool
def get_current_date() -> str:
    """获取当前的年月日

    Returns:
        当前日期字符串，格式为 YYYY-MM-DD
    """
    from datetime import datetime
    return datetime.now().strftime("%Y-%m-%d")


@tool
def execute_python_code(code) -> str:
    """执行python代码

    Args:
        code: python代码
    Returns:
        执行结果
    """
    old_stdout = sys.stdout
    redirected_output = io.StringIO()
    sys.stdout = redirected_output
    try:
        print(code)
        exec(code, {})  # 执行代码
    except Exception as e:
        return f"Error: {str(e)}"
    finally:
        sys.stdout = old_stdout
    return redirected_output.getvalue()


@tool
def read_file(file_path: str) -> str:
    """读取指定路径的文件内容

    Args:
        file_path (str): 文件的路径

    Returns:
        str: 文件的内容，若文件不存在则返回错误信息
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read()
    except FileNotFoundError:
        return f"错误：文件 '{file_path}' 不存在"
    except Exception as e:
        return f"读取文件时发生错误: {str(e)}"


if __name__ == '__main__':
    pass
