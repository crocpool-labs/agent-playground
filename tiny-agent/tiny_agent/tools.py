"""两个明确注册的工具；计算器只解释算术 AST，不执行 Python 代码。"""
import ast
import json
import math
import operator

from .config import ROOT

OPERATORS = {ast.Add: operator.add, ast.Sub: operator.sub,
             ast.Mult: operator.mul, ast.Div: operator.truediv}


def calculator(expression: str) -> str:
    if not isinstance(expression, str) or not 1 <= len(expression) <= 128:
        raise ValueError("expression 必须是 1–128 字符的字符串")
    tree = ast.parse(expression, mode="eval")
    if sum(1 for _ in ast.walk(tree)) > 40:
        raise ValueError("表达式过长")

    def visit(node):
        if isinstance(node, ast.Constant) and type(node.value) in (int, float):
            value = node.value
        elif isinstance(node, ast.BinOp) and type(node.op) in OPERATORS:
            value = OPERATORS[type(node.op)](visit(node.left), visit(node.right))
        elif isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            value = visit(node.operand) * (-1 if isinstance(node.op, ast.USub) else 1)
        else:
            raise ValueError("只支持数字、括号和 + - * /")
        if not math.isfinite(value) or abs(value) > 1e12:
            raise ValueError("计算结果超出范围")
        return value

    result = visit(tree.body)
    return format(result, ".12g")


def search_notes(query: str) -> str:
    if not isinstance(query, str) or not 1 <= len(query) <= 100:
        raise ValueError("query 必须是 1–100 字符的字符串")
    notes = json.loads((ROOT / "notes.json").read_text())
    q = query.strip().casefold()
    if not q:
        raise ValueError("query 不能为空白")
    exact = [n for n in notes if n["topic"].casefold() == q]
    hits = exact or [n for n in notes if q in (n["topic"] + n["text"]).casefold()]
    return "\n".join(n["text"] for n in hits[:3]) or "没有找到相关学习笔记。"


def execute(action: dict) -> str:
    """拒绝未注册工具、额外参数及错误参数类型。"""
    registry = {"calculator": (calculator, "expression"), "search_notes": (search_notes, "query")}
    name, args = action.get("tool"), action.get("args")
    if not isinstance(name, str) or name not in registry:
        raise ValueError("未知工具")
    fn, key = registry[name]
    if not isinstance(args, dict) or set(args) != {key}:
        raise ValueError(f"{name} 只接受参数 {key}")
    return fn(args[key])
