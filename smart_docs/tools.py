from __future__ import annotations

import ast
import operator
from datetime import datetime, timedelta, timezone as datetime_timezone
from typing import TYPE_CHECKING, Any
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

if TYPE_CHECKING:
    from smart_docs.rag import RAGService


class CalculatorError(ValueError):
    pass


ALLOWED_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}


def calculator(expression: str) -> float:
    expression = expression.replace("$", "").replace(",", "")
    expression = expression.replace("%", "/100")
    try:
        tree = ast.parse(expression, mode="eval")
        return float(_eval_node(tree.body))
    except Exception as exc:
        raise CalculatorError(f"Could not evaluate expression: {expression}") from exc


def current_datetime(timezone: str = "Asia/Kolkata") -> str:
    try:
        tzinfo = ZoneInfo(timezone)
    except ZoneInfoNotFoundError:
        tzinfo = datetime_timezone(timedelta(hours=5, minutes=30), name="IST")
    return datetime.now(tzinfo).isoformat(timespec="seconds")


def search_documents(rag: "RAGService", query: str, top_k: int) -> list[dict[str, Any]]:
    return rag.search(query, top_k=top_k)


def _eval_node(node: ast.AST) -> float:
    if isinstance(node, ast.Constant) and isinstance(node.value, int | float):
        return float(node.value)
    if isinstance(node, ast.BinOp) and type(node.op) in ALLOWED_OPERATORS:
        return ALLOWED_OPERATORS[type(node.op)](_eval_node(node.left), _eval_node(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in ALLOWED_OPERATORS:
        return ALLOWED_OPERATORS[type(node.op)](_eval_node(node.operand))
    raise CalculatorError("Only numeric expressions are allowed.")
