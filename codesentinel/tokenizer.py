"""Step 1 of MOSS-style fingerprinting: a small lexer.

Source code is split into tokens and *normalised* so that cosmetic changes do not
hide copying:

    - variable / function names  ->  ID
    - numbers                    ->  NUM
    - strings                    ->  STR
    - whitespace and comments    ->  dropped

So ``sum += arr[i]`` and ``total += nums[j]`` both become ``ID += ID [ ID ]``.

The lexer is a tiny DFA: look at the first character to pick a state (identifier,
number, string, comment, operator) and consume characters until that state ends.
"""

from __future__ import annotations

import keyword
from dataclasses import dataclass

# Keywords are kept verbatim: ``for`` cannot be renamed.
KEYWORDS = set(keyword.kwlist)

# Builtins and common methods are kept verbatim as well, but only when they are
# actually called (``len(...)``, ``.append(...)``). A candidate cannot rename those,
# so they reveal real structure. A *variable* called ``sum`` stays a normal ID.
KEEP_NAMES = {
    "len",
    "range",
    "print",
    "input",
    "int",
    "str",
    "float",
    "list",
    "dict",
    "set",
    "tuple",
    "max",
    "min",
    "sum",
    "abs",
    "sorted",
    "reversed",
    "enumerate",
    "zip",
    "map",
    "filter",
    "append",
    "pop",
    "get",
    "add",
    "remove",
    "split",
    "strip",
    "join",
    "items",
    "keys",
    "values",
    "readline",
    "stdin",
    "sys",
    "self",
    "ord",
    "chr",
    "index",
    "count",
    "sort",
    "isalnum",
    "lower",
    "__name__",
    "__main__",
}

# Longest operators first so that ``**=`` is not read as ``*`` ``*`` ``=``.
OPERATORS = sorted(
    [
        "**=",
        "//=",
        ">>=",
        "<<=",
        "->",
        ":=",
        "==",
        "!=",
        "<=",
        ">=",
        "+=",
        "-=",
        "*=",
        "/=",
        "%=",
        "&=",
        "|=",
        "^=",
        "**",
        "//",
        "<<",
        ">>",
        "+",
        "-",
        "*",
        "/",
        "%",
        "=",
        "<",
        ">",
        "&",
        "|",
        "^",
        "~",
        "(",
        ")",
        "[",
        "]",
        "{",
        "}",
        ",",
        ":",
        ".",
        ";",
        "@",
    ],
    key=len,
    reverse=True,
)

STRING_PREFIXES = {"f", "r", "b", "u", "rb", "br", "fr", "rf"}


@dataclass
class Token:
    kind: str  # ID / NUM / STR / KEYWORD / OP
    text: str  # original text, e.g. "total"
    norm: str  # normalised form, e.g. "ID"
    line: int  # source line (used to highlight matched lines)


def read_string(code: str, i: int) -> int:
    """``code[i]`` is an opening quote. Return the index just after the closing quote."""
    quote = code[i] * 3 if code[i : i + 3] in ('"""', "'''") else code[i]
    j = i + len(quote)
    while j < len(code):
        if code[j] == "\\":
            j += 2
            continue
        if code.startswith(quote, j):
            return j + len(quote)
        j += 1
    return j


def tokenize(code: str) -> list[Token]:
    tokens: list[Token] = []
    i = 0
    line = 1

    while i < len(code):
        ch = code[i]

        # state: whitespace -> skip
        if ch.isspace():
            if ch == "\n":
                line += 1
            i += 1

        # state: comment -> skip to end of line
        elif ch == "#":
            while i < len(code) and code[i] != "\n":
                i += 1

        # state: string
        elif ch in "\"'":
            end = read_string(code, i)
            text = code[i:end]
            tokens.append(Token("STR", text, "STR", line))
            line += text.count("\n")
            i = end

        # state: identifier / keyword
        elif ch.isalpha() or ch == "_":
            j = i
            while j < len(code) and (code[j].isalnum() or code[j] == "_"):
                j += 1
            word = code[i:j]

            if word.lower() in STRING_PREFIXES and j < len(code) and code[j] in "\"'":
                # f"..." / r"..." is a string literal, not an identifier
                end = read_string(code, j)
                text = code[i:end]
                tokens.append(Token("STR", text, "STR", line))
                line += text.count("\n")
                i = end
                continue

            if word in KEYWORDS:
                tokens.append(Token("KEYWORD", word, word, line))
            else:
                tokens.append(Token("ID", word, "ID", line))
            i = j

        # state: number
        elif ch.isdigit():
            j = i
            while j < len(code) and (code[j].isalnum() or code[j] in "._"):
                j += 1
            tokens.append(Token("NUM", code[i:j], "NUM", line))
            i = j

        # state: operator / punctuation
        else:
            for op in OPERATORS:
                if code.startswith(op, i):
                    tokens.append(Token("OP", op, op, line))
                    i += len(op)
                    break
            else:
                i += 1  # unknown character: ignore

    # Keep builtin names only when they are called (``len(``) or used as a method (``.append``).
    for k, t in enumerate(tokens):
        if t.kind == "ID" and t.text in KEEP_NAMES:
            called = k + 1 < len(tokens) and tokens[k + 1].text == "("
            method = k > 0 and tokens[k - 1].text == "."
            if called or method or t.text.startswith("__"):
                t.norm = t.text

    return tokens


def normalize(code: str) -> str:
    return " ".join(t.norm for t in tokenize(code))
