import re
import os
from difflib import SequenceMatcher
from typing import Optional

# Logic parsing
from lark import Lark, Transformer

# Symbolic math
from sympy.logic.boolalg import BooleanFalse, BooleanTrue, to_cnf, simplify_logic, And, Or, Not, Equivalent, Implies
from sympy import symbols

LOGIC_GRAMMAR = r"""
    ?start: expr

    ?expr: call
         | atom

    call: OP "(" [expr ("," expr)*] ")"

    atom : NAME                                          -> var
    OP   : "Equivalent" | "Implies" | "And" | "Or" | "Not"
    NAME : /[A-Za-z][A-Za-z0-9_]*/

    %ignore " "
    %ignore "\t"
    %ignore "\n"
"""


class LogicTransformer(Transformer):

    def __init__(self, var_map: dict):
        super().__init__()
        self.var_map = var_map     # tên biến → sympy Symbol

    def _get_sym(self, name: str):
        if name not in self.var_map:
            self.var_map[name] = symbols(name)
        return self.var_map[name]

    def var(self, items):
        return self._get_sym(str(items[0]))

    def call(self, items):
        op = str(items[0])
        args = list(items[1:])
        if op == "Not" and len(args) == 1:
            return Not(args[0])
        if op == "Or" and len(args) >= 2:
            return Or(*args)
        if op == "And" and len(args) >= 2:
            return And(*args)
        if op == "Implies" and len(args) == 2:
            return Implies(args[0], args[1])
        if op == "Equivalent" and len(args) == 2:
            return Equivalent(args[0], args[1])
        raise ValueError(f"Invalid {op} arity: expected valid args, got {len(args)}")

class Lang2Logic:

    # Prompt thiết kế để GPT output đúng format Or/And/Not
    SYSTEM_PROMPT = """You are a logic translator. Convert the given English sentence into a propositional logic expression.

Rules:
1. Use ONLY these 5 operators, no exceptions:
   Or(X, Y), And(X, Y), Not(X), Implies(X, Y), Equivalent(X, Y)
   Any expression using other operators (Xor, Nor, Nand, Xnor, ...) must be rewritten using only these 5.
2. Use single uppercase letters as variables (A, B, C, ...).
3. Output EXACTLY two lines, no extra text:
   Line 1: the logical expression
   Line 2: mapping of every variable used, format: X="proposition", Y="proposition"
4. No explanation, no markdown.
5. Always define variables as POSITIVE propositions. Never use
   'not', 'does not', 'cannot', 'no' in a variable's meaning.
   Use the Not() operator to express negation instead.


Examples:
Input: "A or B"
Output:
Or(A, B)
A="A", B="B"

Input: "If A then B"
Output:
Implies(A, B)
A="A", B="B"

Input: "Not A and B"
Output:
And(Not(A), B)
A="A", B="B"

Input: "The circus does not have a carousel if and only if it has a ferris wheel."
Output:
Equivalent(Not(C), F)
C="The circus has a carousel", F="The circus has a ferris wheel"

Input: "Either A or B, but not both."
Wrong output:
Xor(A, B)

Correct output:
And(Or(A, B), Not(And(A, B)))
A="A", B="B"
"""

    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-4o-mini"):
        self.model = model
        self._api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        self._client = None
        self._parser = Lark(LOGIC_GRAMMAR, parser="earley", ambiguity="resolve")
        self._var_map: dict = {}
        self._meaning_map: dict = {}   # tên biến → nghĩa tiếng Anh
        self._api_cost_tokens = 0   # tracking token usage

    def _get_client(self):
        if not self._api_key:
            raise ValueError(
                "OPENAI_API_KEY is required for English text -> logic translation. "
                "Use parse_expression() for already-formatted logic expressions."
            )
        if self._client is None:
            try:
                from openai import OpenAI
                self._client = OpenAI(api_key=self._api_key)
            except ImportError:
                raise ImportError("Cần cài đặt: pip install openai")
        return self._client

    # ---- Step 1: Tokenize ----

    def tokenize(self, text: str) -> list[str]:
        # Giới hạn 450 từ theo bài báo
        words = text.split()
        if len(words) > 450:
            text = " ".join(words[:450])
        try:
            import nltk
            nltk.download("punkt", quiet=True)
            nltk.download("punkt_tab", quiet=True)
            from nltk.tokenize import sent_tokenize
            sentences = sent_tokenize(text)
        except Exception:
            sentences = re.split(r"(?<=[.!?])\s+", text)
        return [s.strip() for s in sentences if s.strip()]

    # ---- Step 2: NL → Logical Expression ----

    def _normalize_meaning(self, meaning: str) -> str:
        meaning = meaning.lower().strip()
        meaning = re.sub(r"\b(the|a|an)\b", " ", meaning)
        meaning = re.sub(r"[^a-z0-9]+", " ", meaning)
        meaning = re.sub(r"\s+", " ", meaning).strip()
        return meaning

    def _same_meaning(self, a: str, b: str) -> bool:
        na = self._normalize_meaning(a)
        nb = self._normalize_meaning(b)
        if not na or not nb:
            return False
        return na == nb or SequenceMatcher(None, na, nb).ratio() >= 0.88

    def _find_existing_var_for_meaning(self, meaning: str) -> Optional[str]:
        for var, known in self._meaning_map.items():
            if self._same_meaning(meaning, known):
                return var
        return None

    def _parse_mapping_line(self, line: str) -> dict[str, str]:
        # parse: A="proposition", B="proposition"
        replacements = {}
        for match in re.finditer(r'([A-Za-z][A-Za-z0-9_]*)\s*=\s*"([^"]*)"', line):
            var, meaning = match.group(1), match.group(2)
            existing = self._find_existing_var_for_meaning(meaning)
            if existing and existing != var:
                replacements[var] = existing
            elif var not in self._meaning_map:
                self._meaning_map[var] = meaning
        return replacements

    def _mapping_context(self) -> str:
        if not self._meaning_map:
            return ""
        entries = ", ".join(f'{k}="{v}"' for k, v in self._meaning_map.items())
        return (
            f"\nAlready assigned variables: {entries}"
            "\nReuse these exact letters for the SAME proposition."
            "\nDo NOT invent a new letter for a proposition already listed."
            "\nUse a NEW letter only for propositions not listed above."
        )

    def _replace_vars_in_expr(self, expr_str: str, replacements: dict[str, str]) -> str:
        for old, new in sorted(replacements.items(), key=lambda x: len(x[0]), reverse=True):
            expr_str = re.sub(rf"\b{re.escape(old)}\b", new, expr_str)
        return expr_str

    def nl_to_logical(self, sentence: str) -> str:
        client = self._get_client()
        var_context = self._mapping_context()
        try:
            response = client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.SYSTEM_PROMPT + var_context},
                    {"role": "user", "content": f'Sentence: "{sentence}"'},
                ],
                temperature=0,       # deterministic output
                max_tokens=200,
            )
            raw = response.choices[0].message.content.strip()
            # Track token usage
            self._api_cost_tokens += response.usage.total_tokens
            lines = [l.strip() for l in raw.splitlines() if l.strip()]
            expr_str = self._clean_expression(lines[0]) if lines else ""
            if len(lines) >= 2:
                replacements = self._parse_mapping_line(lines[1])
                if replacements:
                    expr_str = self._replace_vars_in_expr(expr_str, replacements)
            return expr_str
        except Exception as e:
            print(f"  [API Error] {e}")
            return ""

    # ---- Step 3: Parse → SymPy ----

    def _clean_expression(self, expr_str: str) -> str:
        expr_str = expr_str.strip()
        expr_str = re.sub(r"^```(?:text|python)?", "", expr_str).strip()
        expr_str = re.sub(r"```$", "", expr_str).strip()
        if ":" in expr_str and not expr_str.startswith(("Or(", "And(", "Not(", "Implies(", "Equivalent(")):
            expr_str = expr_str.split(":", 1)[1].strip()
        return expr_str

    def parse_expression(self, expr_str: str):
        expr_str = self._clean_expression(expr_str)
        if not expr_str:
            return None
        try:
            tree = self._parser.parse(expr_str)
            transformer = LogicTransformer(self._var_map)
            sympy_expr = transformer.transform(tree)
            return sympy_expr
        except Exception as e:
            print(f"  [Parse Error] '{expr_str}' → {e}")
            # Fallback: thử nhận dạng đơn giản
            return self._fallback_parse(expr_str)

    def _fallback_parse(self, expr_str: str):
        # Nếu chỉ là tên biến đơn
        if re.match(r'^[A-Za-z][A-Za-z0-9_]*$', expr_str):
            return self._get_or_create_sym(expr_str)
        return None

    def _get_or_create_sym(self, name: str):
        if name not in self._var_map:
            self._var_map[name] = symbols(name)
        return self._var_map[name]

    # ---- Step 4: → CNF + Simplify ----

    def to_cnf_simplified(self, sympy_expr) -> str:
        cnf = to_cnf(sympy_expr)
        simplified = simplify_logic(cnf, form="cnf")
        return str(simplified)

    def sympy_to_dimacs(self, sympy_expr) -> dict:
        return sympy_to_dimacs(sympy_expr)

    def save_dimacs(self, sympy_expr, path: str) -> dict:
        dimacs_info = self.sympy_to_dimacs(sympy_expr)
        with open(path, "w", encoding="utf-8") as f:
            f.write(dimacs_info["dimacs"])
        dimacs_info["path"] = path
        return dimacs_info

    # ---- Full Pipeline ----

    def convert(self, text: str, verbose: bool = True) -> dict:
        self._var_map = {}       # reset variable map cho mỗi input mới
        self._meaning_map = {}   # reset meaning map cho mỗi input mới
        self._api_cost_tokens = 0

        # Step 1: Tokenize
        sentences = self.tokenize(text)
        if verbose:
            print(f"\n[Lang2Logic] Input: {len(sentences)} câu")

        logical_exprs = []
        sympy_exprs   = []
        cnf_per_sent  = []

        for i, sent in enumerate(sentences):
            if verbose:
                print(f"\n  Sentence #{i+1}: {sent}")

            # Step 2: NL → Logical
            expr_str = self.nl_to_logical(sent)
            logical_exprs.append(expr_str)
            if verbose:
                print(f"  Model Output: {expr_str}")

            # Step 3: Parse
            sympy_expr = self.parse_expression(expr_str)
            if sympy_expr is None:
                if verbose:
                    print(f"  [Skip] Không parse được.")
                cnf_per_sent.append("(parse error)")
                continue
            sympy_exprs.append(sympy_expr)

            # Step 4: CNF per sentence
            cnf_str = self.to_cnf_simplified(sympy_expr)
            cnf_per_sent.append(cnf_str)
            if verbose:
                print(f"  CNF Expression: {cnf_str}")

        # Kết hợp tất cả → CNF chung
        if sympy_exprs:
            from sympy.logic.boolalg import And as SympyAnd
            combined = sympy_exprs[0]
            for e in sympy_exprs[1:]:
                combined = SympyAnd(combined, e)
            final_cnf = self.to_cnf_simplified(combined)
        else:
            final_cnf = ""

        # Phân tích CNF
        variables = list(self._var_map.keys())
        # Đếm số clauses (đơn giản: đếm dấu &)
        n_clauses = final_cnf.count("&") + 1 if final_cnf else 0

        if verbose:
            print("\n")
            print(f"  Final CNF (Simplified):")
            print(f"  {final_cnf}")
            print(f"  Variables ({len(variables)}): {variables}")
            print(f"  Meanings: { {k: v for k, v in self._meaning_map.items()} }")
            print(f"  Clauses (approx): {n_clauses}")
            print(f"  API tokens used: {self._api_cost_tokens}")
            print("\n")

        result = {
            "sentences"            : sentences,
            "logical_expressions"  : logical_exprs,
            "cnf_per_sentence"     : cnf_per_sent,
            "final_cnf"            : final_cnf,
            "variables"            : variables,
            "meanings"             : dict(self._meaning_map),
            "n_clauses"            : n_clauses,
            "api_tokens_used"      : self._api_cost_tokens,
        }
        if sympy_exprs:
            result["sympy_expr"] = combined
            result["dimacs"] = self.sympy_to_dimacs(combined)
        return result

    def estimate_cost_usd(self) -> float:
        cost_per_million = 0.15   # USD, gpt-4o-mini input
        return self._api_cost_tokens / 1_000_000 * cost_per_million


def sympy_to_dimacs(sympy_expr) -> dict:
    cnf_expr = to_cnf(sympy_expr, simplify=True)
    symbols_sorted = sorted(cnf_expr.free_symbols, key=lambda s: str(s))
    var_to_int = {str(sym): i + 1 for i, sym in enumerate(symbols_sorted)}

    if cnf_expr == BooleanTrue():
        clauses = []
    elif cnf_expr == BooleanFalse():
        clauses = [[]]
    elif isinstance(cnf_expr, And):
        clauses = [_clause_to_dimacs(arg, var_to_int) for arg in cnf_expr.args]
    else:
        clauses = [_clause_to_dimacs(cnf_expr, var_to_int)]

    dimacs_lines = [
        "c generated by LangSAT Lang2Logic",
        f"p cnf {len(var_to_int)} {len(clauses)}",
    ]
    for clause in clauses:
        dimacs_lines.append(" ".join(str(lit) for lit in clause) + " 0")

    return {
        "n_vars": len(var_to_int),
        "n_clauses": len(clauses),
        "var_to_int": var_to_int,
        "int_to_var": {v: k for k, v in var_to_int.items()},
        "clauses": clauses,
        "dimacs": "\n".join(dimacs_lines) + "\n",
    }


def _clause_to_dimacs(expr, var_to_int: dict[str, int]) -> list[int]:
    if isinstance(expr, Or):
        literals = expr.args
    else:
        literals = (expr,)

    clause = []
    for literal in literals:
        if isinstance(literal, Not):
            name = str(literal.args[0])
            clause.append(-var_to_int[name])
        else:
            name = str(literal)
            clause.append(var_to_int[name])
    return clause


if __name__ == "__main__":
    # Test với Example từ bài báo (Figure 3)
    test_paragraph = (
        "The circus has a ferris wheel or the circus has a rollercoaster. "
        "The circus does not have a carousel if and only if the circus has a ferris wheel "
        "and the circus has a rollercoaster. "
        "If the circus does not have a carousel, then the circus has a trapese. "
        "The circus does not have a trapese and the circus has a rollercoaster."
    )

    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        print("[Warning] OPENAI_API_KEY chưa được set.")
        print("Chạy: import os; os.environ['OPENAI_API_KEY'] = 'sk-...'")
    else:
        pipeline = Lang2Logic(api_key=api_key)
        result = pipeline.convert(test_paragraph, verbose=True)
        print(f"\nAPI Cost estimate: ${pipeline.estimate_cost_usd():.6f} USD")
