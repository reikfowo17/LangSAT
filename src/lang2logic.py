import re
import os
from typing import Optional

# NLP
import nltk
nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)
from nltk.tokenize import sent_tokenize

# Logic parsing
from lark import Lark, Transformer, Token

# Symbolic math
from sympy.logic.boolalg import to_cnf, simplify_logic, And, Or, Not, Equivalent, Implies
from sympy import symbols

LOGIC_GRAMMAR = r"""
    ?start: expr

    ?expr: equiv
    ?equiv: implies
          | equiv "Equivalent(" implies "," implies ")" -> equivalent

    ?implies: or_expr
            | "Implies(" or_expr "," or_expr ")"        -> implies

    ?or_expr: and_expr
            | "Or(" args ")"                             -> or_expr

    ?and_expr: not_expr
             | "And(" args ")"                           -> and_expr

    ?not_expr: atom
             | "Not(" atom ")"                           -> not_expr

    ?atom: NAME                                          -> var

    args: expr ("," expr)*

    NAME: /[A-Za-z][A-Za-z0-9_]*/

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

    def not_expr(self, items):
        return Not(items[0])

    def or_expr(self, items):
        args = items[0] if isinstance(items[0], list) else items
        return Or(*args)

    def and_expr(self, items):
        args = items[0] if isinstance(items[0], list) else items
        return And(*args)

    def implies(self, items):
        return Implies(items[0], items[1])

    def equivalent(self, items):
        return Equivalent(items[0], items[1])

    def args(self, items):
        return list(items)

class Lang2Logic:

    # Prompt thiết kế để GPT output đúng format Or/And/Not
    SYSTEM_PROMPT = """You are a logic translator. Convert the given English sentence into a propositional logic expression.

Rules:
1. Use ONLY these operators: Or(X, Y), And(X, Y), Not(X), Implies(X, Y), Equivalent(X, Y)
2. Use single uppercase letters or short names as variables (P, Q, R, S, ...).
3. Use the SAME variable name if the same proposition appears in multiple sentences.
4. Output ONLY the logical expression, nothing else. No explanation, no markdown.

Examples:
- "A or B" → Or(A, B)
- "If A then B" → Implies(A, B)
- "A if and only if B" → Equivalent(A, B)
- "Not A and B" → And(Not(A), B)
- "A or B, and C" → And(Or(A, B), C)
"""

    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-4o-mini"):
        self.model = model
        self._api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        self._client = None
        self._parser = Lark(LOGIC_GRAMMAR, parser="earley", ambiguity="resolve")
        self._var_map: dict = {}
        self._api_cost_tokens = 0   # tracking token usage

    def _get_client(self):
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
        sentences = sent_tokenize(text)
        return [s.strip() for s in sentences if s.strip()]

    # ---- Step 2: NL → Logical Expression ----

    def nl_to_logical(self, sentence: str) -> str:
        client = self._get_client()
        try:
            response = client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.SYSTEM_PROMPT},
                    {"role": "user", "content": f'Sentence: "{sentence}"'},
                ],
                temperature=0,       # deterministic output
                max_tokens=200,
            )
            expr_str = response.choices[0].message.content.strip()
            # Track token usage
            self._api_cost_tokens += response.usage.total_tokens
            return expr_str
        except Exception as e:
            print(f"  [API Error] {e}")
            return ""

    # ---- Step 3: Parse → SymPy ----

    def parse_expression(self, expr_str: str):
        expr_str = expr_str.strip()
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

    # ---- Full Pipeline ----

    def convert(self, text: str, verbose: bool = True) -> dict:
        self._var_map = {}   # reset variable map cho mỗi input mới

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
            print(f"  Final CNF (Simplified):")
            print(f"  {final_cnf}")
            print(f"  Variables ({len(variables)}): {variables}")
            print(f"  Clauses (approx): {n_clauses}")
            print(f"  API tokens used: {self._api_cost_tokens}")

        return {
            "sentences"            : sentences,
            "logical_expressions"  : logical_exprs,
            "cnf_per_sentence"     : cnf_per_sent,
            "final_cnf"            : final_cnf,
            "variables"            : variables,
            "n_clauses"            : n_clauses,
            "api_tokens_used"      : self._api_cost_tokens,
        }

    def estimate_cost_usd(self) -> float:
        cost_per_million = 0.15   # USD, gpt-4o-mini input
        return self._api_cost_tokens / 1_000_000 * cost_per_million


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
