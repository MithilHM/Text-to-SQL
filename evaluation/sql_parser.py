"""SQL Parser, Normalizer, Validity Checker, and Result Set Comparator.

Handles:
1. Normalization (lowercase, strip semicolon, collapse whitespace, sqlglot).
2. SQL extraction (strip markdown code blocks, extract first query).
3. Level 2 Validity: In-memory SQLite DDL execution + EXPLAIN <sql>.
4. Level 3 Execution Accuracy: Multiset and ordered list comparison with timeout.
5. Error categorization (syntax error, unknown column/table, wrong result, timeout).
"""

import collections
import concurrent.futures
import re
import sqlite3
from typing import Any, Dict, List, Optional, Tuple

try:
    import sqlglot
except ImportError:
    sqlglot = None


def extract_sql(raw_output: str) -> str:
    """Extract clean SQL query from model output, stripping markdown code fences."""
    if not raw_output:
        return ""

    text = raw_output.strip()

    # Case 1: Markdown code fences ```sql ... ``` or ``` ... ```
    match = re.search(r"```(?:sql)?\s*([\s\S]*?)\s*```", text, re.IGNORECASE)
    if match:
        text = match.group(1).strip()

    # Case 2: Strip prefix like "SQL:" or "Assistant:"
    text = re.sub(r"^(?:SQL:|Assistant:|Query:)\s*", "", text, flags=re.IGNORECASE).strip()

    # Case 3: Take first statement ending with semicolon if multiple present
    statements = text.split(";")
    if len(statements) > 1 and statements[0].strip():
        text = statements[0].strip() + ";"
    elif not text.endswith(";"):
        text = text + ";"

    return text.strip()


def normalize_sql(sql: str) -> str:
    """Normalize SQL for exact-match comparison."""
    if not sql:
        return ""

    text = sql.strip()
    # Strip trailing semicolon
    if text.endswith(";"):
        text = text[:-1].strip()

    # Strip line comments
    text = re.sub(r"--[^\n]*", " ", text)
    # Strip block comments
    text = re.sub(r"/\*[\s\S]*?\*/", " ", text)

    # Collapse whitespace and lowercase
    text = re.sub(r"\s+", " ", text).strip().lower()

    # Try AST normalization via sqlglot
    if sqlglot:
        try:
            parsed = sqlglot.transpile(text, read="sqlite", write="sqlite", pretty=False)
            if parsed and len(parsed) > 0:
                normalized = parsed[0].lower().rstrip(";").strip()
                return re.sub(r"\s+", " ", normalized)
        except Exception:
            pass

    return text


def check_exact_match(pred_sql: str, gold_sql: str) -> bool:
    """Compare predicted and gold SQL after normalization."""
    pred_clean = normalize_sql(extract_sql(pred_sql))
    gold_clean = normalize_sql(extract_sql(gold_sql))
    return pred_clean == gold_clean and len(pred_clean) > 0


def check_sql_validity(schema_ddl: str, sql: str) -> Tuple[bool, str, str]:
    """Check Level 2 validity against schema using SQLite in-memory DB and EXPLAIN.
    
    Returns:
        (is_valid, error_type, error_message)
        error_type is one of: "", "syntax_error", "unknown_column_table", "execution_error"
    """
    cleaned_sql = extract_sql(sql).rstrip(";").strip()
    if not cleaned_sql:
        return False, "syntax_error", "Empty SQL string"

    conn = sqlite3.connect(":memory:")
    cursor = conn.cursor()

    # 1. Execute schema DDL
    try:
        cursor.executescript(schema_ddl)
    except sqlite3.Error as e:
        conn.close()
        # Schema issue in test data itself - return execution error
        return False, "schema_error", f"Schema DDL failed: {e}"

    # 2. Check query validity using EXPLAIN
    try:
        cursor.execute(f"EXPLAIN {cleaned_sql};")
        conn.close()
        return True, "", ""
    except sqlite3.OperationalError as e:
        conn.close()
        err_msg = str(e).lower()
        if "no such table" in err_msg or "no such column" in err_msg or "has no column" in err_msg:
            return False, "unknown_column_table", str(e)
        elif "syntax error" in err_msg or "incomplete input" in err_msg or "near " in err_msg:
            return False, "syntax_error", str(e)
        else:
            return False, "syntax_error", str(e)
    except sqlite3.Error as e:
        conn.close()
        return False, "execution_error", str(e)


def _exec_single_query(db_path: str, sql: str) -> List[Tuple]:
    """Execute query against sqlite file and fetch all rows."""
    uri = f"file:{db_path}?mode=ro"
    conn = sqlite3.connect(uri, uri=True, timeout=10.0)
    cursor = conn.cursor()
    cursor.execute(sql)
    rows = cursor.fetchall()
    conn.close()
    return rows


def execute_query(db_path: str, sql: str, timeout_sec: float = 10.0) -> Tuple[bool, Optional[List[Tuple]], str]:
    """Execute SQLite query with hard timeout.
    
    Returns:
        (success, rows, error_message)
    """
    cleaned_sql = extract_sql(sql).strip()
    if not cleaned_sql:
        return False, None, "Empty SQL query"

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(_exec_single_query, db_path, cleaned_sql)
        try:
            rows = future.result(timeout=timeout_sec)
            return True, rows, ""
        except concurrent.futures.TimeoutError:
            return False, None, f"Query timed out after {timeout_sec}s"
        except sqlite3.OperationalError as e:
            err = str(e).lower()
            if "no such table" in err or "no such column" in err:
                return False, None, f"unknown_column_table: {e}"
            return False, None, f"syntax_error: {e}"
        except Exception as e:
            return False, None, f"execution_error: {e}"


def compare_query_results(pred_rows: Optional[List[Tuple]], gold_rows: Optional[List[Tuple]], gold_sql: str) -> bool:
    """Compare result sets according to Spider evaluation rules.
    
    If gold query contains ORDER BY: compare as exact ordered list.
    Otherwise: compare as multisets (Counter of row tuples).
    """
    if pred_rows is None or gold_rows is None:
        return False

    has_order_by = bool(re.search(r"\border\s+by\b", gold_sql, re.IGNORECASE))

    if has_order_by:
        return pred_rows == gold_rows

    # Multiset comparison
    # Normalize row elements (float precision / string strip)
    def normalize_row(row):
        norm = []
        for v in row:
            if isinstance(v, float):
                norm.append(round(v, 4))
            elif isinstance(v, str):
                norm.append(v.strip().lower())
            else:
                norm.append(v)
        return tuple(norm)

    pred_norm = [normalize_row(r) for r in pred_rows]
    gold_norm = [normalize_row(r) for r in gold_rows]

    return collections.Counter(pred_norm) == collections.Counter(gold_norm)


def categorize_error(
    exact_match: bool,
    is_valid: bool,
    valid_err_type: str,
    exec_success: Optional[bool] = None,
    exec_err: str = "",
    exec_match: Optional[bool] = None
) -> str:
    """Categorize prediction outcome into mutually exclusive error bucket."""
    if exact_match or (exec_match is True):
        return "correct"

    if "timed out" in exec_err.lower():
        return "timeout"

    if not is_valid:
        if valid_err_type in ("syntax_error", "unknown_column_table"):
            return valid_err_type
        return "syntax_error"

    if exec_success is False:
        if "unknown_column_table" in exec_err.lower():
            return "unknown_column_table"
        elif "syntax_error" in exec_err.lower():
            return "syntax_error"
        return "execution_error"

    return "wrong_result"
