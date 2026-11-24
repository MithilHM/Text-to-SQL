"""Unit tests for evaluation/sql_parser.py"""

import pytest
import sqlite3
import tempfile
import os
from evaluation.sql_parser import (
    extract_sql,
    normalize_sql,
    check_exact_match,
    check_sql_validity,
    execute_query,
    compare_query_results,
    categorize_error
)

def test_extract_sql():
    # Markdown fenced
    raw = "```sql\nSELECT name FROM students;\n```"
    assert extract_sql(raw) == "SELECT name FROM students;"

    # Markdown without language tag
    raw2 = "```\nSELECT id FROM users\n```"
    assert extract_sql(raw2) == "SELECT id FROM users;"

    # Prefix text
    raw3 = "SQL: SELECT count(*) FROM orders"
    assert extract_sql(raw3) == "SELECT count(*) FROM orders;"

    # Multiple statements
    raw4 = "SELECT a FROM t; SELECT b FROM t;"
    assert extract_sql(raw4) == "SELECT a FROM t;"

def test_normalize_and_exact_match():
    sql1 = "SELECT  name, age  FROM  students WHERE age > 20;"
    sql2 = "select name, age from students where age > 20"
    assert check_exact_match(sql1, sql2)

    sql3 = "SELECT name FROM students;"
    sql4 = "SELECT age FROM students;"
    assert not check_exact_match(sql3, sql4)

def test_check_sql_validity():
    schema = "CREATE TABLE employees (id INT PRIMARY KEY, name TEXT, salary REAL);"

    # Valid query
    valid, err_type, err = check_sql_validity(schema, "SELECT name, salary FROM employees WHERE salary > 50000;")
    assert valid
    assert err_type == ""

    # Unknown column
    valid, err_type, err = check_sql_validity(schema, "SELECT non_existent_col FROM employees;")
    assert not valid
    assert err_type == "unknown_column_table"

    # Unknown table
    valid, err_type, err = check_sql_validity(schema, "SELECT name FROM non_existent_table;")
    assert not valid
    assert err_type == "unknown_column_table"

    # Syntax error
    valid, err_type, err = check_sql_validity(schema, "SELECT FROM employees WHERE;")
    assert not valid
    assert err_type == "syntax_error"

def test_compare_query_results():
    gold_unordered = "SELECT name, age FROM students"
    res1 = [("Alice", 20), ("Bob", 22)]
    res2 = [("Bob", 22), ("Alice", 20)]
    # Unordered matches multiset
    assert compare_query_results(res1, res2, gold_unordered)

    gold_ordered = "SELECT name, age FROM students ORDER BY age ASC"
    # Ordered must match exact sequence
    assert not compare_query_results(res1, res2, gold_ordered)
    assert compare_query_results(res1, res1, gold_ordered)

def test_execute_query_and_timeout():
    tmp = tempfile.NamedTemporaryFile(suffix=".sqlite", delete=False)
    db_path = tmp.name
    tmp.close()

    try:
        conn = sqlite3.connect(db_path)
        conn.execute("CREATE TABLE t (x INT);")
        conn.execute("INSERT INTO t VALUES (1), (2), (3);")
        conn.commit()
        conn.close()

        ok, rows, err = execute_query(db_path, "SELECT * FROM t WHERE x >= 2", timeout_sec=2.0)
        assert ok
        assert rows == [(2,), (3,)]
        assert err == ""

        # Bad query
        ok, rows, err = execute_query(db_path, "SELECT bad_col FROM t", timeout_sec=2.0)
        assert not ok
        assert "unknown_column_table" in err or "no such column" in err
    finally:
        try:
            if os.path.exists(db_path):
                os.remove(db_path)
        except OSError:
            pass

def test_categorize_error():
    assert categorize_error(True, True, "") == "correct"
    assert categorize_error(False, False, "syntax_error") == "syntax_error"
    assert categorize_error(False, False, "unknown_column_table") == "unknown_column_table"
    assert categorize_error(False, True, "", exec_err="Query timed out after 10s") == "timeout"
    assert categorize_error(False, True, "", exec_success=True, exec_match=False) == "wrong_result"
