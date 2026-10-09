# DESTRUCTIVE DB TESTS - WIPES A DATABASE

**Do not run these against the live `lens` database.** Every test in this folder
calls `drop_schema()` + `apply_schema()`, which **drop and recreate** the
`papers` and `chunks` tables. On `lens` that destroys the full-corpus data
(166,704 papers / ~4 M chunks).

These tests target a **throwaway** database `<LENS_DB_NAME>_test` (default
`lens_test`), selected in each file via `dbname=_TEST_DB`. They never touch
`lens`.

## Why they are hidden from the normal suite
`unittest discover -s tests` does **not** descend into subdirectories that lack
an `__init__.py`. This folder has none, so a normal test run ignores it.

## One-time setup (lens-pg container must be running)
```
docker exec lens-pg psql -U lens -d lens -c "CREATE DATABASE lens_test"
```

## Run (on purpose only)
```
.\.venv\Scripts\python.exe -m unittest discover -s tests\DESTRUCTIVE_db_tests -p "test_*.py" -v
```

If a run reports *"no throwaway lens_test database reachable"*, create it as
above (or start the container).

## Files
- `test_schema.py` - M2 pgvector schema (`papers` / `chunks`, HNSW, `lens_ro`).
- `test_papers.py` - M3 abstract-embedding loader.
- `test_chunks.py` - M4 chunk loader + `chunk_uuid`.

All three now connect to `lens_test`; the destructive `drop_schema` runs there,
never on `lens`.
