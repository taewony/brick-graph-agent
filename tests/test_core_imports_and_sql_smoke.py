from src.core.targets.sql.agent import events as E
from src.core.targets.sql.agent.agent import retrieve
from src.core.targets.sql.eval import FakeSqlReader
from src.core.targets.sql.target import build_target


def _instance():
    return {
        "question_id": "q1",
        "question_type": "count",
        "question": "How many users are there?",
        "schema_id": "smoke",
        "tables": ["users"],
        "columns_by_table": {"users": ["id", "name"]},
        "foreign_keys": [],
        "primary_keys": {"users": "id"},
        "schema_ddl": "CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT);",
        "seed_rows": [
            "INSERT INTO users (id, name) VALUES (1, 'Ada');",
            "INSERT INTO users (id, name) VALUES (2, 'Grace');",
        ],
        "gold_sql": "SELECT COUNT(*) FROM users",
        "gold_result_set": ((2,),),
    }


def _reader():
    return FakeSqlReader({"q1": ("SELECT COUNT(*) FROM users", "", "")})


def test_task_1_2_builds_sql_target_and_runs_simple_question():
    target = build_target(reader=_reader())

    result = target.eval_backend.run_on_split([_instance()])
    outcome = result.outcomes[0]

    assert target.name == "sql"
    assert result.aggregate["overall_accuracy"] == 1.0
    assert outcome.correct is True
    assert outcome.predicted_sql == "SELECT COUNT(*) FROM users"
    assert outcome.predicted_result_set == ((2,),)
    assert outcome.exec_error == ""


def test_task_1_2_sql_agent_event_chain_reaches_query_drafted():
    trace = retrieve(_instance(), reader=_reader())
    event_types = [ev.type for ev in trace.events]

    assert trace.drafted.predicted_sql == "SELECT COUNT(*) FROM users"
    assert _appears_in_order(
        event_types,
        [
            E.QUESTION_ASKED,
            E.SCHEMA_ENCODED,
            E.COLUMNS_SCORED,
            E.PROMPT_ASSEMBLED,
            E.QUERY_DRAFTED,
        ],
    )


def _appears_in_order(values, expected):
    pos = -1
    for item in expected:
        pos = values.index(item, pos + 1)
    return True
