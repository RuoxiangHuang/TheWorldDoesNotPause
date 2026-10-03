from benchmarks.official_tasks import (
    N_TASKS,
    dynamic_libero_tasks,
    dynamic_robotwin_tasks,
    official_tasks,
)


def test_paper_suite_has_95_tasks():
    tasks = official_tasks()
    assert len(tasks["dynamic_libero"]) == 50
    assert len(tasks["dynamic_robotwin"]) == 45
    assert len(dynamic_libero_tasks()) + len(dynamic_robotwin_tasks()) == N_TASKS
    libero_ids = [t.id for t in tasks["dynamic_libero"]]
    robotwin_ids = [t.id for t in tasks["dynamic_robotwin"]]
    assert len(set(libero_ids)) == 50
    assert len(set(robotwin_ids)) == 45
