"""Constants and helpers shared by the CLI commands."""

from pathlib import Path

EXIT_OK = 0
EXIT_CHECK_FAILED = 1
EXIT_INVALID_INPUT = 2
EXIT_ENVIRONMENT = 3
EXIT_INTERRUPTED = 130
DB_NAME = "agentoscopy.db"


def all_task_ids(tasks_dir: Path) -> list[str]:
    if not tasks_dir.is_dir():
        return []
    return sorted(path.name for path in tasks_dir.iterdir() if (path / "task.yaml").is_file())
