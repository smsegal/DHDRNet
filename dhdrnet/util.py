from pathlib import Path
from subprocess import check_output


def get_project_root() -> Path:
    git_root = (
        check_output(["git", "rev-parse", "--show-toplevel"]).decode("utf-8").strip()
    )
    return Path(git_root).absolute()
