"""Course options and source folders used by the application."""

from pathlib import Path

from config import DATA_DIR

GENERAL_COURSE = "General - all courses"


COURSES = {
    "Database Management Systems": {
        "name": "Database Management Systems",
        "folder": DATA_DIR / "course_1" / "static_resources",
        "subtitle": "MIT OpenCourseWare - Fall 2010",
        "mark": "DB",
        "indexed": True,
    },
    "Human-Computer Interaction": {
        "name": "Human-Computer Interaction",
        "folder": DATA_DIR / "course_2",
        "subtitle": "Course 2 - HCI",
        "mark": "HCI",
        "indexed": True,
    },
    "Computer Security": {
        "name": "Computer Security",
        "folder": DATA_DIR / "course_3",
        "subtitle": "CS 401 - Course 3",
        "mark": "CS",
        "indexed": True,
    },
}


def supported_files(folder: Path):
    extensions = {".pdf", ".docx", ".txt", ".csv"}
    return sorted(
        path for path in folder.rglob("*")
        if path.is_file() and path.suffix.lower() in extensions
    ) if folder.is_dir() else []
