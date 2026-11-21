"""Safe extraction of supported course files from a user supplied ZIP."""
from pathlib import Path, PurePosixPath
import re
import shutil
import zipfile

SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt", ".csv"}
MAX_ARCHIVE_BYTES = 200 * 1024 * 1024
MAX_ARCHIVE_ENTRIES = 2000


def _safe_stem(filename):
    stem = Path(filename).stem
    stem = re.sub(r"[^A-Za-z0-9._ -]+", "_", stem).strip(" ._")
    return (stem or "course_materials")[:80]


def extract_course_zip(uploaded_bytes, archive_name, course_folder):
    """Extract supported files under a folder named after the ZIP file.

    ZIP paths are normalized and checked before writing. Unsupported files,
    symlinks, traversal paths, and oversized archives are skipped/rejected.
    Returns (destination_folder, extracted_file_count).
    """
    if len(uploaded_bytes) > MAX_ARCHIVE_BYTES:
        raise ValueError("ZIP exceeds the 200 MB upload limit.")
    course_folder = Path(course_folder).resolve()
    root = course_folder / "uploaded_archives"
    root.mkdir(parents=True, exist_ok=True)
    base = _safe_stem(archive_name)
    destination = root / base
    suffix = 2
    while destination.exists():
        destination = root / f"{base}_{suffix}"
        suffix += 1

    created = False
    extracted = 0
    try:
        with zipfile.ZipFile(__import__("io").BytesIO(uploaded_bytes)) as archive:
            entries = archive.infolist()
            if len(entries) > MAX_ARCHIVE_ENTRIES:
                raise ValueError(f"ZIP contains more than {MAX_ARCHIVE_ENTRIES} entries.")
            total_size = sum(item.file_size for item in entries)
            if total_size > MAX_ARCHIVE_BYTES:
                raise ValueError("Uncompressed ZIP contents exceed the 200 MB limit.")

            safe_entries = []
            for item in entries:
                if item.is_dir():
                    continue
                # Ignore Unix symlinks; they must never be followed on extract.
                mode = (item.external_attr >> 16) & 0xFFFF
                if mode and (mode & 0o170000) == 0o120000:
                    continue
                normalized = item.filename.replace("\\", "/")
                relative = PurePosixPath(normalized)
                if relative.is_absolute() or any(part in {"..", ""} for part in relative.parts):
                    continue
                if Path(relative.name).suffix.lower() not in SUPPORTED_EXTENSIONS:
                    continue
                safe_entries.append((item, relative))

            if not safe_entries:
                raise ValueError("ZIP has no supported PDF, DOCX, TXT, or CSV files.")
            destination.mkdir(parents=True, exist_ok=False)
            created = True
            for item, relative in safe_entries:
                target = (destination / Path(*relative.parts)).resolve()
                if not target.is_relative_to(destination.resolve()):
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                with archive.open(item) as source, target.open("wb") as output:
                    shutil.copyfileobj(source, output)
                extracted += 1
        if not extracted:
            raise ValueError("No supported files could be safely extracted.")
        return destination, extracted
    except Exception:
        if created and destination.exists():
            shutil.rmtree(destination)
        raise
