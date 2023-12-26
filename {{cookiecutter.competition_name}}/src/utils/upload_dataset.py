from pathlib import Path
from typing import Any
from kaggle.api.kaggle_api_extended import KaggleApi

import json
import shutil


def copy_files(source_dir: Path, dest_dir: Path, target_file_list: list):
    for target_file in target_file_list:
        for source_path in source_dir.rglob(f"*{target_file}"):
            relative_path = source_path.relative_to(source_dir)
            dest_path = dest_dir / relative_path
            dest_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_path, dest_path)
            print(f"Copied {source_path} to {dest_path}")


def create_temp_dir() -> Path:
    tmp_dir = Path("./tmp")
    tmp_dir.mkdir(parents=True, exist_ok=True)
    return tmp_dir


def delete_temp_dir(tmp_dir: Path):
    shutil.rmtree(tmp_dir)


def create_dataset_metadata(tmp_dir: Path, user_name: str, title: str):
    dataset_metadata: dict[str, Any] = {
        "id": f"{user_name}/{title}",
        "licenses": [{"name": "CC0-1.0"}],
        "title": title,
    }
    with open(tmp_dir / "dataset-metadata.json", "w") as f:
        json.dump(dataset_metadata, f, indent=4)


def upload_to_kaggle(tmp_dir: Path, new: bool):
    api = KaggleApi()
    api.authenticate()

    if new:
        api.dataset_create_new(
            folder=tmp_dir, dir_mode="tar", convert_to_csv=False, public=False
        )
    else:
        api.dataset_create_version(
            folder=tmp_dir, version_notes="", dir_mode="tar", convert_to_csv=False
        )


def download_from_kaggle(
    competition_name: str,
    download_dir_path: str
):
    api = KaggleApi()
    api.authenticate()

    api.competition_download_files(
        competition=competition_name, path=download_dir_path
    )
