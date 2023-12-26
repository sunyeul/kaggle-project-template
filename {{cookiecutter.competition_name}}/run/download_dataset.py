from kaggle.api.kaggle_api_extended import KaggleApi
from pathlib import Path
from omegaconf import DictConfig
from src.utils.timer import timer

import hydra
import zipfile


@hydra.main(
    config_path="../configs", config_name="download_dataset", version_base="1.2"
)
def main(cfg):
    competition_name = cfg.competition_info.name

    raw_data_dir_path: Path = Path(cfg.dir.root_dir_path + cfg.dir.raw_data_dir_path)
    raw_data_dir_path.mkdir(exist_ok=True, parents=True)

    api = KaggleApi()
    api.authenticate()

    with timer(prefix="Downloading files took ", suffix=" seconds"):
        api.competition_download_files(
            competition=competition_name, path=raw_data_dir_path
        )

    with timer(prefix="Extracting files took ", suffix=" seconds"):
        path_to_zip_file = raw_data_dir_path / f"{competition_name}.zip"
        directory_to_extract_to = raw_data_dir_path

        with zipfile.ZipFile(path_to_zip_file, "r") as zip_ref:
            zip_ref.extractall(directory_to_extract_to)

        path_to_zip_file.unlink()


if __name__ == "__main__":
    main()
