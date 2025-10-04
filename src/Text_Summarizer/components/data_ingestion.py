import os
import urllib.request as request
import zipfile
from Text_Summarizer.logging import logger
from Text_Summarizer.utils.common import get_size
from pathlib import Path
from Text_Summarizer.entity import DataIngestionConfig
import gdown

class DataIngestion:
    def __init__(self, config: DataIngestionConfig):
        self.config = config

    def download_file(self):
        if not os.path.exists(self.config.local_data_file):
            logger.info(f"Downloading file from Google Drive: {self.config.source_URL}")
            gdown.download(self.config.source_URL, str(self.config.local_data_file), quiet=False)
        else:
            logger.info(f"File already exists of size: {get_size(Path(self.config.local_data_file))}")

    def extract_zip_file(self):
        unzip_path = self.config.unzip_dir
        os.makedirs(unzip_path, exist_ok=True)
        with zipfile.ZipFile(self.config.local_data_file, 'r') as zip_ref:
            zip_ref.extractall(unzip_path)
