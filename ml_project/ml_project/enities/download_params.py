from dataclasses import dataclass


@dataclass()
class DownloadParams:
    gdrive_id: str
    output_folder: str
