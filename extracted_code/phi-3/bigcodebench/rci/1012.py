import aiohttp
import asyncio
import logging
import zipfile
from pathlib import Path

logging.basicConfig(level=logging.INFO)

async def download_file(url, filename):
    download_dir = Path("downloads")
    if not download_dir.exists():
        download_dir.mkdir(parents=True)

    file_path = download_dir / filename
    async with aiohttp.ClientSession() as session:
        try:
            async with session.get(url) as response:
                response.raise_for_status()
                with open(file_path, 'wb') as file:
                    while True:
                        chunk = await response.content.read(1024)
                        if not chunk:
                            break
                        file.write(chunk)
        except aiohttp.ClientError as e:
            logging.error(f"Download failed: {e}")
            return None
    return file_path

async def extract_file(zip_path):
    zip_dir = Path("unzipped_files")
    if not zip_dir.exists():
        zip_dir.mkdir(parents=True)

    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(zip_dir)
    except zipfile.BadZipFile as e:
        logging.error(f"Extraction failed: {e}")
        return None
    except Exception as e:
        logging.error(f"Error during extraction: {e}")
        return None
    return zip_dir

async def task_func(url, filename):
    file_path = await download_file(url, filename)
    if not file_path:
        return "Error", []

    zip_dir = await extract_file(file_path)
    if not zip_dir:
        return "Error", []

    files = [f.name for f in zip_dir.glob('**/*') if f.is_file()]
    return "Download and extraction successful", files

loop = asyncio.get_event_loop