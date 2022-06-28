import os

import tarfile
import urllib.request

from tqdm import tqdm


# Helper for loading models (locally or from AWS) ----------------------------------------------------------------

LOCAL_PATH = ".model_cache"
REMOTE_URL = "https://nbfbaselines.s3.eu-west-1.amazonaws.com/"

def load_model_directory(name_or_path):
    if os.path.exists(name_or_path): return name_or_path

    # Try to find locally
    base_path = os.path.dirname(os.path.realpath(__file__))
    base_path = os.path.dirname(base_path)
    local_path = os.path.join(base_path, LOCAL_PATH, name_or_path)
    
    if os.path.isdir(local_path): return local_path

    return download_model(name_or_path, os.path.join(base_path, LOCAL_PATH))


# Download helper ----------------------------------------------------------------

class DownloadProgressBar(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download_model(name, local_path):
    if not os.path.exists(local_path): os.makedirs(local_path)
    archive_path = os.path.join(local_path, "%s.tar.gz" % name)

    remote_url = REMOTE_URL + name + ".tar.gz"

    # Test if url is reachable
    try:
        urllib.request.urlopen(remote_url)
    except urllib.error.HTTPError:
        raise ValueError(f'Model "{name}" does not exist')

    with DownloadProgressBar(unit='B', unit_scale=True,
                             miniters=1, desc="Download %s" % name) as t:
        urllib.request.urlretrieve(remote_url,
                                    filename=archive_path,
                                    reporthook=t.update_to)

    archive_file = tarfile.open(archive_path, 'r:gz')
    archive_file.extractall(local_path)
    archive_file.close()
    os.remove(archive_path)

    return os.path.join(local_path, name)


