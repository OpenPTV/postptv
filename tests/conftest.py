import sys
from pathlib import Path
import yaml
import pytest

SRC_DIR = Path(__file__).parent.parent / 'src'
sys.path.insert(0, str(SRC_DIR))


@pytest.fixture(scope='session')
def src_dir():
    return SRC_DIR


@pytest.fixture(scope='session')
def repo_root():
    return Path(__file__).parent.parent


@pytest.fixture(scope='session')
def config_dict():
    with open(SRC_DIR / 'config.yaml') as f:
        return yaml.safe_load(f)


@pytest.fixture(scope='session')
def grid_dict():
    with open(SRC_DIR / 'grid.yaml') as f:
        return yaml.safe_load(f)
