import sys
from pathlib import Path
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))
from flowtracks.nhist import nhist, nhist_scipy

def test_nhist_vs_scipy():
    np.random.seed(42)
    data = np.random.normal(0, 1, 1000)
    bins = 30
    nn1, xo1 = nhist(data, bins, plot=False)
    nn2, xo2 = nhist_scipy(data, bins, plot=False)
    assert np.allclose(xo1, xo2)
    assert np.allclose(nn1, nn2, atol=1e-8)

def test_nhist_empty():
    nn, xo = nhist([], 10, plot=False)
    assert np.all(nn == 0)
    assert len(xo) == 10

def test_nhist_scipy_empty():
    with pytest.raises(ValueError):
        nhist_scipy([], 10, plot=False)
