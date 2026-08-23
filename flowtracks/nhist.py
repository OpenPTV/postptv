import numpy as np
from typing import Optional, Tuple, Union

def nhist_scipy(
    y: np.ndarray,
    bins: Union[int, np.ndarray] = 20,
    plot: bool = True,
    density: bool = True,
    kde: bool = False
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Fast, robust normalized histogram (PDF) with optional KDE overlay.

    Parameters
    ----------
    y : array-like
        Input data.
    bins : int or array-like, default 20
        Number of bins or bin edges.
    plot : bool, default True
        If True, plot the PDF.
    density : bool, default True
        If True, normalize to form a PDF.
    kde : bool, default False
        If True, overlay a Gaussian KDE.
    ax : matplotlib Axes, optional
        Axis to plot on.
    **plot_kwargs : dict
        Additional plot arguments.

    Returns
    -------
    hist : ndarray
        PDF values.
    bin_centers : ndarray
        Bin centers.
    """
    y = np.asarray(y).ravel()
    if y.size == 0:
        raise ValueError("Input data is empty.")

    hist, bin_edges = np.histogram(y, bins=bins, density=density)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    return hist, bin_centers


def nhist(y, x=None, *args, plot=True):
    """
    Normalized histogram (PDF), Matlab nhist.m equivalent.
    
    Parameters
    ----------
    y : array-like
        Input data (vector or matrix).
    x : int or array-like, optional
        If int, number of bins. If array, bin centers.
    *args :
        Additional arguments for matplotlib plot.
    plot : bool, default True
        If True, plot the histogram. If False, return data.
    
    Returns
    -------
    no : ndarray
        Normalized counts (PDF values).
    xo : ndarray
        Bin centers.
    """
    # Matlab: if nargin == 0, error
    if y is None:
        raise ValueError('Requires one or two input arguments.')
    y = np.asarray(y)
    if x is None:
        x = 20  # Matlab: default 20 bins
    # Matlab: if min(size(y))==1, y = y(:)
    y = y.ravel()
    # Matlab: if isstr(x) | isstr(y), error
    if isinstance(x, str) or isinstance(y, str):
        raise ValueError('Input arguments must be numeric.')
    # Matlab: if isempty(y)
    if y.size == 0:
        if np.isscalar(x):
            bin_edges = np.linspace(0, 1, int(x)+1)
        else:
            bin_edges = np.asarray(x)
        nn = np.zeros(len(bin_edges)-1, dtype=float)
        xo = (bin_edges[:-1] + bin_edges[1:]) / 2
    else:
        if np.isscalar(x):
            miny = np.min(y)
            maxy = np.max(y)
            if miny == maxy:
                miny = miny - np.floor(x/2) - 0.5
                maxy = maxy + np.ceil(x/2) - 0.5
            bin_edges = np.linspace(miny, maxy, int(x)+1)
        else:
            bin_edges = np.asarray(x)
        nn, _ = np.histogram(y, bins=bin_edges)
        xo = (bin_edges[:-1] + bin_edges[1:]) / 2
    # Normalize to PDF
    if len(xo) > 1:
        bin_width = np.abs(xo[1] - xo[0])
        norm = np.sum(nn) * bin_width
        if norm > 0:
            nn = nn / norm
    else:
        nn = np.zeros_like(xo, dtype=float)

    return nn, xo


# --- Test block to compare nhist and nhist_scipy ---
if __name__ == "__main__":
    from matplotlib import pyplot as plt

    np.random.seed(0)
    data = np.random.normal(loc=0, scale=1, size=1000)
    bins = 30

    # Run nhist
    nn1, xo1 = nhist(data, bins, plot=False)
    nn2, xo2 = nhist_scipy(data, bins, plot=False)
 
    # Plot both for comparison
    plt.figure(figsize=(8,5))
    plt.plot(xo1, nn1, label='nhist', marker='o')
    if nn2 is not None and xo2 is not None:
        plt.plot(xo2, nn2, label='nhist_scipy', marker='x')
    plt.xlabel('Value')
    plt.ylabel('PDF')
    plt.title('Comparison of nhist and nhist_scipy')
    plt.legend()
    plt.show()

    # Print numerical comparison
    if nn2 is not None and xo2 is not None:
        print("nhist (first 5):", nn1[:5])
        print("nhist_scipy (first 5):", nn2[:5])
        print("Difference (first 5):", (nn1[:5] - nn2[:5]))

