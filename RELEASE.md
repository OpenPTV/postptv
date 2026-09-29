# Release Process

Versions come from git tags ([setuptools-scm](https://setuptools-scm.readthedocs.io));
nothing in the source tree holds a version number.

## Development releases (automatic)

Every push to `master` that passes the tests is published to PyPI as a
development release: `X.Y.(Z+1).devN`, where `vX.Y.Z` is the last tag and `N`
the number of commits since it (e.g. `1.5.2.dev3`).

`pip install flowtracks` and `uv add flowtracks` ignore development releases,
so users of the stable version never get them by accident. To use one:

```bash
pip install --pre flowtracks                  # newest, including dev releases
pip install "flowtracks>=1.5.2.dev3"          # at least this commit
```

(uv accepts a dev release whenever the requirement itself names one, as in
the second line.)

## Stable releases

We follow [Semantic Versioning](https://semver.org/): MAJOR for incompatible
API changes, MINOR for new backward-compatible functionality, PATCH for
backward-compatible bug fixes.

1. Make sure `master` is pushed and its CI is green.
2. Create the release on GitHub (this creates the tag):
   ```bash
   gh release create vX.Y.Z --title vX.Y.Z --notes "..."
   ```
3. `.github/workflows/python-publish.yml` runs the tests, builds `X.Y.Z` from
   the tag (and fails if the tag and the built version differ) and publishes
   it to PyPI via trusted publishing (GitHub environment `pypi`).

The installed version is `flowtracks.__version__` (written to
`flowtracks/_version.py` at build/install time; not in git).
