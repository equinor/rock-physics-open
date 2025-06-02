<div align="center">

# rock-physics-open

[![License: LGPL v3][license-badge]][license]
[![SCM Compliance][scm-compliance-badge]][scm-compliance]
[![on push main action status][on-push-main-action-badge]][on-push-main-action]

</div>

This repository contains Python code for rock physics modules created in Equinor by
Harald Flesche 2010 - ... Batzle-Wang and Span-Wagner fluid equations are implemented
by Eivind Jahren and Jimmy Zurcher. Some models are based on original Matlab code
by Tapan Mukerji at Stanford University and ported to Python by Harald Flesche.

The modules in this repository are implementations of rock physics models
used in quantitative seismic analysis, in addition to some utilities for handling
of seismic and well data. The repository started as internal Equinor plugins, and was
extracted as a separate repository that could be used within other internal applications
in 2023. In 2025 it was released under LGPL license.

The content of the library can be described as follows:

Functions with inputs and outputs consisting of numpy arrays or in some
cases pandas dataframes. Data frames are used in the cases where there are
many inputs and/or there is a need for checking the name of inputs, such
as when there are multiple inputs of the same type which will have
different purpose. It should be made clear in which cases data dataframes
are expected.
There is normally not any check on inputs, it is just the minimum
definition of equations and other utilities.


## Installation

This module can be installed through [PyPI](https://pypi.org/project/rock-physics-open/) with:

```sh
pip install rock-physics-open
```

Alternatively, you can update the dependencies in your `pyproject.toml` file:

<!-- x-release-please-start-version -->
```toml
dependencies = [
    "rock-physics-open == 0.2.0",
]
```
<!-- x-release-please-end-version -->

<!-- External Links -->
[scm-compliance]: https://developer.equinor.com/governance/scm-policy/
[scm-compliance-badge]: https://scm-compliance-api.radix.equinor.com/repos/equinor/rock-physics-open/badge
[license]: https://www.gnu.org/licenses/lgpl-3.0
[license-badge]: https://img.shields.io/badge/License-LGPL_v3-blue.svg
[on-push-main-action]: https://github.com/equinor/rock-physics-open/actions/workflows/on-push-main.yaml
[on-push-main-action-badge]:  https://github.com/equinor/rock-physics-open/actions/workflows/on-push-main.yaml/badge.svg
