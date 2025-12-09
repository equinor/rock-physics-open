# Changelog

## [0.3.5](https://github.com/equinor/rock-physics-open/compare/v0.3.4...v0.3.5) (2025-12-09)


### Bug Fixes

* Add type annotations fluid/sandstone/shale models ([906d1cf](https://github.com/equinor/rock-physics-open/commit/906d1cf63f2da520b41e16e50849988942cbfb62))
* Move typing-extensions to dependencies ([c55ecf8](https://github.com/equinor/rock-physics-open/commit/c55ecf8e06cd55f17c1775eadd44ce47b5bb7883))

## [0.3.4](https://github.com/equinor/rock-physics-open/compare/v0.3.3...v0.3.4) (2025-12-04)


### Bug Fixes

* Allow float and numpy array inputs for multi-wood. Add tests for multi-wood ([3ec9433](https://github.com/equinor/rock-physics-open/commit/3ec9433a4997bef5a0c1dee3ece02dca154a2f90))

## [0.3.3](https://github.com/equinor/rock-physics-open/compare/v0.3.2...v0.3.3) (2025-11-11)


### Bug Fixes

* correct bug in label_vars and label_units ([09ee80b](https://github.com/equinor/rock-physics-open/commit/09ee80b9751deb93786fde9d5fd946bd413b05ba))

## [0.3.2](https://github.com/equinor/rock-physics-open/compare/v0.3.1...v0.3.2) (2025-11-03)


### Bug Fixes

* Add type annotation equinor_utilities ([5b30f46](https://github.com/equinor/rock-physics-open/commit/5b30f4637b829c086685426bd02ae805339b19a4))

## [0.3.1](https://github.com/equinor/rock-physics-open/compare/v0.3.0...v0.3.1) (2025-10-28)


### Bug Fixes

* Add static typing to std_functions and various_utilities ([1ed5353](https://github.com/equinor/rock-physics-open/commit/1ed5353d0886e0d7625ae1e998535bd5c78b9e31))
* catch cases with non-consistent input lengths ([7843c8e](https://github.com/equinor/rock-physics-open/commit/7843c8ed3e9d4144bbe7079d314af20c087d48be))
* make proper reference to cell in numpy array with ndim &gt; 0 ([0abca8f](https://github.com/equinor/rock-physics-open/commit/0abca8fcb82a5d35a990857d4ecbd2f050077b08))
* remove redundant prototypes ([7843c8e](https://github.com/equinor/rock-physics-open/commit/7843c8ed3e9d4144bbe7079d314af20c087d48be))

## [0.3.0](https://github.com/equinor/rock-physics-open/compare/v0.2.3...v0.3.0) (2025-10-21)


### Features

* add an abstract base class for all pressure sensitivity models, add models for polynomial, friable, patchy cement; add tests ([a8cfbdb](https://github.com/equinor/rock-physics-open/commit/a8cfbdb4e7d890f5199a18cf06a98c030003fbc9))
* Add static type checking to gen_utilities ([f3c9424](https://github.com/equinor/rock-physics-open/commit/f3c94246e86810cc40e2a51682e8168ad8f0a28c))


### Bug Fixes

* enforce SI units in all fluid models, update snapshots ([c8fe5c0](https://github.com/equinor/rock-physics-open/commit/c8fe5c0f443c7821036e56153f82503a6b098642))
* Potential fix for code scanning alert no. 3: Workflow does not contain permissions ([8f520ec](https://github.com/equinor/rock-physics-open/commit/8f520ec8fe9990c50dd0d64611022e5a7c233a3d))
* Potential fix for code scanning alert no. 6: Workflow does not contain permissions ([55fe74e](https://github.com/equinor/rock-physics-open/commit/55fe74eedd4c38f1de54a65e7d9980f0f858e785))
* Potential fix for code scanning alert no. 7: Workflow does not contain permissions ([09aa44f](https://github.com/equinor/rock-physics-open/commit/09aa44f936522c541269b3c0aeab7674084b745b))
* Potential fix for code scanning alert no. 9: Workflow does not contain permissions ([d8983f4](https://github.com/equinor/rock-physics-open/commit/d8983f45542dee6c5d6f253273ee35055d44ef07))
* revert isinstance checking ([f3c9424](https://github.com/equinor/rock-physics-open/commit/f3c94246e86810cc40e2a51682e8168ad8f0a28c))

## [0.2.3](https://github.com/equinor/rock-physics-open/compare/v0.2.2...v0.2.3) (2025-08-21)


### Bug Fixes

* minor change to force a version bump ([b194cd3](https://github.com/equinor/rock-physics-open/commit/b194cd30da7c1e612f0a2afc6f67f42e59181c09))

## [0.2.2](https://github.com/equinor/rock-physics-open/compare/v0.2.1...v0.2.2) (2025-08-20)


### Bug Fixes

* improve get_snapshot_name - better detection of function and directory name ([4103f56](https://github.com/equinor/rock-physics-open/commit/4103f560bdebebdcd7c055f419fd0f02416bbee5))
* improve output and fix bug in regex ([ae560aa](https://github.com/equinor/rock-physics-open/commit/ae560aa208957a4bc8aa9132c7385b30fa3996a8))
* simplify data file copying ([a7da331](https://github.com/equinor/rock-physics-open/commit/a7da3315c9639aa25880c694aae137f4efeb6344))

## [0.2.1](https://github.com/equinor/rock-physics-open/compare/v0.2.0...v0.2.1) (2025-08-14)


### Bug Fixes

* Test/robust snapshot ([#35](https://github.com/equinor/rock-physics-open/issues/35)) ([4d169d7](https://github.com/equinor/rock-physics-open/commit/4d169d7b0e2e464a6e50e8583213bc029f20bc2a))

## [0.2.0](https://github.com/equinor/rock-physics-open/compare/v0.1.3...v0.2.0) (2025-06-02)


### Features

* add support for Python 3.12 ([1de7cb3](https://github.com/equinor/rock-physics-open/commit/1de7cb318cbd0b8e01e54de1f8e9842ae32a4e17))


### Bug Fixes

* make get_snapshot_name more robust in search for calling function ([#31](https://github.com/equinor/rock-physics-open/issues/31)) ([e9e9ebd](https://github.com/equinor/rock-physics-open/commit/e9e9ebd8d9d101fa2a2bdd924f6f12be73476de7))

## [0.1.3](https://github.com/equinor/rock-physics-open/compare/v0.1.2...v0.1.3) (2025-05-13)


### Bug Fixes

* remove local version for scm ([d299b64](https://github.com/equinor/rock-physics-open/commit/d299b64c6cc6a75e0a17dabf105e0446be42a81d))
* initiate standardization of names for input parameters ([#25](https://github.com/equinor/rock-physics-open/issues/25)) ([3d505e3](https://github.com/equinor/rock-physics-open/commit/3d505e39e5e8130dcb9a16bf67fa22c96d47768a))
* standardize names of input parameters ([#27](https://github.com/equinor/rock-physics-open/issues/27)) ([55126fd](https://github.com/equinor/rock-physics-open/commit/55126fd8e2f3d51c9baad3fb5f55a6a2e0499c38))

## [0.1.2](https://github.com/equinor/rock-physics-open/compare/v0.1.1...v0.1.2) (2025-05-09)


### Bug Fixes

* avoid building deps wheels ([e188e9d](https://github.com/equinor/rock-physics-open/commit/e188e9d84d95bad08040dff5411b020c0af1426d))
* check release please output ([275f13e](https://github.com/equinor/rock-physics-open/commit/275f13e018af560d5459e8ac779825de517f0feb))
* use id for step to catch output value ([681c5e3](https://github.com/equinor/rock-physics-open/commit/681c5e3e36fd90dfc43c704a3298688ea6745e05))

## [0.1.1](https://github.com/equinor/rock-physics-open/compare/v0.1.0...v0.1.1) (2025-05-09)


### Bug Fixes

* use pip wheel to build wheels ([7c7b9f4](https://github.com/equinor/rock-physics-open/commit/7c7b9f405309ad8be3c76f91028260936d842b05))

## 0.1.0 (2025-05-08)


### Features

* initial release ([1ecc1c2](https://github.com/equinor/rock-physics-open/commit/1ecc1c2f0bff534bcdc007d4951865c4c37d5435))
