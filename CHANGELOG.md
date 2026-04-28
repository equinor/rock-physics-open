# Changelog

## [1.0.0](https://github.com/equinor/rock-physics-open/compare/v0.6.1...v1.0.0) (2026-04-28)


### ⚠ BREAKING CHANGES

* resolve pyright ignores for first-party code
* remove snapshot utilities, replace with `syrupy`
* **t_matrix_models:** remove unused rho_fluid parameter from carbonate_pressure_model
* remove unused parameters

### Features

* add temperature and psi conversion functions to units module ([6011292](https://github.com/equinor/rock-physics-open/commit/6011292d3c2966107dc28b9ddd1064afe01442e4))
* add units module with conversion functions from rock-physics ([7d52300](https://github.com/equinor/rock-physics-open/commit/7d523004fd9dd77a7072a3d2b9b370614b991537))


### Bug Fixes

* **classification_functions:** correct dtype annotations for class IDs ([782e641](https://github.com/equinor/rock-physics-open/commit/782e64194de0b36596f5d747c6831ce4816d92c0)), closes [#148](https://github.com/equinor/rock-physics-open/issues/148)
* **classification:** correct poly_class type annotations ([ed774c5](https://github.com/equinor/rock-physics-open/commit/ed774c5a35267324d80809bd639345cb1b3755a6))
* **t_matrix_models:** avoid mutating pressure inputs in carbonate_pressure_model ([25941e4](https://github.com/equinor/rock-physics-open/commit/25941e4c14b244e3966c2e55fbd8fb886d7e8459)), closes [#141](https://github.com/equinor/rock-physics-open/issues/141)
* **t_matrix_models:** drop cast in calc_pressure_vec return ([6e5c179](https://github.com/equinor/rock-physics-open/commit/6e5c179b5f0511b0613207fa46a18953ee139e7f))


### Documentation

* standardize docstrings to NumPy convention and add pydoclint ([fd8bd3a](https://github.com/equinor/rock-physics-open/commit/fd8bd3a3d788b8e3ac5f0fa669d5cfd67b2f32be)), closes [#53](https://github.com/equinor/rock-physics-open/issues/53)


### Code Refactoring

* remove unused parameters ([54d0be6](https://github.com/equinor/rock-physics-open/commit/54d0be6dcbb9cd370097105ff396c448ef16704e))
* resolve pyright ignores for first-party code ([78a222f](https://github.com/equinor/rock-physics-open/commit/78a222fcb4b397e2bd0d05609d2705b3127adf83))
* **t_matrix_models:** remove unused rho_fluid parameter from carbonate_pressure_model ([8e9aab8](https://github.com/equinor/rock-physics-open/commit/8e9aab841eeacd43c2c7944d7bbc077c535126e1)), closes [#144](https://github.com/equinor/rock-physics-open/issues/144)


### Tests

* remove snapshot utilities, replace with `syrupy` ([a68beef](https://github.com/equinor/rock-physics-open/commit/a68beef25750076236813d7ef36cde3a14b68136))

## [0.6.1](https://github.com/equinor/rock-physics-open/compare/v0.6.0...v0.6.1) (2026-03-11)


### Bug Fixes

* **span_wagner:** avoid RuntimeWarning in vapor_pressure for above-critical temperatures ([7ebedac](https://github.com/equinor/rock-physics-open/commit/7ebedace60f87bdaf57e4560f3544b1c70449511))
* **std_functions:** suppress divide-by-zero warning in velocity when vs=0 ([f282218](https://github.com/equinor/rock-physics-open/commit/f282218a03dfb41934f032c0edf4dc0ea52c01dc))
* Use github app to get PAT token for release-please ([2c72d32](https://github.com/equinor/rock-physics-open/commit/2c72d322505d8cee426ab5183bc3d5b3462225bf))

## [0.6.0](https://github.com/equinor/rock-physics-open/compare/v0.5.0...v0.6.0) (2026-02-20)


### Features

* add Han and Batzle 2000 model for live oil ([ad80e6f](https://github.com/equinor/rock-physics-open/commit/ad80e6f7bd46d1d91409abe4d04c4d09b9dbbe7d))
* add inverse of bubble point pressure for maximum gas/oil ratio ([5d0f079](https://github.com/equinor/rock-physics-open/commit/5d0f079663b709f364d2ae08f8a15c417964154f))
* set Han and Batzle live oil model as default ([1f7709f](https://github.com/equinor/rock-physics-open/commit/1f7709fcb0b4af813af116d37e19c45d4b172d29))


### Bug Fixes

* explicit 'out' setting in numpy greater ([e9fa323](https://github.com/equinor/rock-physics-open/commit/e9fa323ef0e2d093c6147c94e76659a02747d5b6))
* implement consistent typing ([6dd12b3](https://github.com/equinor/rock-physics-open/commit/6dd12b3f988bb1605fcac37dd6d0766e0e7727ac))
* return numpy array ([5e41099](https://github.com/equinor/rock-physics-open/commit/5e410998944c6461e0d629fdb1b178fef2572ce6))


### Documentation

* add copilot coding agent instructions ([077e650](https://github.com/equinor/rock-physics-open/commit/077e65047f3f5bba09bce1c285173c847ec70dfb))
* Add sympy removal transition docs ([9e19b4a](https://github.com/equinor/rock-physics-open/commit/9e19b4a5d22dde3f10a3e53b46135ba213d54f51))
* **readme:** add PyPI version badge ([6332c4e](https://github.com/equinor/rock-physics-open/commit/6332c4e2da2100812d732ae7852e9ad3a8adc675))

## [0.5.0](https://github.com/equinor/rock-physics-open/compare/v0.4.0...v0.5.0) (2026-02-16)


### Features

* Support python 3.14 ([8d53696](https://github.com/equinor/rock-physics-open/commit/8d53696ad80d413ff124544ec400c2ecba74ba6d))


### Bug Fixes

* Remove workaround for fixed pandas bug ([#57](https://github.com/equinor/rock-physics-open/issues/57)) ([2ec1707](https://github.com/equinor/rock-physics-open/commit/2ec17071710845ff06a08f28ca6c2ef05c479cb2))

## [0.4.0](https://github.com/equinor/rock-physics-open/compare/v0.3.5...v0.4.0) (2026-01-22)


### Features

* Add typing span_wanger and ternary_plots ([2896d7b](https://github.com/equinor/rock-physics-open/commit/2896d7b7988faf34a581e7e08e906bf8548cc0b3))


### Bug Fixes

* Add type annotations for tmatrix ([f2c97fa](https://github.com/equinor/rock-physics-open/commit/f2c97fa8286da8335781145b46012a79947c9d1c))
* Fully type-annotate `rock-physics-open` ([9a90289](https://github.com/equinor/rock-physics-open/commit/9a9028967fc3bb404a8e8a2c8acfb88d9b88233e))

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
