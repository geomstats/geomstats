# Backend Interface

In order to expose a new backend function/attribute to the rest of the
codebase, it is necessary to add the name to the respective list in the
`BACKEND_ATTRIBUTES` dictionary in `geomstats/_backend/__init__.py`.
This serves two purposes:

1. Define a clear boundary between backend interface and backend-internal code:
   Only functions/attributes which are used outside the backend should be made
   available to the rest of the codebase.
1. Guarantee each backend exposes the same attributes:
   When loading a backend, the backend importer verifies that a backend
   provides each attribute listed in the `BACKEND_ATTRIBUTES` dict.
   This way, we guarantee that unit tests fail during CI builds when a
   maintainer/contributor forgets to provide an implementation of a feature for
   a particular backend.
   If a feature cannot be supported for some reason, the function should raise
   a `NotImplementedError` for the time being.
