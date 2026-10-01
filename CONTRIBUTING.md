# Contributing to SeqFidelity

Use Python 3.10 or newer. Install the package with its test dependencies in your
chosen environment:

```bash
python -m pip install -e '.[test]'
python -m pytest
```

Keep changes focused on a documented numerical or input/output behavior. Add a
small, deterministic synthetic case when a change affects that behavior. Derive
expected values analytically where practical. Use CPU execution for the default
test suite and pass file destinations explicitly.

Write source comments, command-line help, and documentation in English. Preserve
third-party notices and explain changes to public configuration fields, tensor
shapes, units, and serialized model formats. Update the relevant API documentation
and example when an interface changes.

The test fixtures and examples are independently constructed toy inputs. Their
values document software behavior and have no interpretation as experimental
performance estimates.
