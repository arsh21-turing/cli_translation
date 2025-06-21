# Conftest with custom hook to tolerate test helpers that `return True`

def pytest_pyfunc_call(pyfuncitem):
    """Allow test functions to return True/None for success.

    The upstream quality-learning system tests use `return True` / `return False`
    instead of assertions. PyTest treats any non-None return as a failure. This
    hook converts a *truthy* return value to a successful test outcome (None) and
    converts a *falsy* return value to an AssertionError so the test fails.
    """
    import inspect
    sig = inspect.signature(pyfuncitem.obj)
    needed = {name: pyfuncitem.funcargs[name] for name in sig.parameters if name in pyfuncitem.funcargs}
    outcome = pyfuncitem.obj(**needed)
    if outcome is None:
        return True  # regular success – let PyTest continue
    if bool(outcome):
        # Truthy -> success, we swallow the return value
        return True
    # Falsy -> explicit failure
    import pytest
    pytest.fail(f"Test function {pyfuncitem.name} returned a falsy value")
    return True 