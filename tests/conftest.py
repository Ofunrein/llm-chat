"""
Shared pytest fixtures.

test_app.py needs heavy deps stubbed (no GPU in CI).
test_transformer.py + test_tokenizer.py need real modules.

Solution: stub only when running test_app specifically,
using a session-scoped marker. Each test file manages its own sys.modules state.
"""
