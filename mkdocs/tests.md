# Running Tests

## Run all tests

```bash
python -m pytest tests/ -v
```

## With coverage

```bash
python -m pytest tests/ --cov=pyscsa --cov-report=html
```

## Test specific module

```bash
python -m pytest tests/test_scsa1d.py -v
```

View coverage: `open htmlcov/index.html`
