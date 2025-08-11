# pipeline-demo

A minimal, publishable Python package demonstrating a scikit-learn Pipeline with:
- Preprocessing via ColumnTransformer (imputation, scaling, one-hot encoding)
- A LogisticRegression classifier
- A small synthetic dataset packaged as CSV
- Two console scripts: training and pipeline HTML visualization

## Installation

Build and install locally:

```bash
python -m pip install --upgrade pip
python -m pip install build
python -m build
pip install dist/pipeline_demo-0.1.0-py3-none-any.whl
```

## Usage

Train and evaluate on the packaged data:

```bash
pipeline-demo-train
```

Optionally specify a CSV path:

```bash
pipeline-demo-train --data path/to/your.csv --test-size 0.2 --random-state 123
```

Export an HTML diagram of the pipeline to `pipeline_demo.html`:

```bash
pipeline-demo-viz
# or choose a path
pipeline-demo-viz --out my_pipeline.html
```

## Data

The packaged CSV lives at `pipeline_demo/data/synthetic_customers.csv` and has columns:
`age, income_gtq, monthly_visits, city, has_kids, churn`.

## License

MIT
