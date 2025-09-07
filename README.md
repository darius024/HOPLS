# HOPLS - High-Performance Partial Least Squares

A high-performance implementation of Partial Least Squares (PLS) regression using JAX for numerical computations. This library provides both educational and production-ready implementations with comprehensive testing and documentation.

## Quick Start

### Installation
```bash
pip install -r requirements.txt
```

### Basic Usage

#### Command Line Interface
```bash
python main.py demo
python main.py run path/to/X_data.csv path/to/Y_data.csv
python main.py run X_data.txt Y_data.txt --components 3
```

#### Python API
```python
import pls
import numpy as np

X = np.loadtxt('X_data.csv', delimiter=',')
Y = np.loadtxt('Y_data.csv', delimiter=',')

model = pls.pls_regression(X, Y, n_components=2)
Y_pred = pls.predict(X, model)

print(f"R² Score: {1 - np.sum((Y - Y_pred)**2) / np.sum((Y - np.mean(Y))**2):.4f}")
```


## Project Structure

```
HOPLS/
├── main.py                      
├── demo.py                      
├── run_tests.py                 
├── pls/                         
│   ├── __init__.py             
│   ├── implementations/         
│   │   ├── basic/              
│   │   │   ├── helpers.py      
│   │   │   ├── pls_regression.py 
│   │   │   └── main.py         
│   │   └── optimized/          
│   │       ├── helpers.py      
│   │       └── pls_regression.py 
│   ├── src/
│   │   └── evaluation/         
│   └── examples/               
│       └── [various datasets]
└── tests/                      
    ├── test_quick.py          
    ├── test_comprehensive.py  
    ├── test_benchmark.py      
    └── test_mathematical_verification.py
```

## Usage Examples

### Linear Regression Alternative
```python
import pls
import numpy as np

X = np.random.randn(100, 5)
Y = np.random.randn(100, 2)

model = pls.pls_regression(X, Y, n_components=2)
Y_new = pls.predict(X_new, model)
```

### High-Dimensional Data
```python
X = np.random.randn(50, 200)   
Y = np.random.randn(50, 10)    

model = pls.pls_regression(X, Y, n_components=5)
```

### Using CSV Files
```python
import pls
import numpy as np

X = np.loadtxt('data/spectral_data.csv', delimiter=',')
Y = np.loadtxt('data/concentrations.csv', delimiter=',')

model = pls.pls_regression(X, Y, n_components=3)

Y_pred = pls.predict(X, model)
mse = np.mean((Y - Y_pred)**2)
print(f"Mean Squared Error: {mse:.6f}")
```

## Testing

```bash
python run_tests.py quick
python run_tests.py comprehensive
python run_tests.py benchmark
python run_tests.py all
```
