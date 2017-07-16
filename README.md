## What is agios?
agios is an open source, Python 3 library for playing with genetic algorithms. Main functionality includes:   
* Generic API allowing to easily implement custom tasks
* Highly customizable algorithm execution cycle
* Multithreading support
* Built-in support for images processing
* Multidimensional data processing
   
TODO list includes:
* Support for PyCUDA and processing on GPU

## How to install it?
```
pip install agios
```

## Where is an example code?
```python
from agios import evolution
from agios import extras

blueprint = extras.load_normalized_image('input/mona_lisa.jpg', extras.Greyscale)

evolution_problem_solver = evolution.SimpleSolver(
    population_size=100,
    best_samples_to_take=2,
    blueprint=evolution.NumpyArraySample(blueprint),
    mutator=evolution.SimplePaintbrushMatrixMutator((10, 15), (10, 50)),
    crosser=evolution.MeanValueMatrixCrosser(),
    loss_calculator=evolution.SquaredMeanMatrixLossCalculator(),
    initial_sample_state_generator=evolution.RandomMatrixGenerator(blueprint.shape)
)

for _ in range(10000):
    evolution_problem_solver.step()
```

Live examples can be found in examples/ directory.

## How to contribute?
Report observed issues or provide working pull request. Pull request must be verified before merging and it must include the following:
* Unit tests
* Public API marked with static typing annotations (typing module)
* Public classes must include brief documentation
