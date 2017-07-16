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
It is meant to reproduce Mona Lisa in greyscale. Firstly it loads actual Mona Lisa as a blueprint.
Then it creates the Solver and performs 10000 steps in order to paint it. 
Creating a Solver includes a few configuration choices. 
You must choose how many samples are included in population, and how many of them are took as the best samples per each iteration.
You must specify a blueprint for which total loss will be calculated, in this case it's an image represented as a NumPy array.
Mutator is a class for changing the sample in some way. In this case simple implementation of paintbrush tool is used.
Crosser is a class for combining two samples into one, MeanValueMatrixCrosser computes average value for each element of array.
LossCalculator is a way of calculating total loss for sample (how far it is from a blueprint).
RandomMatrixGenerator defines how initial samples are generated, in this case they are just random.
   
More live examples can be found in examples/ directory.

## How to contribute?
Report observed issues or provide working pull request. Pull request must be verified before merging and it must include the following:
* Unit tests
* Public API marked with static typing annotations (typing module)
* Public classes must include brief documentation
