# TSP Learning Lab

A Streamlit teaching application for exploring the **Travelling Salesperson Problem (TSP)**.

The app is designed to help students understand the complete journey from mathematical formulation to solution, computational complexity, and interpretation.

## Features

- Editable city coordinates
- Editable distance matrix
- Symmetric and asymmetric TSP
- Mathematical formulation of the TSP
- Brute-force enumeration
- Nearest Neighbour heuristic
- 2-opt local search
- Comparison of solution methods
- Optimality-gap calculation
- Computational-complexity demonstration
- Interpretation of solutions

## Solution Methods

### Brute Force

Brute force enumerates all possible tours and identifies the optimal solution.

For a symmetric TSP with a fixed starting city, the number of unique tours is:

\[
\frac{(n-1)!}{2}
\]

For an asymmetric TSP with a fixed starting city, the number of tours is:

\[
(n-1)!
\]

Brute force guarantees the optimal solution, but the number of tours grows very rapidly as the number of cities increases.

### Nearest Neighbour

The Nearest Neighbour heuristic starts from a selected city and repeatedly chooses the nearest unvisited city.

It is computationally fast and easy to understand, but it does **not** guarantee an optimal solution.

### 2-opt

2-opt starts from an existing tour, such as the Nearest Neighbour solution, and attempts to improve it by exchanging pairs of edges.

It can improve the initial heuristic solution substantially, but it may stop at a local optimum rather than the global optimum.

## Teaching Objectives

The application helps students understand:

- how a TSP is formulated mathematically;
- the role of decision variables, objective functions, and constraints;
- the difference between exact and heuristic solution methods;
- why heuristic methods may not produce the global optimum;
- how local search can improve an initial solution;
- the difference between symmetric and asymmetric TSP;
- how problem size affects computational effort;
- why TSP becomes computationally difficult as the number of cities increases;
- how to interpret optimization results rather than only obtain a numerical answer.

## Inputs

The app supports two types of input.

### 1. Coordinates

Students can modify the X and Y coordinates of each city.

The app automatically calculates Euclidean distances between cities.

Because Euclidean distance satisfies:

\[
d(i,j)=d(j,i)
\]

this produces a **symmetric TSP**.

### 2. Distance Matrix

Students can directly enter travel distances or costs between cities.

This allows both:

- symmetric TSP, where \(d(i,j)=d(j,i)\);
- asymmetric TSP, where \(d(i,j)\neq d(j,i)\).

The starting city can also be changed.

## Computational Complexity

The app demonstrates how the number of possible tours increases rapidly with the number of cities.

For a symmetric TSP:

\[
\frac{(n-1)!}{2}
\]

unique tours must be considered when the starting city is fixed and reverse tours are treated as equivalent.

For an asymmetric TSP:

\[
(n-1)!
\]

tours must be considered.

This allows students to observe why exhaustive enumeration becomes impractical for larger problems.

## Running the App Locally

Install the required packages:

```bash
pip install -r requirements.txt
```

Run the Streamlit application:

```bash
streamlit run tsp_learning_lab_app.py
```

Alternatively:

```bash
python -m streamlit run tsp_learning_lab_app.py
```

## Files

```text
tsp-learning-lab/
├── tsp_learning_lab_app.py
├── requirements.txt
└── README.md
```

## Intended Use

This application is intended for classroom teaching and demonstration in courses related to:

- Mathematical Modelling
- Optimization
- Operations Research
- Graph Theory
- Heuristics and Local Search
- Computational Complexity

## Future Extensions

Possible extensions include:

- Branch and Bound
- Integer Programming / MILP formulation
- Subtour visualization
- Held-Karp dynamic programming
- OR-Tools routing solver
- Comparison of additional heuristics
- Larger benchmark instances
- Downloadable experiment results
- Student exercises and quizzes
