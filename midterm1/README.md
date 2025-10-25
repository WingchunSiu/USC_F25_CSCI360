# Midterm 1 (CSCI 360 - Introduction to Artificial Intelligence)

Lab instruction: See Problem 8 in `Midterm 1 Fall 25.pdf`. Due on Oct 24th, 2025, EOD PT.

**Submission**: Submit your **midterm1_astar.py** to Gradescope. Please keep the file name as **midterm1_astar.py** or else the autograder will fail.

## Setting up the environment

Follow the command line instruction below to initialize your repository locally.

```bash
git clone <YOUR-GITHUB-REPO-HERE>
cd <YOUR-GITHUB-REPO-HERE>
conda create -n <env-name> python=3.9
conda activate <env-name>
pip install -r requirements.txt
```

## Writing Your Code

All of the code that will be evaluated and graded will live in [`midterm1_astar.py`](midterm1_astar.py).

You can use any Python package that is installed by `pip` or `conda` when you create the environment.

## Testing Your Code

We have given you the test cases that you can run from the terminal by invoking:

```bash
python midterm1_astar.py
```

from inside your the directory.

All of your algorithm should be contained in the designated blocks inside of `a_star_search` and should return the optimal sequence of coordinates and the corresponding cost that your search algorithms find. Have fun!
: