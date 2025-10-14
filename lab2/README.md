# Lab 2 (CSCI 360 - Introduction to Artificial Intelligence)

Lab instruction: [`lab2.pdf`](lab2.pdf). Due on Oct 14th, 2025, EOD PT.

**Submission**: Submit your lab2.py to Gradescope. Please keep the file name as **lab2.py** or else the autograder will fail.

Extra credit instruction: Submit a separate pdf `lab2-extra-credit.pdf` with your analysis, figures, code to generate the figures, etc. Note that extra credits **do not** overflow, meaning a final score >100pts will be treated as 100pts. Check [`lab2_extra_credits_rubrics.pdf`](lab2_extra_credits_rubrics.pdf) for details on extra credits grading.

## Setting up the environment

Follow the command line instruction below to initialize your repository
locally.

```bash
git clone <YOUR-GITHUB-REPO-HERE>
cd <YOUR-GITHUB-REPO-HERE>
conda create -n <env-name> python=3.9
conda activate <env-name>
pip install -r requirements.txt
```

## Writing Your Code

All of the code that will be evaluated and graded will live in [`lab2.py`](lab2.py).

You can use any Python package that is installed by `pip` or `conda` when you create the environment.

You are provided with the `TextbookStack` class. 
Same as lab 1, the constructor for this class expects two lists that represent the order. The first list `initial_order` is a list of length `n` that expects each integer `[0, n-1]` to be present once. The second list `initial_orientation` should be a list of length `n` of exclusively `0`s and `1`s, which represent the whether Textbook faces up.
For examples on how to use it, refer to [lab1](../lab1/README.md).

## Testing Your Code

We have given you an example test that you can run from the terminal by invoking:

```bash
python lab2.py
```
from inside your the directory. This is a starting point that reflect the same type of tests that we will run in order to evaluate and grade the correctness of your algorithm.

## How we will grade

### Main part (max 100 pts; problem 2)

- General soundness of your code (60 pts): You should provide full implementation of function `a_star_search` with comprehensible comments. If your code does not implement the required algorithm, we will deduct points from the 60 pts.
- Passing multiple test cases (40 pts)
    - Part 1 (24 pts): Your program should pass three example test cases with correct sequences. Each test case worths 8 pts.
    - Part 2 (16 pts): We will further test your implementation through all permutations of stacks.

### Extra credit (max 20 pts; problem 3)

Check [`lab2_extra_credits_rubrics.pdf`](lab2_extra_credits_rubrics.pdf) for details on extra credits grading.


All of your algorithm should be contained in the designated blocks inside of `a_star_search` (and `weighted_a_star_search` for extra credit) and should return the sequence of flips that your search algorithms find. Have fun!

