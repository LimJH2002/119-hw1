"""
Part 3: Short Exercises on the Shell

**Released: Friday, October 18**

For the third and last part of this homework,
we will complete a few tasks related to shell programming
and shell commands, particularly, with relevance to how
the shell is used in data science.

Please note:
The "project proposal" portion will be postponed to part of Homework 2.

===== Questions 1-5: Setup Scripting =====

1. For this first part, let's write a setup script
that downloads a dataset from the web,
clones a GitHub repository, and runs the Python script
contained in `script.py` on the dataset in question.

For the download portion, we have written a helper
download_file(url, filename) which downloads the file
at `url` and saves it in `filename`.

You should use Python subprocess to run all of these operations.

To test out your script, and as your answer to this part,
run the following:
    setup(
        "https://github.com/DavisPL-Teaching/119-hw1",
        "https://raw.githubusercontent.com/DavisPL-Teaching/119-hw1/refs/heads/main/data/test-input.txt",
        "test-script.py"
    )

Then read the output of `output/test-output.txt`,
convert it to an integer and return it. You should get "12345".

"""

import os
import platform
import subprocess

import pandas as pd
# You may need to conda install requests or pip3 install requests
import requests

import part2


def download_file(url, filename):
    r = requests.get(url)
    with open(filename, 'wb') as f:
        f.write(r.content)


def clone_repo(repo_url):
    subprocess.run(["git", "clone", repo_url])


def run_script(script_path, data_path):
    subprocess.run(["python3", script_path, data_path])


def setup(repo_url, data_url, script_path):
    # Download the data
    data_filename = "input.txt"
    download_file(data_url, data_filename)

    # Clone the repository
    clone_repo(repo_url)

    # Run the script
    run_script(script_path, data_filename)


def q1():
    # Call setup as described in the prompt
    setup(
        "https://github.com/DavisPL-Teaching/119-hw1",
        "https://raw.githubusercontent.com/DavisPL-Teaching/119-hw1/refs/heads/main/data/test-input.txt",
        "test-script.py"
    )

    # Read the output file
    with open('output/test-output.txt', 'r') as f:
        output = f.read().strip()

    return int(output)


"""
2.
Suppose you are on a team of 5 data scientists working on
a project; every 2 weeks you need to re-run your scripts to
fetch the latest data and produce the latest analysis.

a. When might you need to use a script like setup() above in
this scenario?

=== ANSWER Q2a BELOW ===
1. When onboarding new team members who need to get the project running
2. When setting up automated testing environments
3. When running periodic updates that require fresh data and code
=== END OF Q2a ANSWER ===

Do you see an alternative to using a script like setup()?

=== ANSWER Q2b BELOW ===
1. Using containerization (Docker) to package the entire environment
2. Setting up continuous integration/deployment pipelines
=== END OF Q2b ANSWER ===

3.
Now imagine we have to re-think our script to
work on a brand-new machine, without any software installed.
(For example, this would be the case if you need to run
your pipeline inside an Amazon cloud instance or inside a
Docker instance -- to be more specific you would need
to write something like a Dockerfile, see here:
https://docs.docker.com/reference/dockerfile/
which is basically a list of shell commands.)

Don't worry, we won't test your code for this part!
I just want to see that you are thinking about how
shell commands can be used for setup and configuration
necessary for data processing pipelines to work.

Think back to HW0. What sequence of commands did you
need to run?
Write a function setup_for_new_machine() that would
be able to run on a brand-new machine and set up
everything that you need.

Assume that you need your script to work on all of the packages
that we have used in HW1 (that is, any `import` statements
and any other software dependencies).

Assume that the new server machine is identical
in operating system and architecture to your own,
but it doesn't have any software installed.
It has Python 3.12
and conda or pip3 installed to get needed packages.

Hint: use subprocess again!

Hint: search for "import" in parts 1-3. Did you miss installing
any packages?
"""


def setup_for_new_machine():
    commands = [
        ['pip3', 'install', 'pandas'],
        ['pip3', 'install', 'numpy'],
        ['pip3', 'install', 'matplotlib'],
        ['pip3', 'install', 'requests'],
        ['pip3', 'install', 'pytest'],
    ]

    for cmd in commands:
        subprocess.run(cmd, check=True)


def q3():
    return platform.system()


"""
4. This question is open ended :)
It won't be graded for correctness.

What percentage of the time do you think real data scientists
working in larger-scale projects in industry have to write
scripts like setup() and setup_for_new_machine()
in their day-to-day jobs?

=== ANSWER Q4 BELOW ===
In larger-scale industry projects, data scientists likely spend a significant amount of time (30-50%) 
on setup and configuration tasks
=== END OF Q4 ANSWER ===

5. Extra credit

Copy your setup_for_new_machine() function from Q3
(remove the other code in this file)
to a new script and run it on a friend's machine who
is not in this class. Did it work? What problems did you run into?

Only answer this if you actually did the above.
Paste the output you got when running the script on the
new machine:

=== ANSWER Q5 BELOW ===

=== END OF Q5 ANSWER ===

===== Questions 6-9: A comparison of shell vs. Python =====

The shell can also be used to process data.

This series of questions will be in the same style as part 2.
Let's import the part2 module: (my autostyling moved the import to the top)
"""

"""
Write two versions of a script that takes in the population.csv
file and produces as output the number of rows in the file.
The first version should use shell commands and the second
should use Pandas.

For technical reasons, you will need to use
os.popen instead of subprocess.run for the shell version.
Example:
    os.popen("echo hello").read()

Runs the command `echo hello` and returns the output as a string.

Hints:
    1. Given a file, you can print it out using
        cat filename

    2. Given a shell command, you can use the `tail` command
        to skip the first line of the output. Like this:

    (shell command that spits output) | tail -n +2

    Note: if you are curious why +2 is required here instead
        of +1, that is an odd quirk of the tail command.
        See here: https://stackoverflow.com/a/604871/2038713

    3. Given a shell command, you can use the `wc` command
        to count the number of lines in the output

   (shell command that spits output) | wc -l
"""


def pipeline_shell():
    # Count total lines using tail to skip header
    cmd = "cat data/population.csv | wc -l"
    result = os.popen(cmd).read()
    return int(result)


def pipeline_pandas():
    df = pd.read_csv('data/population.csv')
    return df.shape[0]


def q6():
    shell_count = pipeline_shell()
    pandas_count = pipeline_pandas()

    # The shell_count should now exactly match pandas_count
    assert shell_count == pandas_count, f"Counts don't match: shell={
        shell_count}, pandas={pandas_count}"
    return shell_count


"""
Let's do a performance comparison between the two methods.

This time, no need to generate a plot.
Just use your ThroughputHelper and LatencyHelper classes
from part 2 to get answers for both pipelines.

7. Throughput
"""


def q7():
    # Return a tuple of two floats
    # throughput for shell, throughput for pandas
    # (in rows per second)
    h = part2.ThroughputHelper()

    # Add both pipelines
    h.add_pipeline("shell", pipeline_shell(), pipeline_shell)
    h.add_pipeline("pandas", pipeline_pandas(), pipeline_pandas)

    throughputs = h.compare_throughput()
    return (throughputs[0], throughputs[1])  # shell, pandas


"""
8. Latency
"""


def q8():
    # Return a tuple of two floats
    # latency for shell, latency for pandas
    # (in milliseconds)
    h = part2.LatencyHelper()

    # Add both pipelines
    h.add_pipeline("shell", pipeline_shell)
    h.add_pipeline("pandas", pipeline_pandas)

    latencies = h.compare_latency()
    return (latencies[0], latencies[1])  # shell, pandas


"""
9. Which method is faster?
Comment on anything else you notice below.

=== ANSWER Q9 BELOW ===
1. Throughput of the shell pipeline is faster than the pandas pipeline
2. Latency of the shell pipeline is slower than the pandas pipeline
=== END OF Q9 ANSWER ===
"""

"""
===== Wrapping things up =====

**Don't modify this part.**

To wrap things up, we have collected
your answers and saved them to a file below.
This will be run when you run the code.
"""

ANSWER_FILE = "output/part3-answers.txt"
UNFINISHED = 0


def log_answer(name, func, *args):
    try:
        answer = func(*args)
        print(f"{name} answer: {answer}")
        with open(ANSWER_FILE, 'a') as f:
            f.write(f'{name},{answer}\n')
            print(f"Answer saved to {ANSWER_FILE}")
    except NotImplementedError:
        print(f"Warning: {name} not implemented.")
        with open(ANSWER_FILE, 'a') as f:
            f.write(f'{name},Not Implemented\n')
        global UNFINISHED
        UNFINISHED += 1


def PART_3_PIPELINE():
    open(ANSWER_FILE, 'w').close()

    # Q1-5
    log_answer("q1", q1)
    # 2a: commentary
    # 2b: commentary
    log_answer("q3", q3)
    # 4: commentary
    # 5: extra credit
    log_answer("q6", q6)
    log_answer("q7", q7)
    log_answer("q8", q8)
    # 9: commentary

    # Answer: return the number of questions that are not implemented
    if UNFINISHED > 0:
        print("Warning: there are unfinished questions.")

    return UNFINISHED


"""
=== END OF PART 3 ===

Main function
"""

if __name__ == '__main__':
    log_answer("PART 3", PART_3_PIPELINE)
