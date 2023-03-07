import subprocess
from util import TIME_LIMIT
from func_timeout import func_timeout, FunctionTimedOut


def call_algorithm(algorithm, instance_data):
    if algorithm == 1:  # Internal code for CS2 algorithm, which does not run over lemon
        res = subprocess.run(f'"../cs2/cs2"', capture_output=True, shell=True, input=instance_data)
        stderr = res.stderr.decode("utf-8")
        if "Error 2" in stderr:
            output = "There is no feasible solution! Jumping to next file."
        elif stderr:
            output = "Encountered unknown cs2 problem! Aborting: " + stderr
        else:
            output = res.stdout.decode("utf-8")
    else:
        res = subprocess.run(f'"../lemon/build/src/lemon-project" {algorithm}', capture_output=True, shell=True,
                             input=instance_data)
        stderr = res.stderr.decode("utf-8")
        if stderr:
            output = "Encountered unkown lemon problem! Aborting: " + stderr
        else:
            output = res.stdout.decode("utf-8")
    return output


def call_algorithm_timeout(algorithm, instance_data):
    try:
        return False, func_timeout(int(TIME_LIMIT / 10 ** 6), call_algorithm, args=(algorithm, instance_data))
    except FunctionTimedOut:
        return True, ""
