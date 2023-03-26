import subprocess
from util import TIME_LIMIT


def call_algorithm(algorithm, instance_data):
    if algorithm == 1:  # Internal code for CS2 algorithm, which does not run over lemon
        try:
            res = subprocess.run(f'"../cs2/cs2"', capture_output=True, input=instance_data,
                                 timeout=int(TIME_LIMIT / 10 ** 6))
            stderr = res.stderr.decode("utf-8")
            if "Error 2" in stderr:
                output = "There is no feasible solution! Jumping to next file."
            elif stderr:
                output = "Encountered unknown cs2 problem! Aborting: " + stderr
            else:
                output = res.stdout.decode("utf-8")
        except subprocess.TimeoutExpired:
            return True, ""
    else:
        try:
            res = subprocess.run(f'"../lemon/build/src/lemon-project" {algorithm}', capture_output=True,
                                 input=instance_data, timeout=int(TIME_LIMIT / 10 ** 6))

            stderr = res.stderr.decode("utf-8")
            if stderr:
                output = "Encountered unkown lemon problem! Aborting: " + stderr
            else:
                output = res.stdout.decode("utf-8")
        except subprocess.TimeoutExpired:
            return True, ""
    return False, output
