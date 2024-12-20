import subprocess
from util import TIME_LIMIT


def call_algorithm(algorithm, instance_data, cs2_path="../../cs2", lemon_path="../../lemon"):
    if algorithm == 1:  # Internal code for CS2 algorithm, which does not run over lemon
        try:
            res = subprocess.run(f'{cs2_path}/cs2', text=True, capture_output=True, input=instance_data,
                                 timeout=int(TIME_LIMIT / 10 ** 6))
            stderr = res.stderr
            if "Error 2" in stderr:
                output = "There is no feasible solution! Jumping to next file."
            elif stderr:
                output = "Encountered unknown cs2 problem! Aborting: " + stderr
            else:
                output = res.stdout
        except subprocess.TimeoutExpired:
            return True, ""
    else:
        try:
            res = subprocess.run(
                [f'{lemon_path}/build/src/lemon-project', str(algorithm)],
                text=True,
                capture_output=True,
                input=instance_data,
                timeout=int(TIME_LIMIT / 10 ** 6),
            )

            stderr = res.stderr
            if stderr:
                output = "Encountered unkown lemon problem! Aborting: " + stderr
            else:
                output = res.stdout
        except subprocess.TimeoutExpired:
            return True, ""
    return False, output
