import signal
from contextlib import contextmanager
import subprocess


class TimeoutException(Exception): pass


import sys


@contextmanager
def time_limit(seconds):
    def signal_handler(signum, frame):
        raise TimeoutException("Timed out!")

    signal.signal(signal.SIGALRM, signal_handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)


TIME_LIMIT = 300 * 10 ** 6  # 5min


def call_lemon(file, algorithm):
    res = subprocess.run(["bash", "-c", f"./build/src/lemon-project {file} {algorithm}"], capture_output=True)
    output = res.stdout.decode("utf-8")
    return output


if __name__ == "__main__":
    try:
        with time_limit(int(TIME_LIMIT / 10 ** 6)):
            print(call_lemon(sys.argv[1], sys.argv[2]))
    except TimeoutException:
        print(f"{sys.argv[1]} {TIME_LIMIT} {-1}")
