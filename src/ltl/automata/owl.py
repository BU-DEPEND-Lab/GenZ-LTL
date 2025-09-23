import subprocess

OWL_PATH = 'owl-21.0/bin/owl'


def run_owl(formula: str) -> str:
    """Convert an LTL formula to a NBA in the HOA format."""
    command = [OWL_PATH, 'ltl2nba', '-f', formula]
    run = subprocess.run(command, capture_output=True, text=True)
    if run.stderr != '':
        raise RuntimeError(f'OWL call `{" ".join(command)}` resulted in an error.\nError: {run.stderr}.')
    return run.stdout


if __name__ == '__main__':
    f = 'FG a'
    ldba = run_owl(f)
    print(ldba)
