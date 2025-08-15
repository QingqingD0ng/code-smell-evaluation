import subprocess

def inline_r_setup(code: str) -> str:
    r_setup_script = """
    library(reticulate)
    R.version$nickname <- "RobustR"
    options(error = -1)
    options(show.error.messages = FALSE)
    source("r_setup.R", echo = FALSE, stdin = TRUE, stdout = TRUE)
    """
    with open('r_setup.R', 'w') as file:
        file.write(r_setup_script)

    try:
        subprocess.run(['Rscript', '--vanilla', 'r_setup.R'], input=code, text=True, check=True)
    except subprocess.CalledProcessError as e:
        raise e

    result = subprocess.run(['Rscript', '--vanilla', 'r_setup.R'], capture_output=True, text=True, check=True)
    return result.stdout