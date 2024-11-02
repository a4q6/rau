import subprocess
from pathlib import Path
from typing import Dict, Optional

import pandas as pd


def run_notebook(
    notebook_path: str, generate_html=False, html_path: Optional[str] = None
):
    """Run notebook with `nbconvert` command.

    Args:
        notebook_path (str):
        generate_report (bool):
        html_path (Optional[str]): Put html with same filename on current dir by default.
    """

    fname = notebook_path.split("/")[-1].rstrip(".ipynb")
    if generate_html:
        # run notebook and generate html.
        if html_path is None:
            html_path = f"{fname}.html"
        if Path(html_path).is_dir():
            html_path = Path(html_path).joinpath(f"{fname}.html")

        html_path = Path(html_path).absolute()
        cmd = f"""jupyter nbconvert {notebook_path} --ExecutePreprocessor.timeout=7200 --execute --no-input --to html --output {html_path}"""
    else:
        # only run target notebook and overwrite.
        cmd = f"""jupyter nbconvert {notebook_path} --ExecutePreprocessor.timeout=7200 --execute --inplace"""

    try:
        subprocess.check_output(cmd, shell=True, stderr=subprocess.STDOUT)

        path = notebook_path if html_path is None else html_path
        return path

    except subprocess.CalledProcessError as e:
        print(e.output.decode("utf-8"))
        raise e


def generate_from_template(
    template_path: str, notebook_path: str, params: Dict[str, str] = None
):
    """Generate new `ipynb` from <template>.ipynb. "<PARAM>" in notebook content will be replaced as "VALUE". Only string is supported.

    Args:
        template_path (str):
        notebook_path (str):
        params (Dict[str, str]): Variables should be expressed as "<PARAM_NAME>" in code blocks. They will be replaces as "VALUE".
            example: {"<PARAM>": "hoge"}
    """
    if params is None:
        params = {}

    # load
    with open(template_path) as f:
        content = f.read()

    # edit
    for param_name, value in params.items():
        content = content.replace(f"{param_name}", f"{value}")

    # write
    with open(notebook_path, "w") as f:
        f.write(content)

    return notebook_path


def pd_set_options():
    pd.set_option("display.max_rows", 100)
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", None)
    pd.set_option("display.max_colwidth", None)
