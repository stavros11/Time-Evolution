"""Utilities that read setup files for `main.py`.

Setup files can be used as an alternative to the argparser. These should have
the following structure:
  * main argument = value for this argument,
  * next main argument = value,
  etc.
"""

from typing import Dict, Union


def isfloat(value: str) -> bool:
  """Checks if a string represents a float number."""
  try:
    float(value)
    return True
  except ValueError:
    return False


def read_main_setup(file_dir: str) -> Dict[str, Union[str, bool, int, float]]:
  """Transforms an input setup file to the dictionary with `main` args.

  Args:
    file_dir: The directory of the setup file.

  Returns:
    Dictionary with `main` arguments as the keys and their corresponding values.
  """
  with open(file_dir, "r") as file:
    text = file.read()
    settings = text.replace("\n", "").replace(" ","").split(",")
    d = {}
    for setting in settings:
        k, v = setting.split("=")
        if v.isdigit():
          d[k] = int(v)
        elif v in {"True", "False"}:
          d[k] = (v == "True")
        elif isfloat(v):
          d[k] = float(v)
        else: # v is simply a string
          d[k] = v.replace('"', '')
    return d