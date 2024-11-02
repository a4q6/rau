import json
import os
from copy import deepcopy
from glob import glob
from hashlib import sha256
from os import PathLike
from pathlib import Path
from typing import (
    Any,
    Callable,
    Collection,
    Dict,
    List,
    Optional,
    Sequence,
    Tuple,
    Union,
)

import dill
import pandas as pd
import pyarrow
import pyarrow.parquet
import yaml
from IPython.display import HTML, Image, display
from pyarrow import Table


def flatten(items: Union[List, Tuple]):
    """Flatten(1D-nize) given nested List|Tuple"""

    assert isinstance(items, (list, tuple))
    if len(items) == 0:
        return items
    if isinstance(items[0], (list, tuple)):
        return flatten(items[0]) + flatten(items[1:])
    return items[:1] + flatten(items[1:])

def read_hdf_keys(path: os.PathLike) -> List[str]:
    with pd.HDFStore(path, "r") as s:
        hdf_keys = s.keys()
    return hdf_keys

def read_hdf(path) -> Union[Dict[str, Any], Any]:
    keys = read_hdf_keys(path)

    if len(keys) == 0:
        raise ValueError("No keys found. Probably empty HDF file.")

    if len(keys) == 1:
        return pd.read_hdf(path)

    else:
        res = {}
        for key in keys:
            res[key] = pd.read_hdf(path, key)
        return res

def read_text(path: os.PathLike) -> str:
    with open(path, "r") as f:
        text = f.read().splitlines()
    return text

def to_text(obj: str, path: os.PathLike) -> str:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        f.write(obj)

def read_json(path: PathLike) -> Dict:
    with open(path, "r") as f:
        data = json.load(f)
    return data

def to_json(obj, path: os.PathLike) -> str:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(obj, "w") as f:
        json.dump(obj, f)

def read_yaml(path):
    with open(path, "r") as f:
        content = yaml.safe_load(f)
    return content

def read_pickle(path: PathLike) -> Any:
    return dill.load(open(path, "rb"))

def to_pickle(obj: Any, path: os.PathLike) -> str:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        dill.dump(obj, f)

def read_csv(path) -> pd.DataFrame:
    return pd.read_csv(path)

def read_html(path: PathLike) -> None:
    display(HTML(filename=path))

def _write_attrs(table: Table, df: pd.DataFrame):

    schema_meta = table.schema.metadata or {}
    # extract pandas metadata
    pandas_meta = json.loads(schema_meta.get(b"pandas", "{}"))
    column_attrs = {}
    for col in df.columns:
        attrs = df[col].attrs
        if not attrs or not isinstance(col, str):
            continue
        column_attrs[col] = attrs
    pandas_meta.update(
        attrs=df.attrs, column_attrs=column_attrs,
    )
    # override metadatas
    schema_meta[b"pandas"] = json.dumps(pandas_meta)
    return table.replace_schema_metadata(schema_meta)


def _read_attrs(table: Table, df: pd.DataFrame):
    schema_meta = table.schema.metadata or {}
    pandas_meta = json.loads(schema_meta.get(b"pandas", "{}"))
    df.attrs = pandas_meta.get("attrs", {})
    col_attrs = pandas_meta.get("column_attrs", {})
    for col in df.columns:
        df[col].attrs = col_attrs.get(col, {})


def read_parquet(path: PathLike) -> pd.DataFrame:
    """Load DataFrame in parquet with metadata.

    Args:
        path (PathLike):

    Returns:
        pd.DataFrame:
    """
    table = pyarrow.parquet.read_pandas(path)
    df = table.to_pandas()
    _read_attrs(table, df)
    return df


def to_parquet(df: pd.DataFrame, path: PathLike, compression="zstd"):
    """Save DataFrame in apache parquet format.

    Args:
        df (pd.DataFrame):
        path (PathLike):
        compression (str, optional): Defaults to "zstd".
    """
    table = pyarrow.Table.from_pandas(df)
    table = _write_attrs(table, df)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    pyarrow.parquet.write_table(table, path, compression=compression)

def delete_from_list(base_list: Sequence, delete_target: Sequence) -> List:
    """delete sequence from sequence."""
    if not isinstance(delete_target, (set, list, tuple)):
        delete_target = [delete_target]
    return [item for item in base_list if item not in delete_target]

def to_hash(obj: Any) -> str:
    if isinstance(obj, set):
        return tuple(sorted([to_hash(e) for e in obj]))
    elif isinstance(obj, (tuple, list)):
        return tuple([to_hash(e) for e in obj])
    elif not isinstance(obj, dict):
        return sha256(repr(obj).encode("utf-8")).hexdigest()

    new_obj = deepcopy(obj)
    for k, v in new_obj.items():
        new_obj[k] = to_hash(v)

    return sha256(json.dumps(new_obj, sort_keys=True).encode("utf-8")).hexdigest()

def load(path: PathLike) -> Any:
    """load file on memory. file handler will chosen based on file format.

    Args:
        path (PathLike):

    Returns:
        Any:
    """
    path = Path(path)
    file_fmt = ".".join(path.as_posix().split("/")[-1].split(".")[1:])
    if any([fmt in file_fmt for fmt in [".hdf5", ".h5", ".hdf"]]):
        return read_hdf(path.as_posix())
    elif any([fmt in file_fmt for fmt in [".parquet", ".pqt"]]):
        return read_parquet(path.as_posix())
    elif any([fmt in file_fmt for fmt in [".csv"]]):
        return read_csv(path.as_posix())
    elif any([fmt in file_fmt for fmt in [".json"]]):
        return read_json(path.as_posix())
    elif any([fmt in file_fmt for fmt in [".yaml", ".yml"]]):
        return read_yaml(path.as_posix())
    elif any([fmt in file_fmt for fmt in [".pkl", ".pickle"]]):
        return read_pickle(path.as_posix())
    elif any([fmt in file_fmt for fmt in [".html", ".htm"]]):
        read_html(path.as_posix())
    elif any([fmt in file_fmt for fmt in [".png"]]):
        display(Image(filename=path.as_posix()))
    elif any([fmt in file_fmt for fmt in [".txt", ".text"]]):
        return read_text(path.as_posix())
    else:
        raise ValueError(f"File format `{file_fmt}` is not supported.")

def save(obj: Any, path: PathLike) -> None:
    """Save given object into file. File format is automatically handled with `path`.

    Args:
        obj (Any): Target object.
        path (PathLike): File path to store given object.
    """
    path = Path(path)
    file_fmt = ".".join(path.as_posix().split("/")[-1].split(".")[1:])
    if any([fmt in file_fmt for fmt in [".hdf5", ".h5", ".hdf"]]):
        return obj.to_hdf(path.as_posix())
    elif any([fmt in file_fmt for fmt in [".parquet", ".pqt"]]):
        return to_parquet(path.as_posix())
    elif any([fmt in file_fmt for fmt in [".csv"]]):
        return obj.to_csv(path.as_posix())
    elif any([fmt in file_fmt for fmt in [".json"]]):
        return to_json(obj, path.as_posix())
    # elif any([fmt in file_fmt for fmt in [".yaml", ".yml"]]):
    #     return read_yaml(path.as_posix())
    elif any([fmt in file_fmt for fmt in [".pkl", ".pickle"]]):
        to_pickle(obj, path.as_posix())
    elif any([fmt in file_fmt for fmt in [".txt", ".text"]]):
        to_text(obj, path.as_posix())
    else:
        raise ValueError(f"File format `{file_fmt}` is not supported.")
