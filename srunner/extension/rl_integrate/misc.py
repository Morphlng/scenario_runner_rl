from __future__ import annotations

import numpy as np


def str2bool(v: str):
    """Convert a string to boolean.

    Args:
        v (str): string to be converted.

    Returns:
        bool: converted boolean.
    """
    v = v.lower()
    if v in ("y", "yes", "t", "true", "on", "1"):
        return True
    elif v in ("n", "no", "f", "false", "off", "0"):
        return 0
    else:
        raise ValueError("invalid truth value %r" % (v,))


def str2list(
    string: str,
    split_with: str = ",",
    filter_empty: bool = True,
    convert_to: type = str,
) -> list:
    """Parse a string to a list of variables

    Args:
        string (str): String to parse
        filter_empty (bool, optional): Whether to filter empty strings. Defaults to True.
        convert_to (type, optional): Type to convert each string to. Defaults to str.
    """
    if string == "" or string is None:
        return []

    return list(
        filter(
            lambda x: x != "" or not filter_empty,
            map(lambda x: convert_to(x.strip()), string.split(split_with)),
        )
    )


def get_attributes(
    obj: object, include_callable: bool = False, include_private: bool = False
):
    """Get all attributes of an object.

    Args:
        obj (object): object to be checked.
        include_callable (bool): whether to include callable attributes.
        include_private (bool): whether to include private attributes.

    Returns:
        dict: attributes of the object.
    """
    attrs = {}
    for attr in dir(obj):
        value = getattr(obj, attr)
        if not include_private and attr.startswith("_"):
            continue
        if not include_callable and callable(value):
            continue
        attrs[attr] = value
    return attrs


def one_hot_dict(ids_list: "list | set", first_id: str = None):
    """One-hot encoding for a list of ids.

    Args:
        ids_list (list | set): list of ids.
        first_id (str): rearrange the ids_list to make the first_id as the first element.

    Returns:
        dict: one-hot encoding dict.
    """
    ids_list = list(ids_list)

    if first_id is not None:
        try:
            ids_list.remove(first_id)
            ids_list.insert(0, first_id)
        except ValueError:
            pass

    one_hot_array = np.eye(len(ids_list))
    one_hot_dict = {}
    for i, id in enumerate(ids_list):
        one_hot_dict[id] = one_hot_array[i].tolist()
    return one_hot_dict


def flatten_dict(dicionary: dict):
    values = []
    for key in dicionary.keys():
        data = dicionary[key]
        if isinstance(data, dict):
            values.extend(flatten_dict(data))
        elif isinstance(data, (list, np.ndarray)):
            values.extend(np.array(data).flatten())
        else:
            values.append(data)

    return np.array(values, dtype=np.float32).flatten()
