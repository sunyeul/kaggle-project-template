import numpy as np
import pandas as pd
import polars as pl

from logging import getLogger


logger = getLogger(__name__)


def reduce_memory_usage_pd(df: pd.DataFrame, verbose=True) -> pd.DataFrame:
    """
    Reduce memory usage by converting data types in a pandas DataFrame.
    For compatibility with feather, float16 is not used.

    Parameters:
        df (pd.DataFrame): DataFrame to be compressed.
        verbose (bool): If True, logs the reduction in memory usage.

    Returns:
        pd.DataFrame: DataFrame with reduced memory usage.
    """
    num_dtypes = ["int16", "int32", "int64", "float32", "float64"]
    start_mem_usage = df.memory_usage().sum() / 1024**2

    for col in df.columns:
        col_type = df[col].dtype
        if col_type in num_dtypes:
            c_min, c_max = df[col].min(), df[col].max()
            if str(col_type).startswith("int"):
                if c_min >= np.iinfo(np.int8).min and c_max <= np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif (
                    c_min >= np.iinfo(np.int16).min and c_max <= np.iinfo(np.int16).max
                ):
                    df[col] = df[col].astype(np.int16)
                elif (
                    c_min >= np.iinfo(np.int32).min and c_max <= np.iinfo(np.int32).max
                ):
                    df[col] = df[col].astype(np.int32)
                elif (
                    c_min >= np.iinfo(np.int64).min and c_max <= np.iinfo(np.int64).max
                ):
                    df[col] = df[col].astype(np.int64)
            else:
                if (
                    c_min >= np.finfo(np.float32).min
                    and c_max <= np.finfo(np.float32).max
                ):
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)

    end_mem_usage = df.memory_usage().sum() / 1024**2
    if verbose:
        logger.warning(
            f"Memory reduced from {start_mem_usage:.2f} MB to {end_mem_usage:.2f} MB"
        )

    return df


def reduce_memory_usage_pl(
    df: pl.DataFrame, name: str, verbose: bool = True
) -> pl.DataFrame:
    """
    Reduce memory usage of a polars DataFrame by changing its data types.

    Parameters:
        df (pl.DataFrame): Polars DataFrame to be compressed.
        name (str): Name of the DataFrame for logging purposes.

    Returns:
        pl.DataFrame: DataFrame with reduced memory usage.
    """
    start_mem_usage = df.estimated_size("mb")
    numeric_int_types = [pl.Int8, pl.Int16, pl.Int32, pl.Int64]
    numeric_float_types = [pl.Float32, pl.Float64]

    for col in df.columns:
        col_type = df.dtypes[col]
        c_min = df[col].min()
        c_max = df[col].max()
        if col_type in numeric_int_types:
            if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                df = df.with_columns(df[col].cast(pl.Int8))
            elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                df = df.with_columns(df[col].cast(pl.Int16))
            elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                df = df.with_columns(df[col].cast(pl.Int32))
            elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                df = df.with_columns(df[col].cast(pl.Int64))
        elif col_type in numeric_float_types:
            if c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                df = df.with_columns(df[col].cast(pl.Float32))
        elif col_type == pl.Utf8:
            df = df.with_columns(df[col].cast(pl.Categorical))

    end_mem_usage = df.estimated_size("mb")
    if verbose:
        logger.warning(
            f"Memory reduced from {start_mem_usage:.2f} MB to {end_mem_usage:.2f} MB"
        )

    return df
