import pandas as pd
import os
import numpy as np

# ----- EXTRACT BASE UPSTREAM COURIER AND ORDERS DATA -----

def courier_upstream(df: pd.DataFrame):    
    """Extracts courier data from waybill data

    Args:
        df (pd.DataFrame): waybill data containing (time, courier_id, order_id) triplets

    Returns:
        pd.DataFrame: waybill data filtered for courier data
    """
    # filter for ones where is_courier_grabbed == 1 since those have the courier (lng, lat) data, otherwise
    # those are 0
    courier = df.loc[df.is_courier_grabbed == 1, ['dt', 'dispatch_time', 'courier_id','grab_lng','grab_lat', 'is_weekend']]
    courier.rename(columns={'dispatch_time': 'time'}, inplace=True)
    
    # reverse shifted (lng, lat) into normal degree range
    courier['grab_lng'] = courier['grab_lng'] / 1e6
    courier['grab_lat'] = courier['grab_lat'] / 1e6
    return courier    
    

def orders_upstream(df: pd.DataFrame):
    """extracts orders data from waybill data

    Args:
        df (pd.DataFrame): waybill data containing (time, courier_id, order_id) triplets

    Returns:
        pd.DataFrame: waybill data filtered for orders data
    """
    # since for each order (order_id), it may come up multiple times if previous couriers rejected it,
    # we grab the order_id instance where is_courier_grabbed == 1 to avoid duplicates
    orders = df.loc[df.is_courier_grabbed == 1, ['dt','order_push_time','order_id','sender_lng', 'sender_lat', 'da_id', 'poi_id', 'is_prebook', 'is_weekend']]
    orders.rename(columns={'order_push_time': 'time'}, inplace=True)
    
    # reverse shifted (lng, lat) into normal degree range
    orders['sender_lng'] = orders['sender_lng'] / 1e6
    orders['sender_lat'] = orders['sender_lat'] / 1e6
    return orders


def upstream_pipeline(waybill_df: pd.DataFrame, save_folder: str):
    """

    Args:
        waybill_df (pd.DataFrame): contains triplets of (time, courier_id, order_id) data that we want to split
            into courier and orders dataframes
    """    
    
    # convert time columns to datetime
    time_cols = ['platform_order_time', 'order_push_time', 'dispatch_time', 'estimate_meal_prepare_time', 'estimate_arrived_time', 'grab_time', 'fetch_time', 'arrive_time']
    for t_col in time_cols:
        waybill_df[t_col] = pd.to_datetime(waybill_df[t_col], unit='s')
        
    # get rid of entries where date is too early
    waybill_df = waybill_df[waybill_df['dispatch_time'] > pd.to_datetime('1970-01-01 00:00:00')]    
    
    # extract courier and orders data    
    courier = courier_upstream(waybill_df)
    orders = orders_upstream(waybill_df)
    
    # save the data
    courier.to_csv(os.path.join(save_folder, 'courier.csv'), index=False)
    orders.to_csv(os.path.join(save_folder, 'orders.csv'), index=False)
    
    return courier, orders
    
    
# ----- CREATE TIMEBLOCKS OF COURIER AND ORDERS FOR MATCHING -----
def chunk_by_timeblocks(
    df: pd.DataFrame, 
    time_col: str, 
    time_increment: pd.Timedelta,     
    start_time: pd.Timestamp = None,
):    
    df = df.copy() # copy to avoid modifying original
    df[time_col] = pd.to_datetime(df[time_col]) # convert to datetime just in case
    
    # get starting increment time
    if start_time is None:
        min_time = df[time_col].min()
        start_time = (min_time.floor(time_increment))  # round down to nearest increment
    else:
        # remove any rows that are before the start time
        df = df[df[time_col] >= start_time]
        
    # create timeblock
    df["timeblock"] = ((df[time_col] - start_time) // time_increment)
    df["timeblock_start"] = start_time + ((df[time_col] - start_time) // time_increment) * time_increment

    df = df.sort_values(by='timeblock')
    
    return df
    
    
def compare_chunk_id_overlap(
    df: pd.DataFrame, 
    id_col: str, 
    timeblock_col: str="timeblock", 
    timeblock_start_col: str="timeblock_start"
    ):
    """Given a dataframe with timeblocks, calculate the overlap between successive timeblocks

    Args:
        df (pd.DataFrame): dataframe with timeblocks
        id_col (str): column name of the ids
        timeblock_col (str, optional): timeblock column. Defaults to "timeblock".
        timeblock_start_col (str, optional): actual time start of the timeblock. Defaults to "timeblock_start_col".

    Returns:
        pd.DataFrame: dataframe with overlap metrics
    """
    block_df = (
        df.groupby([timeblock_col, timeblock_start_col])[id_col]
        .agg(set)
        .reset_index()
        .rename(columns={id_col: "ids", 
                        timeblock_col: "timeblock", 
                        timeblock_start_col: "timeblock_start"})
        .sort_values(timeblock_col)
    )
    block_df["ids_ct"] = block_df["ids"].apply(len)
    block_df["next_timeblock"] = block_df["timeblock"].shift(-1)
    block_df["next_ids"] = block_df["ids"].shift(-1)
    block_df = block_df[(block_df.timeblock + 1) == block_df.next_timeblock] # successive timeblocks only

    # calculate overlap metrics
    block_df["overlap_ids"] = block_df.apply(lambda row: row["ids"] & row["next_ids"], axis=1)
    block_df["overlap_count"] = block_df["overlap_ids"].apply(len)
    block_df["overlap_pct"] = block_df["overlap_count"] / block_df["ids_ct"]
        
    return block_df[["timeblock", "timeblock_start", "ids_ct", "next_timeblock", "overlap_count", "overlap_pct"]]


# ----- MATCH COURIER AND ORDERS DATA -----
# Haversine formula (vectorized)
def haversine_distance_matrix(coords1, coords2, radius=6371000):
    """
    Compute pairwise haversine distances between two sets of (lat, lon) coordinates.
    Inputs in degrees, output in meters.
    """
    coords1 = np.radians(coords1)  # shape (n, 2)
    coords2 = np.radians(coords2)  # shape (n, 2)

    lat1, lon1 = coords1[:, 0][:, np.newaxis], coords1[:, 1][:, np.newaxis]
    lat2, lon2 = coords2[:, 0][np.newaxis, :], coords2[:, 1][np.newaxis, :]

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    return radius * c  # Output shape: (n, n)


def courier_order_chunk_matching():
    pass


def chunk_and_save_by_timeblocks(
    df: pd.DataFrame,
    time_col: str,
    time_increment: pd.Timedelta,
    output_dir: str,
    start_time: pd.Timestamp = None,
    prefix: str = "timeblock"
):
    """
    Splits the DataFrame into timeblocks and saves each block as a separate parquet file.

    Args:
        df (pd.DataFrame): Input DataFrame.
        time_col (str): Name of the datetime column.
        time_increment (pd.Timedelta): Duration of each timeblock (e.g., pd.Timedelta('1h')).
        output_dir (str): Directory to save the parquet files.
        start_time (pd.Timestamp, optional): Rounded start time to begin timeblocks. If None, inferred from data.
        prefix (str, optional): Prefix for the output files.

    Returns:
        List of output file paths.
    """
    df = df.copy()
    df[time_col] = pd.to_datetime(df[time_col])
    if start_time is None:
        min_time = df[time_col].min()
        start_time = (min_time.floor(time_increment))  # round down to nearest increment

    os.makedirs(output_dir, exist_ok=True)
    df["timeblock"] = ((df[time_col] - start_time) // time_increment)

    output_files = []
    for block_id, group in df.groupby("timeblock"):
        block_start = start_time + block_id * time_increment
        filename = f"{prefix}_{block_start.strftime('%Y%m%dT%H%M%S')}.parquet"
        filepath = os.path.join(output_dir, filename)
        group.drop(columns="timeblock").to_parquet(filepath, index=False)
        output_files.append(filepath)

    return output_files
