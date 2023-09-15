def recency_weight(timestamp, current_time, inverse_decay_rate=1):
    """
    Calculate recency weight for a given timestamp.
    
    Args:
        timestamp (datetime.datetime): The timestamp of the event.
        current_time (datetime.datetime): The current time or reference time.
        inverse_decay_rate (float): The decay rate for the time decay function. Must be greater than 0.5. Defaults to 1.
        inverse_decay_rate range: The higher the decay rate, the less the rate of decay and thus a narrow range of weights. The smaller the decay rate, the higher the rate of decay and thus a wider range of weights.
        Example: Highest decay rate can be obtained by setting inverse_decay_rate to 0.51. This gives exhorbitantly high weights to recently added products and very small weights to older products.

    Returns:
        float: The recency weight of the event.
    """
    time_difference = current_time - timestamp
    time_difference_in_days = time_difference.days
    if time_difference_in_days == 0:
        weight = 1  # Assign a maximum weight for events occurring on the same day
    else:
        weight = 1/np.log(inverse_decay_rate * time_difference_in_days)
    return weight

current_time = dt.datetime.now()

# Calculate recency weights for each event
df_clicks['recency_weight'] = df_clicks['original_timestamp'].apply(lambda x: recency_weight(x, current_time, inverse_decay_rate=1.3))
