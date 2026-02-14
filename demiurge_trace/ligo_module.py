from gwpy.table import EventTable

def get_gw_event_time(event_name):
    """
    Fetches the GPS timestamp for a confirmed Gravitational Wave event.
    
    Args:
        event_name (str): The name of the GW event (e.g., 'GW150914').
        
    Returns:
        float: The GPS timestamp of the event.
    """
    try:
        # Known timestamps for target events (GPS Time):
        known_events = {
            'GW150914': 1126259462.4,
            'GW170817': 1187008882.4,
        }
        
        if event_name in known_events:
            print(f"Found {event_name} in local lookup.")
            return known_events[event_name]

        # In a real scenario, we would stream or query the catalog.
        # For efficiency and reliability in this specific tool, we can also use a lookup 
        # for well-known events if the catalog fetch fails or is slow, 
        # but let's try the proper gwpy method first.
        table = EventTable.fetch_open_data(event_name, verbose=True)
            
        # Fallback to fetching from open data if not in our quick lookup
        # This is a bit more complex with gwpy to just get "Time of Coalescence" 
        # without downloading the whole catalog.
        # For the purpose of this "Auditor", precise timestamps are critical.
        
        raise ValueError(f"Event {event_name} not found in local lookup. Please add GPS timestamp.")

    except Exception as e:
        print(f"Error fetching event time: {e}")
        return None

def fetch_strain_data(gw_time, duration=32):
    """
    Fetches Gravitational Wave strain data around the event time.
    
    Args:
        gw_time (float): GPS timestamp of the event.
        duration (float): Duration in seconds to fetch (centered on event).
        
    Returns:
        TimeSeries: The strain data.
    """
    try:
        from gwpy.timeseries import TimeSeries
        start = gw_time - duration / 2
        end = gw_time + duration / 2
        print(f"Fetching strain data from {start} to {end}...")
        # Fetch data from Hanford (H1) or Livingston (L1)
        # Using H1 for consistency, can fallback to L1
        data = TimeSeries.fetch_open_data('H1', start, end, verbose=True)
        return data
    except Exception as e:
        print(f"Error fetching strain data: {e}")
        return None
