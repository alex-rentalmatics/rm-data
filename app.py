import streamlit as st
import pandas as pd
import pydeck as pdk
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime

st.title("📍 GPS Data Explorer with Bearing Arrows & Timestamps")

# File upload
uploaded_file = st.file_uploader(
    "Upload a CSV file with Latitude, Longitude, Bearing, and optionally 'date time' or 'epoch' columns",
    type=["csv"]
)

if uploaded_file is not None:
    # Read CSV
    try:
        # Try reading without specific date parsing first
        df = pd.read_csv(uploaded_file)
    except Exception as e:
        st.error(f"Error reading CSV file: {e}")
        st.stop()

    # --- Column Name Normalization (Optional but recommended) ---
    # Make column names lowercase and replace spaces for easier access
    df.columns = df.columns.str.lower().str.replace(' ', '_')

    # --- Ensure required base columns exist ---
    required_core = {'latitude', 'longitude', 'bearing'}
    # Check using the normalized column names
    if not required_core.issubset(df.columns):
        st.error(f"CSV must contain at least these columns (case-insensitive): Latitude, Longitude, Bearing")
        st.stop()

    # --- Data Cleaning & Conversion ---
    # Rename columns if they are not already normalized (e.g., if input was 'Latitude')
    # This step is less critical now due to normalization above, but good practice
    df = df.rename(columns={'lat': 'latitude', 'lon': 'longitude',
                           'long': 'longitude'}) # Add common alternatives

    # Convert core numeric columns
    df['latitude'] = pd.to_numeric(df['latitude'], errors='coerce')
    df['longitude'] = pd.to_numeric(df['longitude'], errors='coerce')
    df['bearing'] = pd.to_numeric(df['bearing'], errors='coerce')

    # Convert optional numeric columns
    df['speed_(kph)'] = pd.to_numeric(df.get('speed_(kph)'), errors='coerce') # Use normalized name
    df['altitude'] = pd.to_numeric(df.get('altitude'), errors='coerce')

    # Drop rows where essential coordinates or bearing are missing
    df = df.dropna(subset=['latitude', 'longitude', 'bearing'])
    if df.empty:
        st.warning("No valid data points found after cleaning (Latitude, Longitude, Bearing).")
        st.stop()


    # --- Timestamp Handling ---
    df['timestamp_str'] = "N/A" # Default value
    has_timestamp = False

    # Prioritize 'date_time' column
    if 'date_time' in df.columns:
        try:
            # Attempt parsing with the specified format
            df['datetime_parsed'] = pd.to_datetime(df['date_time'], format='%d/%m/%Y %H:%M:%S', errors='coerce')

            # Check if parsing was successful for at least some rows
            if df['datetime_parsed'].notna().any():
                 # Format for display, handle NaT values resulting from parsing errors
                df['timestamp_str'] = df['datetime_parsed'].dt.strftime('%Y-%m-%d %H:%M:%S').fillna("Invalid Date Format")
                st.info("Using 'date_time' column for timestamps.")
                has_timestamp = True
            else:
                st.warning("Found 'date_time' column, but failed to parse values using format DD/MM/YYYY HH:MM:SS. Checking for 'epoch'.")
        except Exception as e:
            st.warning(f"Error processing 'date_time' column: {e}. Checking for 'epoch'.")

    # Fallback to 'epoch' column if 'date_time' wasn't successful
    if not has_timestamp and 'epoch' in df.columns:
        # Assume epoch is in milliseconds
        df['epoch'] = pd.to_numeric(df['epoch'], errors='coerce')
        if df['epoch'].notna().any():
            try:
                # Convert milliseconds to seconds for pd.to_datetime
                df['datetime_parsed'] = pd.to_datetime(df['epoch'] / 1000, unit='s', errors='coerce')
                 # Format for display, handle NaT values
                df['timestamp_str'] = df['datetime_parsed'].dt.strftime('%Y-%m-%d %H:%M:%S').fillna("Invalid Epoch Value")
                st.info("Using 'epoch' column (assuming milliseconds) for timestamps.")
                has_timestamp = True
            except Exception as e:
                 st.warning(f"Found 'epoch' column, but failed to parse values: {e}")
        else:
            st.warning("Found 'epoch' column, but it contains non-numeric or missing values.")

    if not has_timestamp:
        st.info("No valid 'date_time' or 'epoch' column found for timestamps.")

    # --- Compute direction vectors for arrows ---
    def bearing_to_vector(bearing, scale=0.0001): # Adjust scale as needed for arrow length
        if pd.isna(bearing):
            return 0, 0
        rad = np.deg2rad(bearing)
        # Note: Latitude corresponds to Y, Longitude to X
        dx = scale * np.sin(rad) # Change in Longitude
        dy = scale * np.cos(rad) # Change in Latitude
        return dx, dy

    # Apply the function safely
    vectors = df['bearing'].apply(lambda b: bearing_to_vector(b, scale=0.0001)) # Adjust scale if arrows are too long/short
    df['dx'] = vectors.apply(lambda v: v[0])
    df['dy'] = vectors.apply(lambda v: v[1])

    df['target_lat'] = df['latitude'] + df['dy']
    df['target_lon'] = df['longitude'] + df['dx']

    # --- Map Visualization ---
    st.subheader("🧭 Point and Bearing Visualization on Map")

    # Layer for the actual points (circles) - Tooltip will attach here
    point_layer = pdk.Layer(
        "ScatterplotLayer",
        data=df,
        get_position='[longitude, latitude]',
        get_radius=1, # Radius of points in pixels
        get_fill_color=[0, 150, 255, 180], # Blue points
        pickable=True, # Enable hover tooltip
        auto_highlight=False,
    )

    # Layer for the bearing arrows (lines)
    arrow_layer = pdk.Layer(
        "LineLayer",
        data=df,
        get_source_position='[longitude, latitude]',
        get_target_position='[target_lon, target_lat]',
        get_width=3, # Width of lines in pixels
        get_color=[255, 255, 0, 150], # Yellow lines, slightly transparent
        pickable=False, # No separate tooltip for the line itself
    )

    # Calculate initial view state
    mid_lat = df['latitude'].mean()
    mid_lon = df['longitude'].mean()

    if pd.isna(mid_lat) or pd.isna(mid_lon):
        st.warning("Could not calculate center of map due to missing coordinate data.")
        # Provide default coordinates if calculation fails
        mid_lat = 0
        mid_lon = 0


    view_state = pdk.ViewState(
        latitude=mid_lat,
        longitude=mid_lon,
        zoom=15, # Slightly closer default zoom
        pitch=45,
    )

    # Tooltip configuration
    tooltip_html = """
        <b>Timestamp:</b> {timestamp_str}<br/>
        <b>Lat:</b> {latitude}<br/>
        <b>Lon:</b> {longitude}<br/>
        <b>Bearing:</b> {bearing}° <br/>
        <small><i>(Speed: {speed_(kph)} kph, Alt: {altitude} m)</i></small>
    """
    # Fill missing optional values for tooltip display
    df_display = df.fillna({'speed_(kph)': 'N/A', 'altitude': 'N/A'})

    tooltip_data = {
        "html": tooltip_html,
        "style": {
            "backgroundColor": "steelblue",
            "color": "white"
        }
    }

    # Map style selection
    map_style = st.selectbox("🗺️ Select map style", options=[
        "Streets", "Dark", "Light", "Satellite", "Outdoors", "None"
    ])

    style_dict = {
        "Streets": "mapbox://styles/mapbox/streets-v11",
        "Dark": "mapbox://styles/mapbox/dark-v9",
        "Light": "mapbox://styles/mapbox/light-v9",
        "Satellite": "mapbox://styles/mapbox/satellite-v9",
        "Outdoors": "mapbox://styles/mapbox/outdoors-v11",
        "None": None,
    }

    # Render the deck.gl map
    st.pydeck_chart(pdk.Deck(
        layers=[point_layer, arrow_layer], # Point layer first so arrows don't obscure hover
        initial_view_state=view_state,
        map_style=style_dict[map_style],
        tooltip=tooltip_data # Use the prepared tooltip config
    ))

    # --- Plots ---
    # Speed plot
    st.subheader("🚗 Speed Over Time")
    # Use normalized column name and handle potential missing column
    speed_col = 'speed_(kph)'
    if speed_col in df.columns:
        speed_df = df[[speed_col]].dropna() # Select only the speed column for plotting
        if not speed_df.empty:
            fig, ax = plt.subplots()
            ax.plot(speed_df[speed_col].reset_index(drop=True))
            ax.set_xlabel("Data Point Index")
            ax.set_ylabel("Speed (kph)")
            ax.grid(True)
            st.pyplot(fig)
        else:
            st.info("No valid speed data available to plot.")
    else:
        st.info("Speed column ('speed_(kph)') not found in the uploaded file.")


    # Altitude plot
    st.subheader("⛰️ Altitude Over Time")
    alt_col = 'altitude'
    if alt_col in df.columns:
        alt_df = df[[alt_col]].dropna()
        if not alt_df.empty:
            fig2, ax2 = plt.subplots()
            ax2.plot(alt_df[alt_col].reset_index(drop=True), color='green')
            ax2.set_xlabel("Data Point Index")
            ax2.set_ylabel("Altitude (m)")
            ax2.grid(True)
            st.pyplot(fig2)
        else:
             st.info("No valid altitude data available to plot.")
    else:
        st.info("Altitude column ('altitude') not found in the uploaded file.")

    # Raw data table
    st.subheader("📊 Data Table")
    # Show relevant columns including the parsed timestamp string
    display_cols = ['latitude', 'longitude', 'bearing', 'timestamp_str']
    # Add optional columns if they exist
    if 'speed_(kph)' in df.columns: display_cols.append('speed_(kph)')
    if 'altitude' in df.columns: display_cols.append('altitude')
    # Filter df to only include existing columns before displaying
    st.dataframe(df[[col for col in display_cols if col in df.columns]])

else:
    st.info("Please upload a CSV file to begin.")