import streamlit as st
import pandas as pd
import pydeck as pdk
import matplotlib.pyplot as plt
import numpy as np

st.title("📍 GPS Data Explorer with Bearing Arrows")

# File upload
uploaded_file = st.file_uploader("Upload a CSV file with Latitude, Longitude, Bearing columns", type=["csv"])

if uploaded_file is not None:
    # Read CSV
    df = pd.read_csv(uploaded_file)

    # Ensure required columns exist
    required = {'Latitude', 'Longitude', 'Bearing'}
    if not required.issubset(df.columns):
        st.error(f"CSV must contain at least these columns: {', '.join(required)}")
        st.stop()

    # Clean & convert data
    df['Speed (kph)'] = pd.to_numeric(df.get('Speed (kph)'), errors='coerce')
    df['Altitude'] = pd.to_numeric(df.get('Altitude'), errors='coerce')
    df['Bearing'] = pd.to_numeric(df['Bearing'], errors='coerce')

    df = df.rename(columns={'Latitude': 'latitude', 'Longitude': 'longitude'})

    # Compute direction vectors for arrows
    def bearing_to_vector(bearing, scale=0.0001):
        rad = np.deg2rad(bearing)
        dx = scale * np.sin(rad)
        dy = scale * np.cos(rad)
        return dx, dy

    df = df[df['Bearing'].notna()]
    df['dx'], df['dy'] = zip(*df['Bearing'].map(bearing_to_vector))
    df['target_lat'] = df['latitude'] + df['dy']
    df['target_lon'] = df['longitude'] + df['dx']

    # Map with bearing arrows
    st.subheader("🧭 Bearing Visualization on Map")
    arrow_layer = pdk.Layer(
        "LineLayer",
        data=df,
        get_source_position='[longitude, latitude]',
        get_target_position='[target_lon, target_lat]',
        get_width=4,
        get_color=[255, 255, 0],
        pickable=True,
    )
    view_state = pdk.ViewState(
        latitude=df['latitude'].mean(),
        longitude=df['longitude'].mean(),
        zoom=16,
        pitch=45,
    )

    # Optional: select map style
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

    st.pydeck_chart(pdk.Deck(
        layers=[arrow_layer],
        initial_view_state=view_state,
        map_style=style_dict[map_style]
    ))

    # Speed plot
    st.subheader("🚗 Speed Over Time")
    speed_df = df[df['Speed (kph)'].notna()]
    if not speed_df.empty:
        fig, ax = plt.subplots()
        ax.plot(speed_df['Speed (kph)'].reset_index(drop=True))
        ax.set_xlabel("Sample")
        ax.set_ylabel("Speed (kph)")
        ax.grid(True)
        st.pyplot(fig)
    else:
        st.info("No speed data available to plot.")

    # Altitude plot
    st.subheader("⛰️ Altitude Over Time")
    alt_df = df[df['Altitude'].notna()]
    if not alt_df.empty:
        fig2, ax2 = plt.subplots()
        ax2.plot(alt_df['Altitude'].reset_index(drop=True), color='green')
        ax2.set_xlabel("Sample")
        ax2.set_ylabel("Altitude (m)")
        ax2.grid(True)
        st.pyplot(fig2)
    else:
        st.info("No altitude data available to plot.")

    # Raw data table
    st.subheader("📊 Data Table")
    st.dataframe(df[['latitude', 'longitude', 'Speed (kph)', 'Altitude', 'Bearing']])
else:
    st.info("Please upload a CSV file to begin.")
