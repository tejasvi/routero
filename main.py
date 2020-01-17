import inspect
import textwrap
import streamlit as st
import pandas as pd
import pydeck as pdk


def demo():

#   def from_data_file(filename):
#       return pd.read_json(filename)

    DATA_URL = "https://raw.githubusercontent.com/uber-common/deck.gl-data/master/website/bart-lines.json"
    df = pd.read_json(DATA_URL)
    df = pd.read_json('fulldf.json')
    df = pd.DataFrame(df.iloc[:, -1])
 

    def hex_to_rgb(h):
        h = h.lstrip('#')
        return tuple(int(h[i:i+2], 16) for i in (0, 2, 4))

    layer = pdk.Layer(
        type='PathLayer',
        data=df,
        rounded=True,
        billboard=True,
        pickable=True,
        width_min_pixels=2,
        get_path='path',
    )

    st.pydeck_chart(pdk.Deck(
        map_style="mapbox://styles/mapbox/light-v9",
        initial_view_state={"latitude": 48.8566, "longitude": 2.3522, "zoom": 8, "pitch": 0},
        layers=[layer],
    ))

def overview():

#   def from_data_file(filename):
#       return pd.read_json(filename)

    DATA_URL = "https://raw.githubusercontent.com/uber-common/deck.gl-data/master/website/bart-lines.json"
    df = pd.read_json(DATA_URL)
    df = pd.read_json('thedf.json')
    df = pd.DataFrame(df.iloc[:, -1])
 

    def hex_to_rgb(h):
        h = h.lstrip('#')
        return tuple(int(h[i:i+2], 16) for i in (0, 2, 4))

    st.write(df)
    layer = pdk.Layer(
	"ScatterplotLayer",
	data=df,
	pickable=True,
	opacity=0.8,
	stroked=True,
	filled=True,
	radius_scale=6,
	radius_min_pixels=1,
	radius_max_pixels=100,
	line_width_min_pixels=1,
	get_position="coordinates",
	get_radius="exits_radius",
	get_fill_color=[255, 140, 0],
	get_line_color=[0, 0, 0],
        type='PathLayer',
        data=df,
        rounded=True,
        billboard=True,
        pickable=True,
        width_min_pixels=2,
        get_path='path',
    )

    st.pydeck_chart(pdk.Deck(
        map_style="mapbox://styles/mapbox/light-v9",
        initial_view_state={"latitude": 48.8566, "longitude": 2.3522, "zoom": 8, "pitch": 0},
        layers=[layer],
    ))

def run():

    st.write("# Working..")

    overview()

    st.markdown("## Code")
    sourcelines, _ = inspect.getsourcelines(demo)
    st.code(textwrap.dedent("".join(sourcelines[1:])))


if __name__ == "__main__":
    run()
