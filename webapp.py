from dash import Dash
import dash_mantine_components as dmc
import dash_bootstrap_components as dbc
from dash import Input, Output, html, dcc
from datetime import datetime, date
from dash_iconify import DashIconify
import requests
import joblib
import pandas as pd

import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import dash_loading_spinners as dls


BASE_PROPHET_MODEL_PATH = "prophet_model"
BASE_ISOLATION_FOREST_MODEL_PATH = "isolation_forest_model"
PAST_DATA = "past_data/cities_dict.joblib"


external_stylesheets = [
    "https://fonts.googleapis.com/css2?family=Work+Sans:wght@500&display=swap",
    "https://tailwindcss.com/",
    {"src": "https://cdn.tailwindcss.com"},
    dbc.themes.BOOTSTRAP,
]

CITIES = [
    "balikpapan",
    "bandung",
    "batam",
    "jakarta",
    "makassar",
    "medan",
    "palembang",
    "pekanbaru",
    "surabaya",
    "yogyakarta",
]
CONTENT_STYLE = {"margin-left": "12rem", "margin-right": "2rem", "padding": "2rem 1rem"}

select_city_components = dmc.Tooltip(
    label="Select City",
    children=dmc.Select(
        placeholder="City",
        id="city_select",
        value="balikpapan",
        data=[{"value": f"{city}", "label": f"{city}"} for city in CITIES],
        style={"width": 200},
    ),
)

date_picker = dmc.Tooltip(
    label="Select Date to Indicate Outlier",
    children=dmc.DatePicker(
        id="date-picker",
        minDate=date(2020, 11, 19),
        value=datetime.now().date(),
        style={"width": 200},
    ),
)
y_true_input = dmc.Tooltip(
    label="Price Change",
    children=dmc.NumberInput(
        value="",
        id="ytrue-price",
        precision=2,
        step=5,
        icon=[DashIconify(icon="fa6-solid:weight-scale")],
        style={"width": 250},
    ),
)

group_input = dmc.Group(
    position="center",
    direction="row",
    spacing="xl",
    children=[select_city_components, date_picker, y_true_input],
)
plot_btn = dmc.Button(
    "Plot", id="plot-btn", style={"width": "100px", "margin-top": "30px"}
)
app = Dash(__name__, external_stylesheets=external_stylesheets)


app.layout = html.Div(
    children=[
        dbc.Row(
            children=[
                dbc.Row(
                    html.H3(
                        "Anomaly Detection Kenaikan Harga Cabe Merah",
                        style={"text-align": "center"},
                    )
                ),
                dbc.Row(group_input),
                html.Br(),
                html.Br(),
                dbc.Row(dmc.Group(plot_btn, position="center", direction="row")),
                dls.Hash(
                    dbc.Row(
                        id="outlier-content",
                        style={"margin-top": "20px", "margin-left": "12rem"},
                    ),
                    color="#228be6",
                    show_initially=False,
                ),
            ]
        )
    ],
    style=CONTENT_STYLE,
)


@app.callback(
    Output("outlier-content", "children"),
    [
        Input("city_select", "value"),
        Input("date-picker", "value"),
        Input("ytrue-price", "value"),
        Input("plot-btn", "n_clicks"),
    ],
)
def draw_outlier_graph(city, date_, price_change, n_clicks):
    if n_clicks > 0:

        ENDPOINT_EXACT_DATE = f"https://cabaimerahanomalyapi-production.up.railway.app/nowcasting_price/{city}/{date_}/{price_change}"
        exact_date_df = pd.read_json(requests.get(ENDPOINT_EXACT_DATE).json())
        ENDPOINT_HISTORIC = f"https://cabaimerahanomalyapi-production.up.railway.app/past_outlier_data/{city}"
        historic_df = pd.read_json(requests.get(ENDPOINT_HISTORIC).json())
        combined_data = pd.concat([exact_date_df, historic_df])
        combined_data["ds"] = pd.to_datetime(combined_data["ds"])
        combined_data = combined_data.sort_values("ds")
        print(combined_data)

        def generate_outlier_plot(combined_data, city=city):
            figure_outlier = go.Figure()
            # parsed_data = combined_data[[city]].reset_index()
            trace1 = go.Scatter(
                y=combined_data["ytrue"],
                x=combined_data.ds,
                mode="lines",
                name=f"{city}",
            )
            figure_outlier.add_trace(trace=trace1)
            # #get anotated result
            filter_date = combined_data[f"outlier"] == "outlier"
            list_date = combined_data.loc[filter_date, "ds"].to_list()
            parsed_outlier_filter = combined_data["ds"].isin(list_date)
            parsed_data_outlier_df = combined_data.loc[parsed_outlier_filter]
            trace2 = go.Scatter(
                y=parsed_data_outlier_df["ytrue"],
                x=parsed_data_outlier_df.ds,
                mode="markers",
                name="anomaly",
            )
            figure_outlier.add_trace(trace=trace2)
            figure_outlier.update_layout(
                title=f"Deteksi Anomali Cabai Merah di {city}",
                width=1000,
                height=500,
                margin=dict(l=0, r=0, b=0, t=0),
                xaxis_rangeslider_visible=True,
            )
            figure_outlier.add_vline(
                x=pd.to_datetime(date_),
                line_width=3,
                line_dash="dash",
                line_color="green",
            )
            return figure_outlier

        val_outlier = exact_date_df["outlier"].values.squeeze()
        text_outlier = "anomaly" if val_outlier == "outlier" else "normal"
        container = [
            dbc.Row(dcc.Graph(figure=generate_outlier_plot(combined_data))),
            dbc.Row(
                dmc.Text(
                    f"Kenaikan Harga Sebesar Rp{price_change} pada tanggal {date_} di {city} berada dalam level {text_outlier}",
                    style={"margin-top": "30px"},
                )
            ),
        ]
        return container


if __name__ == "__main__":
    app.run_server(debug=True, port=7088, dev_tools_hot_reload=True)
