import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash import (
    Dash,
    dcc,
    html,
    Input,
    Output,
)

app = Dash(__name__)
app.title = "US Earthquakes Visualization"

colors = {"background": "#f1f1f1", "text": "#6B7597"}

# Data Preparation
earthquakes = pd.read_csv(
    "datasets/us_earthquakes.csv",
    parse_dates=["time"],
)


earthquakes["magType"] = earthquakes["magType"].astype("category")
earthquakes["type"] = earthquakes["type"].astype("category")
earthquakes["status"] = earthquakes["status"].astype("category")
earthquakes["locationSource"] = earthquakes["locationSource"].astype("category")
earthquakes["state"] = earthquakes["state"].astype("category")
earthquakes["state_code"] = earthquakes["state_code"].astype("category")
earthquakes["month"] = earthquakes["month"].astype("category")

# Data Preparation - Heatmap
heatmap_df = earthquakes.copy()
heatmap_df = heatmap_df.groupby(["month", "state_code"]).agg(
    earthquakes=("id", "count")
)
heatmap_df = heatmap_df.reset_index()

# Data Preparation - Scatterplot
scatter_plot_df = earthquakes[
    [
        "id",
        "time",
        "latitude",
        "longitude",
        "mag",
        "place",
        "type",
        "state_code",
        "month",
    ]
]
scatter_plot_df = scatter_plot_df.assign(date=scatter_plot_df["time"].dt.date)

# App layout
app.layout = html.Div(
    id="dashboard",
    style={"backgroundColor": colors["background"]},
    children=[
        # Header
        html.Div(
            id="header",
            style={"text-align": "center", "color": colors["text"]},
            children=[
                html.H1("US Earthquakes Visualization"),
            ],
        ),
        # Sub Header
        html.Div(
            id="subheader",
            style={
                "text-align": "center",
                "color": colors["text"],
            },
            children="Visualization of earthquakes in the states of the US",
        ),
        # Dropdown
        html.Div(
            style={
                "display": "flex",
                "alignItems": "center",
                "color": colors["text"],
                "fontWeight": "bold",
            },
            children=[
                html.P("Select month:"),
                # Dropdown
                dcc.Dropdown(
                    id="month-selection",
                    options=["January", "February", "March"],
                    value="January",
                    style={"width": "30%", "marginLeft": "10px"},
                ),
            ],
        ),
        # Map
        html.Div(
            id="map",
            style={
                "backgroundColor": "#fff",
                "display": "flex",
                "flexDirection": "row",
                "height": "700px",
                "width": "100%",
            },
            children=[
                # Heatmap
                dcc.Graph(id="state-choropleth", figure={}, style={"width": "60%"}),
                # Scatter Plot
                dcc.Graph(id="state-scatter-plot", figure={}, style={"width": "40%"}),
            ],
        ),
        # Dropdowns for Barchart and Piechart
        html.Div(
            style={
                "display": "flex",
                "width": "60%",
                "alignItems": "center",
                "color": colors["text"],
                "fontWeight": "bold",
            },
            children=[
                html.P("Select information:"),
                # Dropdown - Information
                dcc.Dropdown(
                    id="attribute-selection",
                    options=[
                        {"label": "Earthquakes by type.", "value": "type"},
                        {
                            "label": "Earthquakes by review status.",
                            "value": "status",
                        },
                    ],
                    value="type",
                    style={"width": "90%", "marginLeft": "10px", "marginRight": "10%"},
                ),
                html.P("Select chart:"),
                # Dropdown - Chart Type
                dcc.Dropdown(
                    id="chart-selection",
                    options=[
                        {"label": "Bar Chart", "value": "bar-chart"},
                        {"label": "Pie Chart", "value": "pie-chart"},
                    ],
                    value="bar-chart",
                    style={"width": "90%", "marginLeft": "10px"},
                ),
            ],
        ),
        html.Div(
            id="chart-selector",
            style={
                "backgroundColor": "#fff",
                "color": colors["text"],
            },
            children=[
                # Chart
                dcc.Graph(
                    id="info-chart",
                    figure={},
                ),
            ],
        ),
    ],
)

# Callbacks
@app.callback(
    [
        Output(component_id="state-choropleth", component_property="figure"),
        Output(component_id="state-scatter-plot", component_property="figure"),
    ],
    Input(component_id="month-selection", component_property="value"),
)
def update_maps(selected_month):
    # print(selected_month)

    heatmap_dff = heatmap_df.copy()
    heatmap_dff = heatmap_dff[heatmap_dff["month"] == selected_month]

    heatmap = px.choropleth(
        data_frame=heatmap_dff,
        locationmode="USA-states",
        locations="state_code",
        scope="usa",
        color="earthquakes",
        color_continuous_scale="sunset",
        hover_data=["state_code", "earthquakes"],
        labels={"earthquakes": "Number of earthquakes"},
        title=f"Number of earthquakes in the USA in {selected_month} by States.",
    )

    scatter_plot_dff = scatter_plot_df.copy()
    scatter_plot_dff = scatter_plot_dff[scatter_plot_dff["month"] == selected_month]
    top_10_quakes = scatter_plot_dff.sort_values(
        by=["mag"],
        ascending=False,
    ).head(10)
    top_10_quakes["label"] = (
        top_10_quakes["id"]
        + ", "
        + top_10_quakes["mag"].astype(str)
        + ", "
        + top_10_quakes["place"]
        + ", "
        + top_10_quakes["date"].astype(str)
    )

    scatter_plot = go.Figure(
        data=go.Scattergeo(
            locationmode="USA-states",
            lat=top_10_quakes["latitude"],
            lon=top_10_quakes["longitude"],
            text=top_10_quakes["label"],
            marker=dict(
                opacity=0.6,
                color=top_10_quakes["mag"],
                colorbar_title="Magnitude",
            ),
        )
    )
    scatter_plot.update_layout(
        title="Top 10 earthquakes in the USA by magnitude.",
        geo=dict(
            scope="usa",
        ),
    )

    return heatmap, scatter_plot


@app.callback(
    Output(component_id="info-chart", component_property="figure"),
    [
        Input(component_id="attribute-selection", component_property="value"),
        Input(component_id="chart-selection", component_property="value"),
    ],
)
def update_chart(attribute, chart_type):
    # print(attribute, chart_type)

    attribute_name = attribute.title()
    chart_title = (
        f'Earthqukes by {"review" if attribute == "status" else ""} {attribute}.'
    )
    chart_df = (
        earthquakes[attribute]
        .value_counts()
        .to_frame()
        .reset_index()
        .rename(columns={"index": attribute_name, attribute: "Count"})
    )
    if chart_type == "bar-chart":
        chart = px.bar(
            chart_df,
            x=attribute_name,
            y="Count",
            title=chart_title,
        )
    else:
        chart = px.pie(
            chart_df,
            values="Count",
            names=attribute_name,
            hole=0.3,
            title=chart_title,
        )
    return chart


if __name__ == "__main__":
    app.run_server(debug=True)
