import pandas as pd
import geocoder


def get_state(df):
    g = geocoder.osm([df["latitude"], df["longitude"]], method="reverse")
    if g.ok:
        geojson = g.geojson
        address = geojson["features"][0]["properties"]["raw"]["address"]
        return address.get("ISO3166-2-lvl4")
    return ""


def enrich_data(filename):
    month_df = pd.read_csv(
        filename,
        parse_dates=["time"],
        usecols=[
            "id",
            "time",
            "latitude",
            "longitude",
            "mag",
            "magType",
            "place",
            "type",
            "status",
            "locationSource",
        ],
        index_col="id",
    )

    month_df["state"] = month_df[["latitude", "longitude"]].apply(get_state, axis=1)

    updated_file = f"{filename.split('.')[0]}-enriched.csv"
    month_df.to_csv(updated_file)
