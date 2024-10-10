import pandas as pd
import ee
from datetime import datetime
import numpy as np
import multiprocessing as mp
from tqdm import tqdm
import os
import logging
import glob

ee.Initialize()

input_csv = "/Users/milindsoni/Documents/projects/rice/bamboo/2021_v3.csv"
output_csv = "bamboovector_full_with_weather_2021.csv"
temp_output_folder = "temp_results_2021"
log_file = "processing_log_2021.txt"

NUM_WORKERS = mp.cpu_count() - 1
SAVE_INTERVAL = 1000

start_date = ee.Date("2022-05-01")
end_date = ee.Date("2022-11-30")

s2_bands = ["B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B11", "B12"]
s1_bands = ["VV", "VH"]
weather_bands = [
    "u_component_of_wind_10m_max",
    "v_component_of_wind_10m_max",
    "temperature_2m_max",
    "temperature_2m_min",
    "total_precipitation_sum",
    "potential_evaporation_sum",
    "surface_net_solar_radiation_sum",
    "volumetric_soil_water_layer_1",
]

logging.basicConfig(
    filename=log_file, level=logging.INFO, format="%(asctime)s - %(message)s"
)


def create_15day_mosaics(
    collection, start_date, end_date, bands, reducer=ee.Reducer.median()
):
    def create_mosaic(d):
        date = ee.Date(d)
        end = date.advance(15, "day")
        mosaic = collection.filterDate(date, end).reduce(reducer)
        return mosaic.set("system:time_start", date.millis())

    dates = ee.List.sequence(start_date.millis(), end_date.millis(), 15 * 86400 * 1000)
    mosaics = ee.ImageCollection(dates.map(lambda d: create_mosaic(d)))
    return mosaics.select([f"{b}_.*" for b in bands], bands)


def process_row(row):
    lat, lon = row["lattitude"], row["longitude"]

    if pd.isna(lat) or pd.isna(lon):
        logging.warning(f"Skipping row with invalid coordinates: Lat {lat}, Lon {lon}")
        return pd.Series()

    point = ee.Geometry.Point([lon, lat])

    # Sentinel-2 collection
    s2_collection = (
        ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filterBounds(point)
        .filterDate(start_date, end_date)
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 20))
    )

    # Sentinel-1 collection
    s1_collection = (
        ee.ImageCollection("COPERNICUS/S1_GRD")
        .filterBounds(point)
        .filterDate(start_date, end_date)
        .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VV"))
        .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VH"))
        .filter(ee.Filter.eq("instrumentMode", "IW"))
    )

    # Weather collection
    weather_collection = (
        ee.ImageCollection("ECMWF/ERA5_LAND/DAILY_AGGR")
        .filterBounds(point)
        .filterDate(start_date, end_date)
        .select(weather_bands)
    )

    # Soil type
    soil_image = ee.Image("users/2019ht12445/gldas_soil")

    s2_mosaics = create_15day_mosaics(s2_collection, start_date, end_date, s2_bands)
    s1_mosaics = create_15day_mosaics(s1_collection, start_date, end_date, s1_bands)
    weather_mosaics = create_15day_mosaics(
        weather_collection, start_date, end_date, weather_bands, ee.Reducer.mean()
    )
    results = {
        "farmer_code": row.get("Farmer_Code", ""),
        "lattitude": lat,
        "longitude": lon,
        "Declared_Area(acre)": row.get("Declared_Area(acre)", ""),
        "Sowing_Area(acre)": row.get("Sowing_Area(acre)", ""),
        "Actual_Harvest_Completed": row.get("Actual_Harvest_Completed", ""),
        "Lot_No": row.get("Lot_No", ""),
        "Yield": row.get("Yield", ""),
    }

    all_dates = set()
    temp_results = {}

    for prefix, mosaics, bands in [
        ("Sentinel_", s2_mosaics, s2_bands),
        ("SAR_", s1_mosaics, s1_bands),
        ("ECMWF_", weather_mosaics, weather_bands),
    ]:
        try:
            data = mosaics.getRegion(point, 10000).getInfo()

            headers = data[0]
            date_index = headers.index("time")
            band_indices = [headers.index(band) for band in bands]

            for row in data[1:]:
                date_str = datetime.utcfromtimestamp(row[date_index] / 1000).strftime(
                    "%Y-%m-%d"
                )
                all_dates.add(date_str)
                for i, band in enumerate(bands):
                    value = row[band_indices[i]]
                    if prefix == "ECMWF_":
                        if "temperature" in band:
                            value -= 273.15  # Convert from Kelvin to Celsius
                        elif band == "total_precipitation_sum":
                            value *= 1000  # Convert from meters to millimeters

                    # Handle null or empty values
                    if value is None or pd.isna(value):
                        value = -9999  # Use a sentinel value for missing data
                    else:
                        value = f"{value:.4f}"

                    band_key = f"{prefix}{band}"
                    if band_key not in temp_results:
                        temp_results[band_key] = {}
                    temp_results[band_key][date_str] = value

        except ee.ee_exception.EEException as e:
            logging.error(
                f"Error retrieving {prefix} data for point ({lat}, {lon}): {e}"
            )

    # Extract soil type
    try:
        soil_value = (
            soil_image.reduceRegion(ee.Reducer.first(), point, 30).get("b1").getInfo()
        )
        results["soil_type"] = soil_value if soil_value is not None else -9999
    except ee.ee_exception.EEException as e:
        logging.error(f"Error retrieving soil type for point ({lat}, {lon}): {e}")
        results["soil_type"] = -9999

    # Reorganize results
    all_dates = sorted(list(all_dates))
    for band in temp_results:
        for date in all_dates:
            results[f"{band}_{date}"] = temp_results[band].get(date, -9999)

    return pd.Series(results)


def find_last_processed_file():
    temp_files = glob.glob(os.path.join(temp_output_folder, "temp_results_*.csv"))
    if not temp_files:
        return None
    return max(temp_files, key=lambda f: int(f.split("_")[-1].split(".")[0]))


def combine_temp_files():
    temp_files = sorted(
        glob.glob(os.path.join(temp_output_folder, "temp_results_*.csv")),
        key=lambda f: int(f.split("_")[-1].split(".")[0]),
    )

    if not temp_files:
        logging.warning("No temporary files found to combine.")
        return None

    combined_df = pd.concat([pd.read_csv(f) for f in temp_files], ignore_index=True)

    logging.info(f"Combined {len(temp_files)} temporary files.")
    logging.info(f"Total rows in combined dataframe: {len(combined_df)}")

    return combined_df


def continue_processing():
    df = pd.read_csv(input_csv)
    last_file = find_last_processed_file()

    if last_file:
        last_processed = int(last_file.split("_")[-1].split(".")[0])
        logging.info(f"Continuing from row {last_processed}")
        start_index = last_processed
    else:
        logging.info("Starting from the beginning")
        start_index = 0

    if not os.path.exists(temp_output_folder):
        os.makedirs(temp_output_folder)

    pool = mp.Pool(processes=NUM_WORKERS)

    results = []
    with tqdm(total=len(df) - start_index, desc="Processing rows") as pbar:
        for i, result in enumerate(
            pool.imap(process_row, df.iloc[start_index:].to_dict("records")),
            start=start_index,
        ):
            if not result.empty:
                results.append(result)
            pbar.update()

            if (i + 1) % SAVE_INTERVAL == 0 or i == len(df) - 1:
                temp_df = pd.DataFrame(results)
                temp_output_file = os.path.join(
                    temp_output_folder, f"temp_results_{i+1}.csv"
                )
                temp_df.to_csv(temp_output_file, index=False)
                logging.info(f"Intermediate results saved to {temp_output_file}")

                # Clear the results list to free up memory
                results = []

    pool.close()
    pool.join()

    logging.info("Processing complete. Concatenating all temporary files...")
    final_df = combine_temp_files()

    if final_df is not None:
        # Reorder columns
        static_columns = [
            "farmer_code",
            "lattitude",
            "longitude",
            "Declared_Area(acre)",
            "Sowing_Area(acre)",
            "Actual_Harvest_Completed",
            "Lot_No",
            "Yield",
            "soil_type",
        ]
        dynamic_columns = [col for col in final_df.columns if col not in static_columns]

        # Sort dynamic columns by band and then by date
        dynamic_columns.sort(key=lambda x: (x.split("_")[0], x.split("_")[-1]))

        final_df = final_df[static_columns + dynamic_columns]

        final_df.to_csv(output_csv, index=False)
        logging.info(
            f"Full data with weather information and soil type saved to {output_csv}"
        )

        # Print summary statistics
        logging.info("\nSummary:")
        logging.info(f"Total number of rows in input: {len(df)}")
        logging.info(f"Total number of rows processed: {len(final_df)}")
        logging.info(f"Number of rows skipped: {len(df) - len(final_df)}")
        logging.info(f"Total number of features: {len(final_df.columns)}")

        # Print a few sample values from the first row
        logging.info("\nSample values from the first row:")
        for col in final_df.columns[:10]:  # Print first 10 columns as an example
            logging.info(f"{col}: {final_df[col].iloc[0]}")
    else:
        logging.error(
            "Failed to combine temporary files. Please check the temporary files in the output folder."
        )


if __name__ == "__main__":
    continue_processing()
    print(
        f"Processing complete. Check {log_file} for details and {output_csv} for results."
    )
