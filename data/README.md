# Data

The satellite images and surface observations come from the Korea
Meteorological Administration (KMA) and the WMO Global Telecommunication
System (GTS). They are not redistributed in this repository. Place the files
below in this directory, or point `RAP_DATA_DIR` to the directory that holds them.

| File | Content |
|---|---|
| `coms_25744_224.hdf5` | Dataset `images`: uint8, shape (25744, 3, 224, 224), IR, SWIR and WV bands at 3-hour intervals. Dataset `datetimes`: bytes `YYYYMMDDhhmm`, shape (25744,), chronological (2011-02-24 00:00 to 2020-03-31 21:00). |
| `coms_with_weather_dataset.pkl` | Python list with one record per timestamp. `record[0]` is the timestamp string matching `datetimes`, `record[2]` is the 112-dim surface-observation vector. |
| `weather_scaler.pkl` | Optional. Fitted `StandardScaler` for the observation vectors. Created on first use if absent. |
| `weather_vectors_raw.npz` | Optional. Raw observation vectors in image order for `analysis/statistical_audit.py`. Create with `python analysis/export_weather_vectors.py`. |
| `precomputed_embeddings/baseline_embeddings.pkl` | Optional. Precomputed embeddings of earlier baselines. Skipped when absent. |

## Observation vector layout

Seven consecutive 16-station blocks, in storage order: relative humidity,
16-direction wind code, wind speed, station-level pressure, sea-level pressure,
temperature, dew-point temperature. Stations: Chuncheon, Gangneung, Seoul,
Incheon, Ulleungdo, Cheongju, Daejeon, Pohang, Daegu, Jeonju, Ulsan, Gwangju,
Busan, Mokpo, Yeosu, Jeju.

Residual sentinel codes are imputed with the column median before
standardization (`rap_weather.models.clean_weather_vectors`): a value is a
sentinel if `|x - median| / (1.4826 * MAD) > 20` or `x <= -99`. On this
dataset that is 488 values in 441 of the 25,744 samples.
