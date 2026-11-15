"""Export the raw 112-dim weather-observation vectors in image order.

statistical_audit.py reads weather_vectors_raw.npz instead of the full
weather pickle so the statistics run without loading the ~10 GB file. The
vectors are written uncleaned; the audit applies the same sentinel cleaning
as rap_weather.models.clean_weather_vectors.

Output: <data dir>/weather_vectors_raw.npz with keys "datetimes" (str) and
"vectors" (float, shape [N, 112]).
"""
import pickle

import h5py
import numpy as np

from _paths import DATA_DIR, H5, WEATHER_NPZ


def main():
    with open(DATA_DIR / "coms_with_weather_dataset.pkl", "rb") as handle:
        records = pickle.load(handle)
    weather = {str(item[0]): np.asarray(item[2]) for item in records}
    with h5py.File(H5, "r") as handle:
        image_dates = handle["datetimes"][:].astype(str)
    missing = [d for d in image_dates if d not in weather]
    if missing:
        raise RuntimeError(f"{len(missing)} image timestamps have no weather vector, e.g. {missing[:3]}")
    vectors = np.stack([weather[d] for d in image_dates])
    np.savez(WEATHER_NPZ, datetimes=image_dates, vectors=vectors)
    print(f"wrote {WEATHER_NPZ}: {vectors.shape}")


if __name__ == "__main__":
    main()
