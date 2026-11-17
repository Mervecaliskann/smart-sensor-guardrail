"""
[TR] Sensör verisini denetleyen kontroller. Her kontrol satır bazlı bir boolean maske döndürür.
[EN] Sensor data checks. Each check returns a row-level boolean mask.
"""
import numpy as np
import pandas as pd

CHECKS = ("range", "robust_z", "frozen")


class SensorValidator:
    def __init__(self, thresholds, z_threshold=3.5, frozen_window=10, frozen_tolerance=0.0):
        # [TR] Sensör başına fiziksel alt/üst limitler  [EN] Physical min/max per sensor
        self.thresholds = thresholds
        self.z_threshold = z_threshold
        self.frozen_window = frozen_window
        self.frozen_tolerance = frozen_tolerance

    def range_mask(self, series, limits):
        """[TR] Fiziksel sınır dışı satırlar.  [EN] Rows outside physical limits."""
        return (series < limits["min"]) | (series > limits["max"])

    @staticmethod
    def robust_z_scores(series):
        """[TR] Medyan ve MAD ile robust Z-skoru: 0.6745 * (x - medyan) / MAD.
           [EN] Robust (modified) Z-score using median and MAD.
           MAD = 0 ise ortalama mutlak sapmaya düşer / falls back to mean absolute deviation."""
        median = series.median()
        abs_dev = (series - median).abs()
        mad = abs_dev.median()
        if mad > 0:
            return 0.6745 * (series - median) / mad
        mean_ad = abs_dev.mean()
        if mean_ad > 0:
            return (series - median) / (1.253314 * mean_ad)
        return pd.Series(0.0, index=series.index)

    def robust_z_mask(self, series):
        return self.robust_z_scores(series).abs() > self.z_threshold

    def frozen_mask(self, series):
        """[TR] En az `frozen_window` ardışık ölçüm boyunca değişmeyen değerler.
           [EN] Values unchanged for at least `frozen_window` consecutive readings."""
        unchanged = series.diff().abs() <= self.frozen_tolerance  # first row / NaN -> False
        run_id = (~unchanged).cumsum()
        run_length = series.groupby(run_id).transform("size")
        return run_length >= self.frozen_window

    def validate(self, df):
        """[TR] Her kontrol için bir sütun; herhangi bir sensör hatalıysa satır True.
           [EN] One column per check; a row is True if any sensor fails that check."""
        numeric = df.select_dtypes(include=np.number).columns
        flags = pd.DataFrame(False, index=df.index, columns=list(CHECKS))
        for sensor, limits in self.thresholds.items():
            flags["range"] |= self.range_mask(df[sensor], limits)
        for sensor in numeric:
            flags["robust_z"] |= self.robust_z_mask(df[sensor])
            flags["frozen"] |= self.frozen_mask(df[sensor])
        return flags
