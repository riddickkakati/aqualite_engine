import pandas as pd
import numpy as np
import calendar
import os


class YearlyDataProcessor:
    def __init__(self, file_path, n_consecutive=7):
        self.df_orig = pd.DataFrame(file_path)
        self.n_consecutive = n_consecutive

    def interpolate_missing_data(self):
        df = self.df_orig.copy()
        df.columns = ['year', 'month', 'day', 3, 4] + df.columns[5:].tolist()
        df[[3, 4]] = df[[3, 4]].applymap(lambda x: np.nan if isinstance(x, (int, float)) and x < -100 else x)
        df['date'] = pd.to_datetime(df[['year', 'month', 'day']])
        date_range = pd.date_range(start=df['date'].min(), end=df['date'].max())
        df.set_index('date', inplace=True)
        df = df.reindex(date_range)
        df.reset_index(inplace=True)
        df.rename(columns={'index': 'date'}, inplace=True)

        df['year'] = df['date'].dt.year
        df['month'] = df['date'].dt.month
        df['day'] = df['date'].dt.day

        df_na = df.copy()
        df = df.iloc[:, 1:]

        for y0 in df['year'].astype(int).unique():
            shape = (1, 366) if calendar.isleap(y0) else (1, 365)
            date = np.empty(shape, dtype='datetime64[D]')
            date.fill(np.datetime64('NaT'))
            Tam = np.full(shape, np.nan)
            Twm = np.full(shape, np.nan)
            Tam = Tam.squeeze()
            Twm = Twm.squeeze()
            date = date.squeeze()
            i = 0

            for m in range(1, 13):
                n = calendar.monthrange(y0, m)[1] + 1
                for d in range(1, n):
                    index_value = df[(df.iloc[:, 0] == y0) & (df.iloc[:, 1] == m) & (df.iloc[:, 2] == d)].index.values[
                        0]
                    isnan = np.isnan(df[3].iloc[index_value])

                    # Check for n consecutive NaN values
                    if isnan:
                        consecutive_nan = True
                        for k in range(1, self.n_consecutive):
                            if index_value + k < len(df):
                                if not np.isnan(df[3].iloc[index_value + k]):
                                    consecutive_nan = False
                                    break
                            else:
                                consecutive_nan = False
                                break
                        if consecutive_nan:
                            for k in range(1, self.n_consecutive):
                                if index_value - k >= 0:
                                    if not np.isnan(df[3].iloc[index_value - k]):
                                        consecutive_nan = False
                                        break
                                else:
                                    consecutive_nan = False
                                    break
                    else:
                        consecutive_nan = False

                    if consecutive_nan:
                        if d <= np.max(np.array([31, 28 + (y0 % 4 == 0 and (y0 % 100 != 0 or y0 % 400 == 0))])):
                            i += 1
                            date[i - 1] = np.datetime64(f'{y0:04d}-{m:02d}-{d:02d}')
                            pp = (df['month'] == m) & (df['day'] == d)
                            Tam[i - 1] = np.nanmean(df[3][pp])
                            df[3].iloc[index_value] = Tam[i - 1]

            df[3] = df[3].interpolate(method='linear')

            # Handle the second column (optional)
            '''
            for m in range(1, 13):
                n = calendar.monthrange(y0, m)[1] + 1
                for d in range(1, n):
                    index_value = df[(df.iloc[:, 0] == y0) & (df.iloc[:, 1] == m) & (df.iloc[:, 2] == d)].index.values[0]
                    isnan = np.isnan(df[4].iloc[index_value])
                    if isnan:
                        consecutive_nan = True
                        for k in range(1, self.n_consecutive):
                            if index_value + k < len(df):
                                if not np.isnan(df[4].iloc[index_value + k]):
                                    consecutive_nan = False
                                    break
                            else:
                                consecutive_nan = False
                                break
                        if consecutive_nan:
                            for k in range(1, self.n_consecutive):
                                if index_value - k >= 0:
                                    if not np.isnan(df[4].iloc[index_value - k]):
                                        consecutive_nan = False
                                        break
                                else:
                                    consecutive_nan = False
                                    break
                    else:
                        consecutive_nan = False

                    if consecutive_nan:
                        i += 1
                        date[i - 1] = np.datetime64(f'{y0:04d}-{m:02d}-{d:02d}')
                        pp = (df['month'] == m) & (df['day'] == d)
                        Tam[i - 1] = np.nanmean(df[4][pp])
                        df[4].iloc[index_value] = Tam[i - 1]

            df[4] = df[4].interpolate(method='linear')
            '''
        return df, df_na

    def mean_year(self):
        df, df_na = self.interpolate_missing_data()
        df_orig = self.df_orig

        missing_data_locations = df_na[df_na.isnull().any(axis=1)]
        missing_in_column_3 = df_na[df_na[3].isnull()]
        missing_in_column_3 = missing_in_column_3.iloc[:, 1:4]
        numbermissingdata = missing_data_locations.index.size
        numbermissingcol3 = missing_in_column_3.index.size
        missing_in_column_3 = missing_in_column_3.to_numpy()
        df['Interpolated'] = 0
        df.loc[missing_data_locations.index, 'Interpolated'] = 1
        df = df.to_numpy()

        return df, numbermissingcol3, missing_in_column_3


if __name__ == "__main__":
    owd=os.getcwd()
    filepath = pd.read_csv('/home/dicam01/air2water_test/stndrck_sat_cc.txt',delimiter='\t',header=None)
    processor = YearlyDataProcessor(filepath)
    df, num_missing_col3, missing_col3 = processor.mean_year()
    print(df)
    print(num_missing_col3)
    print(missing_col3)
