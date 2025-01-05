import numpy as np
import pandas as pd
from datetime import datetime
from django.conf import settings
import os
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score


def interpolate_missing_data(self, df_orig):
    """Interpolate missing data in the dataset"""
    df_na = df_orig.copy()
    df_na = pd.DataFrame(df_na)

    df_na.columns = ['year', 'month', 'day', 3, 4] + df_na.columns[5:].tolist()

    # Handle missing values
    df_na[[3, 4]] = df_na[[3, 4]].applymap(lambda x: np.nan if isinstance(x, (int, float)) and x < -100 else x)
    df_na['date'] = pd.to_datetime(df_na[['year', 'month', 'day']])

    # Set 'date' as index and reindex to fill missing dates
    df_na.set_index('date', inplace=True)
    df = df_na.copy()
    date_range = pd.date_range(start=df.index.min(), end=df.index.max())
    df = df.reindex(date_range)

    # Keep the DatetimeIndex
    df['year'] = df.index.year
    df['month'] = df.index.month
    df['day'] = df.index.day
    temp = df[3]

    # Apply interpolation for column 3
    df[3] = DataInterpolator.interpolate(df_na[3], self.n_data_interpolate)
    df_na = df.copy()
    df_na[3] = temp

    return df, df_na


def process_yearly_data(self, df_orig):
    """Process yearly data including interpolation"""
    df, df_na = self.interpolate_missing_data(df_orig)

    # Find missing data locations
    missing_data_locations = df_na[df_na.isnull().any(axis=1)]
    missing_in_column_3 = df_na[df_na[3].isnull()]
    missing_in_column_3 = missing_in_column_3[['year', 'month', 'day']]
    numbermissingdata = missing_data_locations.shape[0]
    numbermissingcol3 = missing_in_column_3.shape[0]

    # Add interpolation flag
    df_na['Interpolated'] = 0
    df_na.loc[missing_in_column_3.index, 'Interpolated'] = 1
    df['Interpolated'] = df_na['Interpolated']

    # Reset index and convert to numpy
    df_na = df_na.reset_index().to_numpy()
    df = df.reset_index().to_numpy()
    df_na = df_na[:, 1:]
    df = df[:, 1:]
    missing_in_column_3 = missing_in_column_3.to_numpy()

    return df, numbermissingcol3, missing_in_column_3


class YearlyDataProcessor:
    def __init__(self, file_path, n_consecutive=7):
        self.df_orig = pd.DataFrame(file_path)
        self.n_consecutive = n_consecutive

    def interpolate_missing_data(self):
        df_na = self.df_orig.copy()
        df_na = pd.DataFrame(df_na)

        df_na.columns = ['year', 'month', 'day', 3, 4] + df_na.columns[5:].tolist()

        # Handle missing values for both columns 3 and 4
        df_na[[3, 4]] = df_na[[3, 4]].applymap(lambda x: np.nan if isinstance(x, (int, float)) and x < -100 else x)
        df_na['date'] = pd.to_datetime(df_na[['year', 'month', 'day']])

        # Set 'date' as index and reindex to fill missing dates
        df_na.set_index('date', inplace=True)
        df = df_na.copy()
        date_range = pd.date_range(start=df.index.min(), end=df.index.max())
        df = df.reindex(date_range)

        # Keep the DatetimeIndex
        df['year'] = df.index.year
        df['month'] = df.index.month
        df['day'] = df.index.day

        # Store original values before interpolation
        temp3 = df[3].copy()
        temp4 = df[4].copy()

        # Apply interpolation for both columns
        df[3] = self.DataInterpolator.interpolate(df_na[3], self.n_consecutive)
        df[4] = self.DataInterpolator.interpolate(df_na[4], self.n_consecutive)

        df_na = df.copy()
        df_na[3] = temp3
        df_na[4] = temp4

        return df, df_na

    class DataInterpolator:
        @staticmethod
        def interpolate(data: pd.Series, n_consecutive: int) -> pd.Series:
            interpolated_data = data.copy()
            indices = data.index
            for i in range(len(indices) - 1):
                prev_index = indices[i]
                next_index = indices[i + 1]
                gap_size = (next_index - prev_index).days - 1
                if gap_size == 0:
                    continue
                elif gap_size > n_consecutive:
                    climatology_values = YearlyDataProcessor.ClimatologyInterpolator.interpolate(data, prev_index,
                                                                                                 next_index)
                    interpolated_data = pd.concat([interpolated_data, climatology_values], axis=0)
                else:
                    linear_values = YearlyDataProcessor.LinearInterpolator.interpolate(data, prev_index, next_index)
                    interpolated_data = pd.concat([interpolated_data, linear_values], axis=0)
            return interpolated_data.sort_index()

    class LinearInterpolator:
        @staticmethod
        def interpolate(data: pd.Series, start: pd.Timestamp, end: pd.Timestamp):
            start_value = data.loc[start]
            end_value = data.loc[end]
            date_range = pd.date_range(start=start + pd.Timedelta(days=1), end=end - pd.Timedelta(days=1))
            interpolated_values = np.linspace(start_value, end_value, len(date_range) + 2)[1:-1]
            return pd.Series(data=interpolated_values, index=date_range)

    class ClimatologyInterpolator:
        @staticmethod
        def interpolate(data: pd.Series, start: pd.Timestamp, end: pd.Timestamp):
            date_range = pd.date_range(start=start + pd.Timedelta(days=1), end=end - pd.Timedelta(days=1))
            values = []
            for day in date_range:
                climatology_mean = data[(data.index.month == day.month) & (data.index.day == day.day)].mean()
                values.append(climatology_mean if pd.notna(climatology_mean) else 0.00)
            return pd.Series(data=values, index=date_range)

    def mean_year(self):
        df, df_na = self.interpolate_missing_data()

        missing_data_locations = df_na[df_na.isnull().any(axis=1)]
        missing_in_columns = df_na[df_na[[3, 4]].isnull().any(axis=1)]
        missing_in_columns = missing_in_columns[['year', 'month', 'day']]
        numbermissingdata = missing_data_locations.shape[0]
        numbermissingcols = missing_in_columns.shape[0]

        df_na['Interpolated'] = 0
        df_na.loc[missing_in_columns.index, 'Interpolated'] = 1
        df['Interpolated'] = df_na['Interpolated']

        df_na = df_na.reset_index().to_numpy()
        df = df.reset_index().to_numpy()
        df_na = df_na[:, 1:]  # Remove the index column
        df = df[:, 1:]  # Remove the index column
        missing_in_columns = missing_in_columns.to_numpy()

        return df, numbermissingcols, missing_in_columns


class ML_Model:
    def __init__(self, user_id=0, group_id=0, interpolate=True, n_data_interpolate=7,
                 validation_required=False, percent=10,
                 model="air2water",
                 missing_data_threshold=30, interpolate_use_rmse=True,
                 air2waterusercalibrationpath=None,
                 air2streamusercalibrationpath=None,
                 air2wateruservalidationpath=None,
                 air2streamuservalidationpath=None,
                 results_file_name="results.db",
                 email_send=0, sim_id=0, email_list=None):

        self.user_id = user_id
        self.group_id = group_id
        self.model = model
        self.interpolate = interpolate
        self.n_data_interpolate = n_data_interpolate
        self.missing_data_threshold = missing_data_threshold
        self.interpolate_use_rmse = interpolate_use_rmse
        self.validation_required = validation_required
        self.percent = percent
        self.air2waterusercalibrationpath = air2waterusercalibrationpath
        self.air2streamusercalibrationpath = air2streamusercalibrationpath
        self.air2wateruservalidationpath = air2wateruservalidationpath
        self.air2streamuservalidationpath = air2streamuservalidationpath
        self.results_file_name = results_file_name
        self.sim_id = sim_id
        self.email_send = email_send
        self.email_list = email_list if email_list else []
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=2)
        self.owd = os.getcwd()

    def datetimecalc(self, dfinput):
        """Calculate datetime and normalized day of year from input data"""
        Y = dfinput[:, 0]
        M = dfinput[:, 1]
        D = dfinput[:, 2]
        date = [datetime(int(y), int(m), int(d)) for y, m, d in zip(Y, M, D)]
        return date, np.asarray([d.timetuple().tm_yday / 366 for d in date])

    def prepare_ml_data(self, data, is_air2stream=False):
        """
        Prepare data for machine learning by creating feature set and target variable
        """
        # Remove rows where target variable (column 4) is -999
        valid_indices = data[:, 4] != -999
        filtered_data = data[valid_indices]

        # Calculate dates and tt for the filtered data
        dates, tt = self.datetimecalc(filtered_data)

        if is_air2stream:
            # For air2stream: Features = [tt, column4, column6], Target = column5
            X = np.column_stack((tt, filtered_data[:, 3], filtered_data[:, 5]))  # Features
            y = filtered_data[:, 4]  # Target
        else:
            # For air2water: Features = [tt, column4], Target = column5
            X = np.column_stack((tt, filtered_data[:, 3]))  # Features
            y = filtered_data[:, 4]  # Target

        return {
            'X': X,
            'y': y,
            'dates': dates,
            'original_data': filtered_data
        }

    def save_results_csv(self, predictions, data, mode, owd):
        """Save results to CSV in the required format"""
        dates_str = [d.strftime('%Y-%m-%d') for d in data['dates']]
        df = pd.DataFrame({
            'Date': dates_str,
            'Air_temperature_data': data['original_data'][:, 3],
            'Best_simulation': predictions,
            'Evaluation_(Water_temperature_data)': data['y']
        })

        ######REMOVE OWD AND OSMAKEDIRS PART

        results_dir = f"{owd}/results/{self.user_id}_{self.group_id}/"
        os.makedirs(results_dir, exist_ok=True)
        filepath = f"{results_dir}results_{self.sim_id}_{mode}.csv"
        df.to_csv(filepath, index=False)
        return df

    def plot_results(self, df, mode, owd, train, mean=None, std=None):
        """Create time series plot in the required format with R² score"""
        fig, ax = plt.subplots(figsize=(12, 6))

        df = df.sort_values('Date')

        dates = pd.to_datetime(df['Date'])

        ax.plot(
            dates,
            df['Best_simulation'],
            color="black",
            linestyle="solid",
            label="Simulated water temperature"
        )
        ax.plot(
            dates,
            df['Evaluation_(Water_temperature_data)'],
            "r.",
            markersize=3,
            label="Observed water temperature data"
        )
        ax.plot(
            dates,
            df['Air_temperature_data'],
            color="blue",
            linewidth=0.2,
            label="Observed air temperature data"
        )

        if train == "calib":
            # Calculate R² score
            r2 = f"{mean:.3f} ± {std:.3f}"

            # Add R² score to plot
            plt.text(0.02, 0.98, f'R² = {r2}',
                     transform=ax.transAxes,
                     bbox=dict(facecolor='white', alpha=0.8),
                     verticalalignment='top')
        elif train== "valid":
            # Calculate R² score
            r2 = r2_score(df['Evaluation_(Water_temperature_data)'], df['Best_simulation'])

            # Add R² score to plot
            plt.text(0.02, 0.98, f'R² = {r2:.3f}',
                     transform=ax.transAxes,
                     bbox=dict(facecolor='white', alpha=0.8),
                     verticalalignment='top')

        plt.xlabel("Year")
        plt.ylabel("Temperature")
        plt.legend(loc="upper right")

        filepath = f"{owd}/results/{self.user_id}_{self.group_id}/{mode}_best_modelrun_{self.sim_id}.png"
        fig.savefig(filepath, dpi=100)
        plt.close()

        return filepath

    def train_svr(self, X_train, y_train, X_val, y_val):
        """Train SVR model and make predictions"""
        # Initialize SVR with RBF kernel
        regressor = SVR(kernel='rbf')

        # Train the model
        regressor.fit(X_train, y_train)

        # Make predictions
        train_pred = regressor.predict(X_train)
        val_pred = regressor.predict(X_val)

        # Perform k-fold cross validation
        cv_scores = cross_val_score(regressor, X_train, y_train, cv=10)

        return {
            'train_pred': train_pred,
            'val_pred': val_pred,
            'cv_scores': cv_scores,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std()
        }

    def train_mlr(self, X_train, y_train, X_val, y_val):
        """Train Multiple Linear Regression model and make predictions"""
        # Initialize Linear Regression
        regressor = LinearRegression()

        # Train the model
        regressor.fit(X_train, y_train)

        # Make predictions
        train_pred = regressor.predict(X_train)
        val_pred = regressor.predict(X_val)

        # Perform k-fold cross validation
        cv_scores = cross_val_score(regressor, X_train, y_train, cv=10)

        return {
            'train_pred': train_pred,
            'val_pred': val_pred,
            'cv_scores': cv_scores,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std()
        }

    def handle_nan_values(self, df):
        """Replace NaN values with climatological means."""
        import pandas as pd
        import numpy as np

        # Create a copy of the dataframe
        df = pd.DataFrame(df)
        if df.shape[1] < 5:
            df.columns = ['year', 'month', 'day', 3]
        else:
            df.columns = ['year', 'month', 'day', 3, 4] + df.columns[5:].tolist()

        # Create month-day grouping for climatological means
        df['month_day'] = df['month'].astype(str).str.zfill(2) + '-' + df['day'].astype(str).str.zfill(2)

        # For each column that needs processing (columns 3 and 4)
        for col in [3, 4]:
            if col in df.columns:
                # Convert -999 to NaN
                df[col] = df[col].replace(-999.000, np.nan)

                # Calculate climatological means for each month-day combination
                climatological_means = df.groupby('month_day')[col].mean()

                # Replace NaN values with climatological means
                for idx in df[df[col].isna()].index:
                    month_day = df.loc[idx, 'month_day']
                    df.loc[idx, col] = climatological_means[month_day]

        # Drop the helper column
        df = df.drop('month_day', axis=1)

        return df.values

    def run(self):
        """Main execution method"""
        # Set data path based on model type
        if self.model == "air2water":
            usercalibrationdatapath = self.air2waterusercalibrationpath
            uservalidationdatapath = self.air2wateruservalidationpath
        elif self.model == "air2stream":  # air2stream
            usercalibrationdatapath = self.air2streamusercalibrationpath
            uservalidationdatapath = self.air2streamuservalidationpath

        # Load calibration data
        df_calibration = np.loadtxt(usercalibrationdatapath)

        # Handle NaN values before any other processing
        df_calibration = self.handle_nan_values(df_calibration)

        # Apply yearly data processing and interpolation if needed
        if self.interpolate:
            print("Interpolating...")
            processor = YearlyDataProcessor(df_calibration, self.n_data_interpolate)
            df_calibration, num_missing_col3, missing_col3 = processor.mean_year()

        # Process validation data based on validation_required option
        if not self.validation_required:
            # Use separate calibration and validation files
            calibration_data = df_calibration
            try:
                validation_data = np.loadtxt(uservalidationdatapath)
                validation_data = self.handle_nan_values(validation_data)
            except:
                print("Validation file not found for validation_required=False")
                return None
        elif self.validation_required == "Uniform Percentage":
            # Use sklearn's train_test_split for uniform percentage split
            calibration_data, validation_data = train_test_split(df_calibration,
                                                                 test_size=self.percent / 100,
                                                                 shuffle=True,
                                                                 random_state=42)
        elif self.validation_required == "Random Percentage":
            # Split calibration data randomly
            mask = np.random.rand(len(df_calibration)) < (self.percent / 100)
            calibration_data = df_calibration[~mask]
            validation_data = df_calibration[mask]
        elif self.validation_required == "Uniform Number":
            # Split calibration data uniformly by taking every nth sample
            indices = np.arange(len(df_calibration))
            validation_indices = indices[self.percent - 1::self.percent]
            calibration_indices = np.delete(indices, validation_indices)
            calibration_data = df_calibration[calibration_indices]
            validation_data = df_calibration[validation_indices]

        # Prepare data for machine learning
        is_air2stream = (self.model == "air2stream")

        # Create feature sets and target variables with dates
        train_data = self.prepare_ml_data(calibration_data, is_air2stream)
        val_data = self.prepare_ml_data(validation_data, is_air2stream)

        # Scale the features
        X_train_scaled = self.scaler.fit_transform(train_data['X'])
        X_val_scaled = self.scaler.transform(val_data['X'])

        # Apply PCA transformation
        X_train_pca = self.pca.fit_transform(X_train_scaled)
        X_val_pca = self.pca.transform(X_val_scaled)

        # Calculate explained variance ratio
        explained_variance_ratio = self.pca.explained_variance_ratio_
        print(f"Explained variance ratio: {explained_variance_ratio}")

        # Train SVR and MLR on original scaled data
        svr_results_orig = self.train_svr(
            X_train_scaled, train_data['y'],
            X_val_scaled, val_data['y']
        )
        mlr_results_orig = self.train_mlr(
            X_train_scaled, train_data['y'],
            X_val_scaled, val_data['y']
        )

        # Train SVR and MLR on PCA transformed data
        svr_results_pca = self.train_svr(
            X_train_pca, train_data['y'],
            X_val_pca, val_data['y']
        )
        mlr_results_pca = self.train_mlr(
            X_train_pca, train_data['y'],
            X_val_pca, val_data['y']
        )

        # Save results for original data
        original_train_df = self.save_results_csv(
            svr_results_orig['train_pred'],
            train_data,
            'calibration_svr',
            self.owd
        )
        original_val_df = self.save_results_csv(
            svr_results_orig['val_pred'],
            val_data,
            'validation_svr',
            self.owd
        )

        mlr_train_df = self.save_results_csv(
            mlr_results_orig['train_pred'],
            train_data,
            'calibration_mlr',
            self.owd
        )
        mlr_val_df = self.save_results_csv(
            mlr_results_orig['val_pred'],
            val_data,
            'validation_mlr',
            self.owd
        )

        # Save results for PCA data
        pca_train_df = self.save_results_csv(
            svr_results_pca['train_pred'],
            train_data,
            'calibration_svr_pca',
            self.owd
        )
        pca_val_df = self.save_results_csv(
            svr_results_pca['val_pred'],
            val_data,
            'validation_svr_pca',
            self.owd
        )

        mlr_pca_train_df = self.save_results_csv(
            mlr_results_pca['train_pred'],
            train_data,
            'calibration_mlr_pca',
            self.owd
        )
        mlr_pca_val_df = self.save_results_csv(
            mlr_results_pca['val_pred'],
            val_data,
            'validation_mlr_pca',
            self.owd
        )

        # Create all plots
        self.plot_results(original_train_df, 'calibration_svr', self.owd, "calib", svr_results_orig['cv_mean'], svr_results_orig['cv_std'])
        self.plot_results(original_val_df, 'validation_svr', self.owd, "valid")
        self.plot_results(mlr_train_df, 'calibration_mlr', self.owd, "calib", mlr_results_orig['cv_mean'], mlr_results_orig['cv_std'])
        self.plot_results(mlr_val_df, 'validation_mlr', self.owd, "valid")
        self.plot_results(pca_train_df, 'calibration_svr_pca', self.owd, "calib", svr_results_pca['cv_mean'], svr_results_pca['cv_std'])
        self.plot_results(pca_val_df, 'validation_svr_pca', self.owd, "valid")
        self.plot_results(mlr_pca_train_df, 'calibration_mlr_pca', self.owd, "calib", mlr_results_pca['cv_mean'], mlr_results_pca['cv_std'])
        self.plot_results(mlr_pca_val_df, 'validation_mlr_pca', self.owd, "valid")

        return {
            'original': {
                'svr': {
                    'train': original_train_df,
                    'val': original_val_df,
                    'cv_scores': svr_results_orig['cv_scores'],
                    'cv_mean': svr_results_orig['cv_mean'],
                    'cv_std': svr_results_orig['cv_std']
                },
                'mlr': {
                    'train': mlr_train_df,
                    'val': mlr_val_df,
                    'cv_scores': mlr_results_orig['cv_scores'],
                    'cv_mean': mlr_results_orig['cv_mean'],
                    'cv_std': mlr_results_orig['cv_std']
                }
            },
            'pca': {
                'svr': {
                    'train': pca_train_df,
                    'val': pca_val_df,
                    'cv_scores': svr_results_pca['cv_scores'],
                    'cv_mean': svr_results_pca['cv_mean'],
                    'cv_std': svr_results_pca['cv_std']
                },
                'mlr': {
                    'train': mlr_pca_train_df,
                    'val': mlr_pca_val_df,
                    'cv_scores': mlr_results_pca['cv_scores'],
                    'cv_mean': mlr_results_pca['cv_mean'],
                    'cv_std': mlr_results_pca['cv_std']
                }
            },
            'explained_variance_ratio': explained_variance_ratio
        }

if __name__ == "__main__":
    import time
    import os

    start_time = time.time()

    # Test paths - replace with your actual file paths
    calibration_file = '/home/riddick/Downloads/SIO_2011_cc.txt'
    validation_file = '/home/riddick/Downloads/SIO_2011_cv.txt'

    # Initialize ML Model
    ml_model = ML_Model(
        user_id=1,
        group_id=1,
        model="air2stream",
        interpolate=True,
        n_data_interpolate=7,
        validation_required="Uniform Percentage",
        percent=20,
        air2streamusercalibrationpath=calibration_file,
        air2streamuservalidationpath=validation_file
    )

    # Run the models and get results
    results = ml_model.run()

    if results is not None:
        print("\nResults Summary:")
        print("\nOriginal Data Results:")
        print("SVR Model:")
        print(
            f"Training R² Score: {results['original']['svr']['cv_mean']:.3f} ± {results['original']['svr']['cv_std']:.3f}")
        print(
            f"Validation R² Score: {r2_score(results['original']['svr']['val']['Evaluation_(Water_temperature_data)'], results['original']['svr']['val']['Best_simulation']):.3f}")

        print("\nMLR Model:")
        print(
            f"Training R² Score: {results['original']['mlr']['cv_mean']:.3f} ± {results['original']['mlr']['cv_std']:.3f}")
        print(
            f"Validation R² Score: {r2_score(results['original']['mlr']['val']['Evaluation_(Water_temperature_data)'], results['original']['mlr']['val']['Best_simulation']):.3f}")

        print("\nPCA Results:")
        print(f"Explained Variance Ratio: {results['explained_variance_ratio']}")
        print("\nSVR Model with PCA:")
        print(f"Training R² Score: {results['pca']['svr']['cv_mean']:.3f} ± {results['pca']['svr']['cv_std']:.3f}")
        print(
            f"Validation R² Score: {r2_score(results['pca']['svr']['val']['Evaluation_(Water_temperature_data)'], results['pca']['svr']['val']['Best_simulation']):.3f}")

        print("\nMLR Model with PCA:")
        print(f"Training R² Score: {results['pca']['mlr']['cv_mean']:.3f} ± {results['pca']['mlr']['cv_std']:.3f}")
        print(
            f"Validation R² Score: {r2_score(results['pca']['mlr']['val']['Evaluation_(Water_temperature_data)'], results['pca']['mlr']['val']['Best_simulation']):.3f}")

    total_time = np.array([time.time() - start_time])[0]
    print(f"\nTotal run time: {total_time:.2f} seconds")