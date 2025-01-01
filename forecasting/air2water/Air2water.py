import pyswarms as ps
from .calibrators.PSO_source import pso_oop
from .calibrators.PSO_A2WS import pso_oop_new
from .calibrators.spotpy_params_air2water_air2stream import spot_setup
import pycup as cp
from pycup.integrate import SpotpySetupConverter
from pycup.integrate import SpotpyDbConverter
import time
import spotpy
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from io import BytesIO
import numpy as np
import sqlite3
import pandas as pd
import os
import glob
import yaml
import fnmatch
from .IO.yaml_parser import read_yaml
import shutil
from spotpy.parameter import Uniform#,logNormal,Normal
import math
from .IO.Interpolator import YearlyDataProcessor
from sklearn.model_selection import train_test_split
from .core import air2stream,air2water
from datetime import datetime
from .calibrators.Parameters import Air2WaterParameters
from django.conf import settings

from spotpy.algorithms import sceua
from spotpy.algorithms import abc  # Artificial Bee Colony
from spotpy.algorithms import dds  # Dynamically Dimensioned Search algorithm
from spotpy.algorithms import demcz  # Differential Evolution Markov Chain
from spotpy.algorithms import dream  # DiffeRential Evolution Adaptive Metropolis
from spotpy.algorithms import fast  # Fourier Amplitude Sensitivity Test
from spotpy.algorithms import fscabc  # Fitness Scaling Artificial Bee Colony
from spotpy.algorithms import lhs  # Latin Hypercube Sampling
from spotpy.algorithms import list_sampler  # Samples from  given spotpy database
from spotpy.algorithms import mc  # Monte Carlo
from spotpy.algorithms import mcmc  # Metropolis Markov Chain Monte Carlo
from spotpy.algorithms import mle  # Maximum Likelihood Estimation
#from spotpy.algorithms import nsgaii
from spotpy.algorithms import padds  # Pareto Archived - Dynamicallly Dimensioned Search algorithm
from spotpy.algorithms import rope  # RObust Parameter Estimation
from spotpy.algorithms import sa  # Simulated annealing
from spotpy.algorithms import sceua  # Shuffled Complex Evolution

from .IO.Send_emails import send_email

def simulation_air2water_log_pyswarms(self, x):
    # Here the model is actualy startet with one paramter combination
    params = air2water(self.Tw_solution, self.Ta_data, self.tt, x[0], 10 ** x[1], 10 ** x[2], x[3], x[4],
                       x[5], x[6], x[7], self.version, self.solver, self.compiler, self.CFL)

    # if self.mode2!="forward":
    #     if x[5]>self.parameters[5].maxbound:
    #         x[5] = x[5] - np.floor(x[5])
    #     if x[5]<self.parameters[5].minbound:
    #         x[5] = np.ceil(np.abs(x[5]))-np.abs(x[5])

    data = params.solve()
    sim = []
    for val in data:
        sim.append(val)
    # The first year of simulation data is ignored (warm-up)
    if self.mode2 == "forward":
        return sim
    else:  # The first year of simulation data is ignored (warm-up)
        return sim[366:]

def simulation_air2water_no_log_pyswarms(self, x):
    # Here the model is actualy startet with one paramter combination
    params = air2water(self.Tw_solution, self.Ta_data, self.tt, x[0], x[1], x[2], x[3], x[4],
                       x[5], x[6], x[7], self.version, self.solver, self.compiler, self.CFL)

    # if self.mode2 != "forward":
    #     if x[5] > self.parameters[5].maxbound:
    #         x[5] = x[5] - np.floor(x[5])
    #     if x[5] < self.parameters[5].minbound:
    #         x[5] = np.ceil(np.abs(x[5])) - np.abs(x[5])

    data = params.solve()
    sim = []
    for val in data:
        sim.append(val)
    # The first year of simulation data is ignored (warm-up)
    if self.mode2 == "forward":
        return sim
    else:  # The first year of simulation data is ignored (warm-up)
        return sim[366:]

def simulation_air2stream_pyswarms(self, x):
    # Here the model is actualy startet with one paramter combination
    params = air2stream(self.Tw_solution, self.Ta_data, self.Q, self.tt, x[0],  # 10**
                        x[1],  # 10**
                        x[2], x[3], x[4], x[5], x[6], x[7], self.version, self.solver, self.compiler)

    # if self.mode2 != "forward":
    #     if x[5] > self.parameters[5].maxbound:
    #         x[5] = x[5] - np.floor(x[5])
    #     if x[5] < self.parameters[5].minbound:
    #         x[5] = np.ceil(np.abs(x[5])) - np.abs(x[5])

    data = params.solve()
    sim = []
    for val in data:
        sim.append(val)
    if self.mode2=="forward":
        return sim
    else:# The first year of simulation data is ignored (warm-up)
        return sim[366:]
def simulation_air2water_log(self, x):
    # Here the model is actualy startet with one paramter combination
    params = air2water(self.Tw_solution, self.Ta_data, self.tt, x[0], 10 ** x[1], 10 ** x[2], x[3], x[4],
                       x[5], x[6], x[7], self.version, self.solver, self.compiler, self.CFL)

    # if self.mode2!="forward":
    #     if x[5]>self.parameters[5].maxbound:
    #         x[5] = x[5] - np.floor(x[5])
    #     if x[5]<self.parameters[5].minbound:
    #         x[5] = np.ceil(np.abs(x[5]))-np.abs(x[5])

    data = params.solve()
    sim = []
    for val in data:
        sim.append(val)
    # The first year of simulation data is ignored (warm-up)
    if self.mode2 == "forward":
        return sim
    else:  # The first year of simulation data is ignored (warm-up)
        return sim[366:]


def simulation_air2water_no_log(self, x):
    # Here the model is actualy startet with one paramter combination
    params = air2water(self.Tw_solution, self.Ta_data, self.tt, x[0], x[1], x[2], x[3], x[4],
                       x[5], x[6], x[7], self.version, self.solver, self.compiler, self.CFL)

    # if self.mode2 != "forward":
    #     if x[5] > self.parameters[5].maxbound:
    #         x[5] = x[5] - np.floor(x[5])
    #     if x[5] < self.parameters[5].minbound:
    #         x[5] = np.ceil(np.abs(x[5])) - np.abs(x[5])

    data = params.solve()
    sim = []
    for val in data:
        sim.append(val)
    # The first year of simulation data is ignored (warm-up)
    if self.mode2 == "forward":
        return sim
    else:  # The first year of simulation data is ignored (warm-up)
        return sim[366:]


def simulation_air2stream(self, x):
    # Here the model is actualy startet with one paramter combination
    params = air2stream(self.Tw_solution, self.Ta_data, self.Q, self.tt, x[0],  # 10**
                        x[1],  # 10**
                        x[2], x[3], x[4], x[5], x[6], x[7], self.version, self.solver, self.compiler)

    # if self.mode2 != "forward":
    #     if x[5] > self.parameters[5].maxbound:
    #         x[5] = x[5] - np.floor(x[5])
    #     if x[5] < self.parameters[5].minbound:
    #         x[5] = np.ceil(np.abs(x[5])) - np.abs(x[5])

    data = params.solve()
    sim = []
    for val in data:
        sim.append(val)
    # The first year of simulation data is ignored (warm-up)
    if self.mode2 == "forward":
        return sim
    else:  # The first year of simulation data is ignored (warm-up)
        return sim[366:]

def dottyplots(df, theoritical_bounds, parameter_low, parameter_high, final_rmse, error_metric,user_id,group_id,sim_id):
    owd = settings.MEDIA_ROOT

    column_order = list(df.columns)
    column_order.insert(0, column_order.pop(-1))
    df = df[column_order]

    # Renaming the plots as a1 to a8
    fig, axs = plt.subplots(2, 4, figsize=(15, 10))  # 2 rows, 4 columns of subplots

    # Plot each subplot and rename
    for i in range(8):
        row, col = i // 4, i % 4
        axs[row, col].plot(np.array(df.iloc[i]), np.array(df.iloc[8]), 'o', color='r')  # 'o' for dot style
        axs[row, col].plot(theoritical_bounds[0, i], final_rmse, 'o', color='g')
        axs[row, col].plot(theoritical_bounds[1, i], final_rmse, 'o', color='b')  # 'o' for dot style
        axs[row, col].plot(parameter_low[i], final_rmse, 'o', color='k')
        axs[row, col].plot(parameter_high[i], final_rmse, 'o', color='k')
        axs[row, col].set_title(f'a{i + 1}')  # Rename as a1, a2, ..., a8

    for ax in axs.flat:
        if error_metric == "RMSE":
            ax.set(xlabel='Parameter Value', ylabel='RMSE')
        elif error_metric == "NS":
            ax.set(xlabel='Parameter Value', ylabel='NSE')
        elif error_metric == "KGE":
            ax.set(xlabel='Parameter Value', ylabel='KGE')

    custom_lines = [Line2D([0], [0], marker='o', color='red', lw=2),
                    Line2D([0], [0], marker='o', color='black', lw=2),
                    Line2D([0], [0], marker='o', color='green', lw=2),
                    Line2D([0], [0], marker='o', color='blue', lw=2)]
    plt.figlegend(custom_lines,
                  ['PSO iterations', 'Physical bounds', 'Regression 6 parameters', 'Regression 4 parameters'],
                  loc='lower center', ncol=4, frameon=False)
    plt.subplots_adjust(hspace=0.3, wspace=0.4, bottom=0.1)

    plt.savefig(f"{owd}/results/{user_id}_{group_id}/dottyplots_{sim_id}.png", dpi=100)
    plt.close()  # Close the plot to avoid displaying it

    return


class Air2water_OOP:
    #owd = os.getcwd() #Activate this for running in PyCharm
    #owd="/home/dicam01/DEV/Work/air2water" #Activate this for fastapi

    def objective_function_pyswarm(self, x):
        # Replace this with your actual objective function
        sim = self.spot_setup.simulation(x)
        eval = self.spot_setup.evaluation()

        return self.spot_setup.objectivefunction(sim, eval)

    def objective_function_pycup(self, x):
        # Replace this with your actual objective function
        sim = self.spot_setup.simulation(x)
        eval = self.spot_setup.evaluation()

        fitness=self.spot_setup.objectivefunction(sim, eval)
        result=np.array(sim).reshape(1,-1)

        return fitness,result

    def datetimecalc(self,calibration):
        Y = calibration[:, 0]
        M = calibration[:, 1]
        D = calibration[:, 2]
        date = [datetime(int(y), int(m), int(d)) for y, m, d in zip(Y, M, D)]
        return date,np.asarray([d.timetuple().tm_yday / 366 for d in date])

    def __init__(self, user_id, group_id, spot_setup=spot_setup, interpolate=True, n_data_interpolate=7, validation_required=True,
                 model = "air2water", core=1, depth = 14.0, swarmsize=20, phi1=2.0, phi2=2.0, omega=0.5, maxiter=20, numbersim= 2000, method = 'SpotPY', mode = "calibration", error="RMSE", kge_button=True, rmse_button=True, ns_button=True, db_file = None,
                 optimizer = "PSO", solver = "cranknicolson", compiler= "fortran", CFL = 0.9, databaseformat = "custom",
                 computeparametersranges = "Yes", computeparameters="No",
                 parameter_ranges=None,
                 forward_parameters=None,
                 air2waterusercalibrationpath=None,
                 air2streamusercalibrationpath=None,
                 uservalidationpath=None,
                 log_flag = 1, results_file_name = "results.db", resampling_frequency_days = 1,
                 resampling_frequency_weeks = 1, email_send=0, sim_id=0, email_list=['riddick.kakati@unitn.it']):

        self.user_id = user_id
        self.group_id = group_id
        self.model = model
        self.spot_setup = spot_setup
        self.interpolate = interpolate
        self.n_data_interpolate = n_data_interpolate
        self.validation_required= validation_required # Whether to use seperate files for calibration and validation or let algorithm split the data. Set it to true for algorithm based split.
        self.depth = depth
        self.swarmsize=swarmsize
        self.phi1= phi1
        self.phi2= phi2
        self.omega= omega
        self.maxiter=maxiter
        self.numbersim=numbersim
        self.core= core
        self.method = method
        self.mode = mode
        self.error=error
        self.RMSE_button= rmse_button
        self.KGE_button= kge_button
        self.NS_button= ns_button
        self.db_file = db_file
        self.optimizer = optimizer
        self.solver = solver
        self.compiler = compiler
        self.CFL = CFL
        self.databaseformat = databaseformat
        self.computeparametersranges = computeparametersranges
        self.computeparameters = computeparameters
        self.parameter_ranges = parameter_ranges
        self.forward_parameters = forward_parameters
        self.air2waterusercalibrationpath = air2waterusercalibrationpath
        self.air2streamusercalibrationpath = air2streamusercalibrationpath
        self.uservalidationpath = uservalidationpath
        self.log_flag = log_flag
        self.results_file_name = results_file_name
        self.resampling_frequency_days = resampling_frequency_days
        self.resampling_frequency_weeks = resampling_frequency_weeks
        self.email_send = email_send
        self.sim_id = sim_id
        self.email_list = email_list

    def run(self):

        owd = settings.MEDIA_ROOT

        datafolder = owd
        # if os.path.exists(datafolder):
        #     print("Simulation exists.")
        #     exit()
        # else:
        if self.mode != "forward":
            air2_water_params = Air2WaterParameters(self.depth, self.method, self.model)
            if self.computeparametersranges == "No":
                _, theoritical_parameters = air2_water_params.calculate_parameters()
            else:

                if self.model == "air2water":

                    parameters, theoretical_parameters = air2_water_params.calculate_parameters()
                    air2_water_params.save_parameters(self.depth, parameters, theoretical_parameters, self.user_id, self.group_id)
                    for root,dirs,files in os.walk(f"{owd}/parameters/{self.user_id}_{self.group_id}/"):
                        for file in files:
                            if fnmatch.fnmatch(file, 'parameters_depth=*m.yaml'):
                                self.parameter_ranges = os.path.join(root, file)
                        #shutil.copy(self.parameter_ranges,
                                    #datafolder + os.sep + str(f'{self.model}_{self.mode}_parameters.yml'))
                else:
                    _, theoritical_parameters = air2_water_params.calculate_parameters()
                    for root,dirs,files in os.walk(f"{owd}/parameters/{self.user_id}_{self.group_id}/"):
                        for file in files:
                            if fnmatch.fnmatch(file, 'parameters_air2stream.yaml'):
                                self.parameter_ranges = os.path.join(root, file)
                        #shutil.copy(self.parameter_ranges,
                                    #datafolder + os.sep + str(f'{self.model}_{self.mode}_parameters.yml'))
        #else:
           # if self.computeparameters == "No":
                #shutil.copy(self.forward_parameters,
                           # datafolder + os.sep + str(f'{self.model}_{self.mode}_parameters.yml'))
            # else:
            #     do find the optimum parameters file, and ask user if run simulation again

        # if self.mode == "forward":
        #     for file in os.listdir('.' + os.sep + str('config')):
        #         if fnmatch.fnmatch(file, 'parameters_forward.yaml'):
        #             self.parameter_ranges = file
        # else:
        #     if self.model=="air2water":
        #         for file in os.listdir('.' + os.sep + str('config')):
        #             if fnmatch.fnmatch(file, 'parameters_depth=*m.yaml'):
        #                 self.parameter_ranges = file
        #
        #     elif self.model=="air2stream":
        #         for file in os.listdir('.' + os.sep + str('config')):
        #             if fnmatch.fnmatch(file, 'parameters*air2stream.yaml'):
        #                 self.parameter_ranges = file


        #all_parameters1 = read_yaml(datafolder + os.sep + str(f'{self.model}_{self.mode}_parameters.yml'))

        if self.mode != "forward":
            try:
                filename = glob.glob(f"{owd}/parameters/{self.user_id}_{self.group_id}/parameters_depth=*m.yaml")[0]
                with open(filename, 'r') as file:
                    all_parameters1 = yaml.safe_load(file)
            except IndexError:
                raise FileNotFoundError("No parameters.yml file found in the directory")
            all_parameters = all_parameters1['Optimizer']['parameters']

            all_parameters_names = list(all_parameters.keys())

            a1 = globals()[all_parameters.get(all_parameters_names[0], {}).get('distribution')](
                name=all_parameters_names[0],
                low=all_parameters.get('a1',
                                       {}).get(
                    'low'),
                high=all_parameters.get('a1',
                                        {}).get(
                    'high'))
            if self.model == "air2water":
                if self.log_flag:
                    a2 = globals()[all_parameters.get(all_parameters_names[1], {}).get('distribution')](
                        name=all_parameters_names[1],
                        low=math.log10(
                            all_parameters.get(
                                all_parameters_names[1],
                                {}).get('low')),
                        high=math.log10(
                            all_parameters.get(
                                all_parameters_names[1],
                                {}).get(
                                'high')))  # Search in logarithmic space
                    a3 = globals()[all_parameters.get(all_parameters_names[2], {}).get('distribution')](
                        name=all_parameters_names[2],
                        low=math.log10(
                            all_parameters.get(
                                all_parameters_names[2],
                                {}).get('low')),
                        high=math.log10(
                            all_parameters.get(
                                all_parameters_names[2],
                                {}).get(
                                'high')))  # Search in logarithmic space
                else:
                    a2 = globals()[all_parameters.get(all_parameters_names[1], {}).get('distribution')](
                        name=all_parameters_names[1],
                        low=
                        all_parameters.get(
                            all_parameters_names[
                                1],
                            {}).get('low'),
                        high=
                        all_parameters.get(
                            all_parameters_names[
                                1],
                            {}).get(
                            'high'))  # Search in logarithmic space
                    a3 = globals()[all_parameters.get(all_parameters_names[2], {}).get('distribution')](
                        name=all_parameters_names[2],
                        low=all_parameters.get(
                            all_parameters_names[2],
                            {}).get('low'),
                        high=all_parameters.get(
                            all_parameters_names[2],
                            {}).get(
                            'high'))
            else:
                a2 = globals()[all_parameters.get(all_parameters_names[1], {}).get('distribution')](
                    name=all_parameters_names[1],
                    low=
                    all_parameters.get(
                        all_parameters_names[
                            1],
                        {}).get('low'),
                    high=
                    all_parameters.get(
                        all_parameters_names[
                            1],
                        {}).get(
                        'high'))  # Search in logarithmic space
                a3 = globals()[all_parameters.get(all_parameters_names[2], {}).get('distribution')](
                    name=all_parameters_names[2],
                    low=all_parameters.get(
                        all_parameters_names[2],
                        {}).get('low'),
                    high=all_parameters.get(
                        all_parameters_names[2],
                        {}).get(
                        'high'))
            a4 = globals()[all_parameters.get(all_parameters_names[3], {}).get('distribution')](
                name=all_parameters_names[3],
                low=all_parameters.get(
                    all_parameters_names[3],
                    {}).get('low'),
                high=all_parameters.get(
                    all_parameters_names[3],
                    {}).get('high'))
            a5 = globals()[all_parameters.get(all_parameters_names[4], {}).get('distribution')](
                name=all_parameters_names[4],
                low=all_parameters.get(
                    all_parameters_names[4],
                    {}).get('low'),
                high=all_parameters.get(
                    all_parameters_names[4],
                    {}).get('high'))
            a6 = globals()[all_parameters.get(all_parameters_names[5], {}).get('distribution')](
                name=all_parameters_names[5],
                low=all_parameters.get(
                    all_parameters_names[5],
                    {}).get('low'),
                high=all_parameters.get(
                    all_parameters_names[5],
                    {}).get('high'))
            a7 = globals()[all_parameters.get(all_parameters_names[6], {}).get('distribution')](
                name=all_parameters_names[6],
                low=all_parameters.get(
                    all_parameters_names[6],
                    {}).get('low'),
                high=all_parameters.get(
                    all_parameters_names[6],
                    {}).get('high'))
            a8 = globals()[all_parameters.get(all_parameters_names[7], {}).get('distribution')](
                name=all_parameters_names[7],
                low=all_parameters.get(
                    all_parameters_names[7],
                    {}).get('low'),
                high=all_parameters.get(
                    all_parameters_names[7],
                    {}).get('high'))

        else:
            try:
                filename = glob.glob(f"{owd}/parameters/{self.user_id}_{self.group_id}/parameters_forward.yaml")[0]
                with open(filename, 'r') as file:
                    all_parameters1 = yaml.safe_load(file)
            except IndexError:
                raise FileNotFoundError("No parameters.yml file found in the directory")
            all_parameters = all_parameters1['Optimized']['parameters']
            a1 = all_parameters.get('a1')
            a2 = all_parameters.get('a2')
            a3 = all_parameters.get('a3')
            a4 = all_parameters.get('a4')
            a5 = all_parameters.get('a5')
            a6 = all_parameters.get('a6')
            a7 = all_parameters.get('a7')
            a8 = all_parameters.get('a8')

        parameters = [a1, a2, a3, a4, a5, a6, a7, a8]

        if self.model == "air2water":
            usercalibrationdatapath = self.air2waterusercalibrationpath
        elif self.model == "air2stream":
            usercalibrationdatapath = self.air2streamusercalibrationpath

        df1 = np.loadtxt(usercalibrationdatapath)
        #shutil.copy(usercalibrationdatapath, datafolder + os.sep + str(f'calibration.txt'))

        uservalidationdatapath = self.uservalidationpath
        if self.validation_required == False:
            df2 = np.loadtxt(uservalidationdatapath)
            #shutil.copy(uservalidationdatapath, datafolder + os.sep + str(f'validation.txt'))

        processor = YearlyDataProcessor(df1,self.n_data_interpolate)
        dfint1, num_missing_col3, missing_col3 = processor.mean_year()
        np.savetxt(f"{owd}/timeseries/{self.user_id}_{self.group_id}/calibration_interpolated.csv", dfint1, delimiter=",", fmt='%s')

        print(f"There are {num_missing_col3} missing data.")
        if self.interpolate == True and num_missing_col3:
            print(f"Interpolating...")
            if self.validation_required == False:
                processor2 = YearlyDataProcessor(df2)
                dfint2, num_missing_col3_2, missing_col3_2 = processor2.mean_year()
                np.savetxt(f"{owd}/timeseries/{self.user_id}_{self.group_id}/validation_interpolated.csv", dfint2, delimiter=",", fmt='%s')
                calibration = dfint1
                validation = dfint2
            else:
                calibration, validation = train_test_split(dfint1, test_size=0.1, shuffle=False)
        else:
            if num_missing_col3 != 0:
                print(f"There are {num_missing_col3} data, and interpolation is false. It might affect results.")
                calibration = df1
                if self.validation_required == False:
                    validation = df2
            else:
                calibration = df1
                if self.validation_required == False:
                    validation = df2

        _,tt = self.datetimecalc(calibration)

        # To replicate the first year of data so that we dont loose data
        if self.mode != "forward":
            initial_rows = calibration[:366].copy()
            calibration = np.concatenate([initial_rows, calibration])

        start_time = time.time()
        if self.model == "air2water":
            if self.mode == "forward":
                self.spot_setup.simulation = simulation_air2water_no_log
            else:
                if self.log_flag:
                    if self.method == "PYSWARMS_PSO" or self.optimizer == "PSO_A2WS.py":
                        self.spot_setup.simulation = simulation_air2water_log_pyswarms
                    else:
                        self.spot_setup.simulation = simulation_air2water_log
                else:
                    if self.method == "PYSWARMS_PSO" or self.optimizer == "PSO_A2WS.py":
                        self.spot_setup.simulation = simulation_air2water_no_log_pyswarms
                    else:
                        self.spot_setup.simulation = simulation_air2water_no_log

        elif self.model == "air2stream":
            if self.method =="PYSWARMS_PSO":
                self.spot_setup.simulation = simulation_air2stream_pyswarms
            else:
                self.spot_setup.simulation = simulation_air2stream

        if self.mode != "forward":
            self.spot_setup = self.spot_setup(parameters,
                                    model = self.model, metric=self.error, df = calibration, tt = tt, db_file = self.db_file, optimizer = self.optimizer,
            threshold = None, solver = self.solver, CFL = self.CFL,mode2=self.mode)

            if self.method == "PYCUP":
                s = SpotpySetupConverter()
                s.convert(self.spot_setup)

                if self.optimizer == "GLUE":
                    cp.GLUE.run(self.swarmsize, s.dim, lb=s.lb, ub=s.ub, fun=self.objective_function_pycup, args=())

                elif self.optimizer == "SSA":
                    cp.SSA.run(pop=self.swarmsize, dim=s.dim, lb=s.lb, ub=s.ub, MaxIter=self.maxiter, fun=self.objective_function_pycup)

                elif self.optimizer == "GWO":
                    cp.GWO.run(pop=self.swarmsize, dim=s.dim, lb=s.lb, ub=s.ub, MaxIter=self.maxiter, fun=self.objective_function_pycup)

                elif self.optimizer == "MFO":
                    cp.MFO.run(pop=self.swarmsize, dim=s.dim, lb=s.lb, ub=s.ub, MaxIter=self.maxiter, fun=self.objective_function_pycup)

                elif self.optimizer == "PSO":
                    cp.PSO.runMP(self.swarmsize, s.dim, s.lb, s.ub, self.maxiter, self.objective_function_pycup, args=(), n_jobs=self.core)

                elif self.optimizer == "SCA":
                    cp.SCA.runMP(pop=self.swarmsize, dim=s.dim, lb=s.lb, ub=s.ub, MaxIter=self.maxiter, fun=self.objective_function_pycup, n_jobs=self.core)

                elif self.optimizer == "SOA":
                    cp.SOA.run(pop=self.swarmsize, dim=s.dim, lb=s.lb, ub=s.ub, MaxIter=self.maxiter, fun=self.objective_function_pycup, fc=2)

                elif self.optimizer == "TSA":
                    cp.TSA.run(pop=self.swarmsize, dim=s.dim, lb=s.lb, ub=s.ub, MaxIter=self.maxiter, fun=self.objective_function_pycup)

                elif self.optimizer == "WOA":
                    cp.WOA.run(pop=self.swarmsize, dim=s.dim, lb=s.lb, ub=s.ub, MaxIter=self.maxiter, fun=self.objective_function_pycup)

                elif self.optimizer == "NSGAII":
                    cp.NSGA2.run(pop=self.swarmsize, dim=s.dim, lb=s.lb, ub=s.ub, MaxIter=self.maxiter, fun=self.objective_function_pycup)

                saver = cp.save.RawDataSaver.load(f"{owd}/results/{self.user_id}_{self.group_id}/RawResult.rst")
                t = SpotpyDbConverter()
                t.RawSaver2csv(f"{owd}/parameters/{self.user_id}_{self.group_id}/RawResult.rst", f"{owd}/parameters/{self.user_id}_{self.group_id}/{self.spot_setup.db_file[:-3]}.csv")
                results = spotpy.analyser.load_csv_results(f"{owd}/results/{self.user_id}_{self.group_id}/{self.spot_setup.db_file[:-3]}.csv")
                df3 = pd.DataFrame(results)
                df3 = df3.iloc[:, :9]
                common_data = results
                best_results = str(spotpy.analyser.get_best_parameterset(results, maximize=False)[0])[1:-1]
                data_list = [float(x) for x in best_results.split(', ')]
                best_position = np.array(data_list)
                conn = sqlite3.connect(self.spot_setup.db_file)
                df3.to_sql(f"Calibration_Data", conn, if_exists='replace', index=False)
                conn.close()
                df3.to_csv(f"{self.spot_setup.db_file[:-3]}.csv", index=False)
                print(f"Best Results {self.optimizer}:",
                      best_results)

                bestindex, bestobjf = spotpy.analyser.get_minlikeindex(common_data)
                best_model_run = common_data[bestindex]

            if self.method == "PYSWARMS_PSO":
                s = SpotpySetupConverter()
                s.convert(self.spot_setup)

                if self.optimizer == "PSO_GLOBAL":
                    options = {'c1': self.phi1, 'c2': self.phi2, 'w': self.omega}
                    bounds = (s.lb.T, s.ub.T)
                    sampler = ps.single.GlobalBestPSO(n_particles=self.swarmsize, dimensions=8, options=options,
                                                      bounds=bounds)
                    cost, best_position = sampler.optimize(self.objective_function_pyswarm, iters=self.maxiter) #The objective function used in pyswarms corresponds to the spot_setup.simulation function. But it returns error. Why?

                best_cost= pd.DataFrame(sampler.cost_history)
                best_pos= pd.DataFrame(sampler.swarm.pbest_pos)
                df3=pd.concat([best_cost,best_pos],axis=1)
                headers = ['like1'] + [f'para{i}' for i in np.arange(1, 1+len(best_pos.columns))]
                df3.columns=headers
                df3.to_csv(f"{self.spot_setup.db_file[:-3]}.csv", index=False)
                results = spotpy.analyser.load_csv_results(f"{self.spot_setup.db_file[:-3]}")
                common_data = results
                best_results = str(spotpy.analyser.get_best_parameterset(results, maximize=False)[0])[1:-1]
                data_list = [float(x) for x in best_results.split(', ')]
                best_position = np.array(data_list)
                conn = sqlite3.connect(self.spot_setup.db_file)
                df3.to_sql(f"{self.spot_setup.db_file[:-3]}", conn, if_exists='replace', index=False)
                conn.close()
                df3.to_csv(f"{self.spot_setup.db_file[:-3]}.csv", index=False)
                print(f"Best Results {self.optimizer}:",
                      best_results)

                bestindex, bestobjf = spotpy.analyser.get_minlikeindex(common_data)
                best_model_run = common_data[bestindex]

            elif self.method == "SpotPY":
                if self.optimizer == "PSO":

                    s = SpotpySetupConverter()
                    s.convert(self.spot_setup)

                    pso_oop.dbformat = self.databaseformat
                    sampler = pso_oop(self.objective_function_pyswarm, self.spot_setup.db_file, pso_oop.dbformat, s.lb, s.ub,
                                      swarmsize=self.swarmsize, phip=self.phi1,
                                      phig=self.phi2, omega=self.omega, maxiter=self.maxiter, minfunc=1e-8, processes=self.core,
                                      threshold=self.spot_setup.threshold)#,model=self.model)
                    if sampler.dbformat == "custom":
                        best_position, _ = sampler.pso()
                    elif sampler.dbformat == "ram":
                        best_position, _, df3, spotpyreader = sampler.pso()
                elif self.optimizer == "PSO_A2WS":
                    s = SpotpySetupConverter()
                    s.convert(self.spot_setup)

                    pso_oop_new.dbformat = self.databaseformat
                    sampler = pso_oop_new(self.objective_function_pyswarm, self.spot_setup.db_file, pso_oop_new.dbformat,
                                      s.lb, s.ub,
                                      swarmsize=self.swarmsize, phip=self.phi1,
                                      phig=self.phi2, omega=self.omega, maxiter=self.maxiter, minfunc=1e-8,
                                      processes=self.core,
                                      threshold=self.spot_setup.threshold ,model=self.model)
                    if sampler.dbformat == "custom":
                        best_position, _ = sampler.pso()
                    elif sampler.dbformat == "ram":
                        best_position, _, df3, spotpyreader = sampler.pso()

                    s = SpotpySetupConverter()
                    s.convert(self.spot_setup)

                    if sampler.dbformat == "custom":
                        best_position, _ = sampler.pso()
                    elif sampler.dbformat == "ram":
                        best_position, _, df3, spotpyreader = sampler.pso()
                else:
                    dict = {}
                    dict['spotpy_algo'] = str(self.optimizer.lower())
                    rep = self.numbersim
                    variable = globals()[dict.get('spotpy_algo')]
                    sampler = globals()[dict.get('spotpy_algo')](self.spot_setup, dbformat=self.databaseformat,
                                                                 save_sim=False)  # dbformat="custom"#, save_sim=False)
                    if self.optimizer == "ABC":
                        sampler.optimization_direction = "minimize"
                        sampler.sample(rep, eb=2000, a=0.2, peps=0.00001)
                    elif self.optimizer == "DDS":
                        sampler.optimization_direction = "minimize"
                        sampler.sample(rep, trials=3)
                    elif self.optimizer == "DEMCZ":
                        sampler.optimization_direction = "minimize"
                        sampler.sample(rep, nChains=5, DEpairs=3, eps=0.01)
                    elif self.optimizer == "DREAM":
                        sampler.optimization_direction = "minimize"
                        sampler.sample(rep, nChains=5, nCr=4, delta=0.1, c=0.5, eps=1e-8, convergence_limit=0.001,
                                       runs_after_convergence=2, acceptance_test_option=6)
                    elif self.optimizer == "FAST":
                        sampler.optimization_direction = "minimize"
                        sampler.sample(rep)
                    elif self.optimizer == "FSCABC":
                        sampler.optimization_direction = "minimize"
                        sampler.sample(rep, eb=2000, a=0.2, peps=0.00001, kpow=4)
                    elif self.optimizer == "LHS":
                        sampler.optimization_direction = "minimize"
                        sampler.sample(rep)
                    elif self.optimizer == "LIST_SAMPLER":
                        sampler.optimization_direction = "minimize"
                        sampler.sample(rep)
                    elif self.optimizer == "MC":
                        sampler.optimization_direction = "minimize"
                        sampler.sample(rep)
                    elif self.optimizer == "MCMC":
                        sampler.optimization_direction = "minimize"
                        sampler.sample(rep)
                    elif self.optimizer == "MLE":
                        sampler.optimization_direction = "minimize"
                        sampler.sample(rep)
                    # elif self.optimizer == "NSGAII":
                    #     sampler.optimization_direction = "minimize"
                    #     sampler.sample(generations=100, n_obj=1, n_pop=100)
                    elif self.optimizer == "PADDS":
                        sampler.optimization_direction = "minimize"
                        sampler.sample(rep)
                    elif self.optimizer == "ROPE":
                        sampler.optimization_direction = "minimize"
                        sampler.sample(rep, repetitions_first_run=20, subsets=5, percentage_first_run=0.5,
                                       percentage_following_runs=0.5, NDIR=2000)
                    elif self.optimizer == "SA":
                        sampler.optimization_direction = "minimize"
                        sampler.sample(rep)
                    elif self.optimizer == "SCEUA":
                        sampler.optimization_direction = "minimize"
                        sampler.sample(rep, ngs=7, kstop=3, peps=0.1, pcento=0.1)

                # result_df_raw=pd.DataFrame(result_list)
                # df_expanded = pd.DataFrame(result_df_raw.iloc[:,1].tolist())
                # result_df = pd.concat([result_df_raw.iloc[:,0], df_expanded], axis=1)
                # headers= ['like1'] + [f'para{i}' for i in np.arange(1,len(result_df.columns))]
                # result_df.columns = headers
                #
                # # filename2= cwd + os.sep + str('PSO_TEST.csv')
                # # result_df.to_csv(filename2, index=False)

                if sampler.dbformat == "custom":
                    # Start the sampler, one can specify ngs, kstop, peps and pcento id desired

                    # For custom database.csv
                    # self.spot_setup.database.close()

                    # Connect to the SQLite database
                    conn = sqlite3.connect(self.spot_setup.db_file)

                    # Use pandas to read the query result directly into a DataFrame
                    df = pd.read_sql_query("SELECT * FROM RESULTS", conn)
                    df3=df
                    # Close the database connection
                    conn.close()

                    # Now you can use the results with Spotpy
                    # Use for sql
                    # Filter rows based on the threshold condition
                    # if threshold != None:
                    #     filtered_df = df[df["like1"] < threshold]
                    #     filtered_df.to_csv("filtered_SCEUA_hymod.csv", index=False)
                    #     results = spotpy.analyser.load_csv_results(
                    #         "filtered_SCEUA_hymod")  # SCEUA_hymod_filtered for filtered database. Run filter_db.py before that
                    # else:
                    df.to_csv(f"{self.spot_setup.db_file[:-3]}.csv", index=False)

                    # Write the filtered DataFrame to a new CSV file
                    results = spotpy.analyser.load_csv_results(
                        self.spot_setup.db_file[
                        :-3])  # SCEUA_hymod_filtered for filtered database. Run filter_db.py before that

                    if self.optimizer != "PSO":
                        best_results = str(spotpy.analyser.get_best_parameterset(results, maximize=False)[0])[1:-1]
                        data_list = [float(x) for x in best_results.split(', ')]
                        best_position = np.array(data_list)
                        print(f"Best Results {self.optimizer}:", best_position)

                    # Load the results gained with the sceua sampler, stored in SCEUA_hymod.csv

                    print(f"Best Results {self.optimizer}:",
                          spotpy.analyser.get_best_parameterset(results, maximize=False))
                    bestindex, bestobjf = spotpy.analyser.get_minlikeindex(results)
                    best_model_run = results[bestindex]


                elif sampler.dbformat == "ram":
                    if self.optimizer == "PSO":
                        results = df3
                        common_data = spotpyreader
                    else:
                        results = sampler.datawriter.data
                        df3 = pd.DataFrame(results)
                        common_data = results
                        best_results = str(spotpy.analyser.get_best_parameterset(results, maximize=False)[0])[1:-1]
                        data_list = [float(x) for x in best_results.split(', ')]
                        best_position = np.array(data_list)
                        print(f"Best Results {self.optimizer}:", best_position)

                    conn = sqlite3.connect(self.spot_setup.db_file)
                    df3.to_sql(f"Calibration_data", conn, if_exists='replace', index=False)
                    conn.close()
                    df3.to_csv(f"{owd}/results/{self.user_id}_{self.group_id}/self.spot_setup.db_file[:-3].csv", index=False)
                    print(f"Best Results {self.optimizer}:",
                          spotpy.analyser.get_best_parameterset(common_data, maximize=False))

                    bestindex, bestobjf = spotpy.analyser.get_minlikeindex(common_data)
                    best_model_run = common_data[bestindex]

            #db_file_location = owd
            #shutil.copy(db_file_location, datafolder + os.sep + str(f'{self.optimizer}_results.db'))
            print("Total Time Taken:", time.time() - start_time)

            # Plot how the objective function was minimized during sampling
            fig = plt.figure(1, figsize=(9, 6))
            plt.plot(results["like1"])
            #plt.show()
            plt.ylabel(str(self.error))
            plt.xlabel("Iteration")
            fig.savefig(f"{owd}/results/{self.user_id}_{self.group_id}/objectivefunctiontrace_{self.sim_id}.png", dpi=100)
        else:
            best_position = parameters
            # Plot the best model run
            # Find the run_id with the minimal objective function value

            # Select best model run

            # Filter results for simulation results
            # fields = [word for word in best_model_run.dtype.names if word.startswith("sim")]

            info_dict = {
                'Optimized': {
                    'model': self.model,
                    'parameters': {
                    }

                }
            }
            best_position_array = np.array(best_position)
            for i in range(len(best_position_array)):
                parameter_name = f'a{i + 1}'
                info_dict["Optimized"]["parameters"][parameter_name] = float(best_position_array[i])

            # Save parameters as YAML file
            with open(f"{owd}/parameters/{self.user_id}_{self.group_id}/best_parameters.yaml", "w") as file:
                yaml.dump(info_dict, file, default_flow_style=False)
            #shutil.copy(f'{owd}/best_parameters.yaml', datafolder + os.sep + str(f'best_parameters.yaml'))


            self.spot_setup = self.spot_setup(parameters,
                                              model=self.model, metric=self.error, df=calibration, tt=tt,
                                              db_file=self.db_file, optimizer=self.optimizer,
                                              threshold=None, CFL=self.CFL, mode2=self.mode)


        self.spot_setup.mode2="forward"
        best_simulation = self.spot_setup.simulation(best_position)[366:]
        best_evaluation = self.spot_setup.evaluation()[366:]
        airtempdata = calibration[:, 3][366:]
        date, _ = self.datetimecalc(calibration)

        rmse2=spotpy.objectivefunctions.rmse(best_evaluation,best_simulation)

        print('RMSE with optimized parameters:',rmse2)

        df_final_means_original = pd.DataFrame({
            'Date': date[366:],
            'Air_temperature_data': airtempdata,
            'Best_simulation': best_simulation,
            'Evaluation_(Water_temperature_data)': best_evaluation
        })

        # Set the index using the 'date' column
        df_final_means = df_final_means_original.copy()
        df_final_means.set_index('Date', inplace=True)
        conn = sqlite3.connect(self.results_file_name)
        df_final_means_original.to_sql(f"Calibration_data", conn, if_exists='replace', index=False)
        conn.close()
        df_final_means_original.to_csv(f"{self.results_file_name[:-3]}.csv", index=False)

        #results_file_location = str(owd + os.sep + self.results_file_name)
        #shutil.copy(results_file_location, datafolder + os.sep + str(f'results.db'))

        if self.resampling_frequency_days == None:
            self.resampling_frequency_days = 1
        if self.resampling_frequency_weeks == None:
            self.resampling_frequency_weeks = 1
        one_day_average= df_final_means.resample('1D').mean()
        one_day_rmse= self.spot_setup.objectivefunction(one_day_average.iloc[:, 2], one_day_average.iloc[:, 1])
        self.rmse=round(one_day_rmse,3)
        few_days_average = df_final_means.resample(f'{self.resampling_frequency_days}D').mean()
        few_weeks_average = df_final_means.resample(f'{self.resampling_frequency_weeks}W').mean()
        weekly_average = df_final_means.resample('W').mean()
        monthly_average = df_final_means.resample('M').mean()
        # weekly_rmse = spotpy.objectivefunctions.rmse(weekly_average.iloc[:, 2], weekly_average.iloc[:, 1])
        monthly_rmse = self.spot_setup.objectivefunction(monthly_average.iloc[:, 2], monthly_average.iloc[:, 1])
        few_days_rmse = self.spot_setup.objectivefunction(few_days_average.iloc[:, 2], few_days_average.iloc[:, 1])
        few_weeks_rmse = self.spot_setup.objectivefunction(few_weeks_average.iloc[:, 2], few_weeks_average.iloc[:, 1])

        monthly_ns = ''
        monthly_kge = ''
        monthly_rms = ''
        few_days_ns = ''
        few_days_kge = ''
        few_days_rms = ''
        few_weeks_ns = ''
        few_weeks_kge = ''
        few_weeks_rms = ''

        if self.error=="RMSE":
            if self.KGE_button:
                self.spot_setup.metric = "KGE"
                monthly_kge = self.spot_setup.objectivefunction(monthly_average.iloc[:, 2],
                                                                 monthly_average.iloc[:, 1])
                few_days_kge = self.spot_setup.objectivefunction(few_days_average.iloc[:, 2],
                                                                  few_days_average.iloc[:, 1])
                few_weeks_kge = self.spot_setup.objectivefunction(few_weeks_average.iloc[:, 2],
                                                                   few_weeks_average.iloc[:, 1])
            if self.NS_button:
                self.spot_setup.metric = "NS"
                monthly_ns = self.spot_setup.objectivefunction(monthly_average.iloc[:, 2],
                                                                monthly_average.iloc[:, 1])
                few_days_ns = self.spot_setup.objectivefunction(few_days_average.iloc[:, 2],
                                                                 few_days_average.iloc[:, 1])
                few_weeks_ns = self.spot_setup.objectivefunction(few_weeks_average.iloc[:, 2],
                                                                  few_weeks_average.iloc[:, 1])
        elif self.error=="NS":
            if self.RMSE_button:
                self.spot_setup.metric = "RMSE"
                monthly_rms = self.spot_setup.objectivefunction(monthly_average.iloc[:, 2],
                                                                monthly_average.iloc[:, 1])
                few_days_rms = self.spot_setup.objectivefunction(few_days_average.iloc[:, 2],
                                                                 few_days_average.iloc[:, 1])
                few_weeks_rms = self.spot_setup.objectivefunction(few_weeks_average.iloc[:, 2],
                                                                  few_weeks_average.iloc[:, 1])
            if self.KGE_button:
                self.spot_setup.metric = "KGE"
                monthly_kge = self.spot_setup.objectivefunction(monthly_average.iloc[:, 2],
                                                                 monthly_average.iloc[:, 1])
                few_days_kge = self.spot_setup.objectivefunction(few_days_average.iloc[:, 2],
                                                                  few_days_average.iloc[:, 1])
                few_weeks_kge = self.spot_setup.objectivefunction(few_weeks_average.iloc[:, 2],
                                                                   few_weeks_average.iloc[:, 1])

        elif self. error=="KGE":
            if self.NS_button:
                self.spot_setup.metric = "NS"
                monthly_ns = self.spot_setup.objectivefunction(monthly_average.iloc[:, 2],
                                                                monthly_average.iloc[:, 1])
                few_days_ns = self.spot_setup.objectivefunction(few_days_average.iloc[:, 2],
                                                                 few_days_average.iloc[:, 1])
                few_weeks_ns = self.spot_setup.objectivefunction(few_weeks_average.iloc[:, 2],
                                                                  few_weeks_average.iloc[:, 1])

            if self.RMSE_button:
                self.spot_setup.metric = "RMSE"
                monthly_rms = self.spot_setup.objectivefunction(monthly_average.iloc[:, 2],
                                                                monthly_average.iloc[:, 1])
                few_days_rms = self.spot_setup.objectivefunction(few_days_average.iloc[:, 2],
                                                                 few_days_average.iloc[:, 1])
                few_weeks_rms = self.spot_setup.objectivefunction(few_weeks_average.iloc[:, 2],
                                                                  few_weeks_average.iloc[:, 1])


        fig = plt.figure(figsize=(9, 6))
        ax = plt.subplot(1, 1, 1)
        default_label=f"{self.error} (D,W,M)=" + str(f"{round(few_days_rmse, 2)},") + str(
                f" {round(few_weeks_rmse, 2)},") + str(f" {round(monthly_rmse, 2)}")
        if monthly_ns!='':
            default_label = default_label + f"\n NSE (D,W,M)= {round(few_days_ns, 2)}, {round(few_weeks_ns, 2)}, {round(monthly_ns, 2)}"
        if monthly_kge!='':
            default_label = default_label + f"\n KGE (D,W,M)= {round(few_days_kge, 2)}, {round(few_weeks_kge, 2)}, {round(monthly_kge, 2)}"
        if monthly_rms!='':
            default_label = default_label + f"\n RMSE (D,W,M)= {round(few_days_rms, 2)}, {round(few_weeks_rms, 2)}, {round(monthly_rms, 2)}"
        ax.plot(
            df_final_means['Best_simulation'],
            color="black",
            linestyle="solid",
            label=default_label,
        )
        ax.plot(df_final_means['Evaluation_(Water_temperature_data)'], "r.", markersize=3,
                label="Observed water temperature data")
        ax.plot(df_final_means['Air_temperature_data'], color="blue", linewidth=0.2,
                label="Observed air temperature data")
        plt.xlabel("Year")
        plt.ylabel("Temperature")
        plt.legend(loc="upper right")
        fig.savefig(f"{owd}/results/{self.user_id}_{self.group_id}/best_modelrun_{self.sim_id}.png", dpi=100)
        self.results_time_series = BytesIO()
        fig.savefig(self.results_time_series, format="png", dpi=100)
        self.results_time_series.seek(0)
        plt.close()

        if self.email_send == 1:
            send_email(self.model, self.email_list, self.results_file_name)

        if self.mode != "forward":
            parameters_low=[self.spot_setup.parameters[i].minbound for i in range(len(parameters))]
            parameters_high=[self.spot_setup.parameters[i].maxbound for i in range(len(parameters))]
            self.Dottyplots= dottyplots(df3,theoretical_parameters,parameters_low,parameters_high,one_day_rmse,self.error,self.user_id,self.group_id,self.sim_id)
        else:
            self.Dottyplots=None

        '''if self.mode!="forward" and rmse2.round(2)!=bestobjf.round(2):
            run_count=None
            print("Internal Error: Please re-run simulation!")'''

        return num_missing_col3


if __name__ == "__main__":
    owd = os.getcwd()  # Activate this for running in PyCharm
    Run = Air2water_OOP(method="SpotPY",optimizer="PSO",swarmsize=10,maxiter=10,core=1,parameter_ranges=str(owd + os.sep + str('config') + os.sep + 'parameters_depth=14m.yaml'),
                 forward_parameters=str(owd + os.sep + str('config') + os.sep + 'parameters_forward.yaml'),
                 air2waterusercalibrationpath=str(owd + os.sep + "data" + os.sep + "stndrck_sat_cc.txt"),
                 air2streamusercalibrationpath=str(owd + os.sep + "data" + os.sep + "SIO_2011_cc.txt"),
                 uservalidationpath=str(owd + os.sep + "data" + os.sep + "stndrck_sat_cv3.txt"),computeparameters=False)
    num_missing_col3=Run.run()
