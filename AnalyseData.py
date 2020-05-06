# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import matplotlib.pylab as plt

# For each key (corresponding to the month number), the associated season is given
SeasonDict={1: "Winter", 2:"Winter", 3:"Spring", 4:"Spring", 5:"Spring", 6:"Summer", 7:"Summer", 8:"Summer", 9:"Autumn", 10:"Autumn", 11:"Autumn", 12:"Winter"}

def GroupMonthBySeas(x):
    """Split data into seasons"""
    return SeasonDict[x.month]


def dailyExtremum(df2study, path2SaveResults, save = False):
    """Calculates the daily minimum and maximum value for each signal.
    
	Parameters
	_ _ _ _ _ _ _ _ _ _ 

			df2study : DataFrame
				The temperature data that has to be processed
            path2SaveResults : string
                Name of the URL where to save the resulting DataFrame if 'save' = True
            save : boolean, default False
                Whether or not the results should be saved in a file

	Returns 
	_ _ _ _ _ _ _ _ _ _ 

			results : pd.DataFrame
                DataFrame containing the daily minimum and maximum for each signal"""
    extremums = ["MIN", "MAX"]
    stations = df2study.columns
    df_min = df2study.resample("D").min()
    df_max = df2study.resample("D").max()
    
    multiInd_columns = [[s for i in extremums for s in stations], 
                        [i for i in extremums for s in stations]]
    results = pd.DataFrame(df_min.join(df_max, rsuffix = "_MAX").values,
                          index = pd.DatetimeIndex(sorted(set(df2study.index.date))), 
                          columns = multiInd_columns)
    if save:
        results.to_csv(path2SaveResults+"dailyExtremum.csv")
    
    return results


def djuCalculation(dfDailyExt, path2SaveFiles, path2SaveFigures, djuBelow = True, refValue = 18, 
                   djuStart = "10-01", djuEnd = "05-20",
                   saveFile = True, saveFigure = True):
    """Calculates for each year of the dataset the degree-day number according
    to the Météo-France definition (2005). For more information about the 
    implementation, cf. Bernard (2017). Note that the period for calculation 
    is always defined in a same year. Thus if you set a period from 1st of October
    to 1st of May, the calculation will be performed for each year from january
    to May and added to the ones performed from October to December.
    
    References:
    Jérémy Bernard. Signature géographique et météorologique des variations spatiales
    et temporelles de la température de l'air au sein d'une zone urbaine.
    Génie civil. École centrale de Nantes, 2017. Français. ⟨NNT : 2017ECDN0006⟩.
    ⟨tel-01449935v2⟩
    
    Météo-France, D. d. l. C. (2005). Fiche méthode Degrés Jours. URL :
        http://climatheque.meteo.fr/Docs/DJC-methode.pdf
    
	Parameters
	_ _ _ _ _ _ _ _ _ _ 

			dfDailyExt : DataFrame
				The temperature data that has to be processed (should be a 
                multiindex dataframe with two levels of columns: the first
                is the name of the station, the second should contain the
                daily "MIN" and "MAX")
            djuBelow : boolean, default True
                Whether the DJU calculated are below a certain threshold (for example
                for heating days), the parameter should be set to 'True', if the
                DJU are or above a certain threshold  (for example for cooling days),
                the parameter should be set to 'False'.
            refValue : float, default 18
                Temperature used as threshold for the DJU calculation
            djuStart : "String", default "10-01"
                Month and day used as the beginning of the period  of calculation of the degree-days
                (month and day should be separated by '-')
            djuEnd : "String", default "05-20"
                Month and day used as the end of the period of calculation of the degree-days
                (month and day should be separated by '-')
            path2SaveFiles : string
                Name of the URL where to save the resulting DataFrame if 'saveFile' = True
            path2SaveFigures : string
                Name of the URL where to save the resulting Figure if 'saveFigure' = True
            saveFile : boolean, default True
                Whether or not the results should be saved in a file
            saveFigure : boolean, default True
                Whether or not thefigure should be saved

	Returns 
	_ _ _ _ _ _ _ _ _ _ 

			results : pd.DataFrame
                DataFrame containing the DJU for each year and each station 
                (column of the initial data)"""
    dailyMin = dfDailyExt.xs("MIN", axis = 1, level = 1)
    dailyMax = dfDailyExt.xs("MAX", axis = 1, level = 1)
    
    # Creates the start and en date (year meaningless)
    startDate = pd.datetime(1998, int(djuStart.split("-")[0]), int(djuStart.split("-")[1]))
    endDate = pd.datetime(1998, int(djuEnd.split("-")[0]), int(djuEnd.split("-")[1]))
    
    # Creates the datetime index to keep for the analysis
    if (startDate > endDate):
        index2keep = pd.concat([pd.Series(index=
                                          pd.date_range(start = pd.datetime(y,1,1),
                                                         end = pd.datetime(y,endDate.month,endDate.day),
                                                         freq = pd.offsets.Day(1)).union(\
                                        pd.date_range(start = pd.datetime(y,startDate.month,startDate.day),
                                                         end = pd.datetime(y,12,31),
                                                         freq = pd.offsets.Day(1))))
                                for y in sorted(set(dfDailyExt.index.year))]).index
    else:
        index2keep = pd.concat([pd.Series(index=
                                          pd.date_range(start = pd.datetime(y,startDate.month,startDate.day),
                                                         end = pd.datetime(y,endDate.month,endDate.day),
                                                         freq = pd.offsets.Day(1)))
                                for y in sorted(set(dfDailyExt.index.year))]).index     

    # For heating days calculation
    if djuBelow:
        # Add the days having Tmax < refValue
        DJ1 = refValue-dfDailyExt.mean(axis = 1, level = 0)[dailyMax < refValue].reindex(index2keep)

        # Add the days having Tmax > refValue but Tmin < refValue
        df_selection = (dailyMin < refValue) & (dailyMax > refValue)
        DJ2 = (refValue-dailyMin).mul(0.08+0.42*(refValue-dailyMin).divide(\
               dailyMax-dailyMin))[df_selection].reindex(index2keep)
        
        # Sum all DJU and group results by years
        DJtot = DJ1.resample("AS").sum().add(DJ2.resample("AS").sum())
        
        # When no data for one year, set values to nan
        df_filter_nan = dailyMin.isna() & dailyMax.isna()
        result = df_set_nan(DJtot, dfDailyExt.mean(axis = 1, level = 0), df_filter_nan, freq = "AS", nb_nan = 2)
        
        # Plot the results
        fig = plt.figure()
        result.boxplot(fontsize = 10, rot = 45)
        
        if saveFile:
            result.to_csv(path2SaveFiles+"heating.csv")
            
        if saveFigure:
            fig.savefig(path2SaveFigures+"DJU_heating.png")
            
    # For cooling days calculation   
    elif not djuBelow:
        # Add the days having Tmax < refValue
        DJ1 = dfDailyExt.mean(axis = 1, level = 0)[dailyMin > refValue].reindex(index2keep)-refValue

        # Add the days having Tmax > refValue but Tmin < refValue
        df_selection = (dailyMin < refValue) & (dailyMax > refValue)
        DJ2 = (dailyMax-refValue).mul(0.08+0.42*(dailyMax-refValue).divide(\
               dailyMax-dailyMin))[df_selection].reindex(index2keep)
        
        # Sum all DJU and group results by years
        DJtot = DJ1.resample("AS").sum().add(DJ2.resample("AS").sum())
    
        # When no data for one year, set values to nan
        df_filter_nan = dailyMin.isna() | dailyMax.isna()
        result = df_set_nan(DJtot, dfDailyExt.mean(axis = 1, level = 0), df_filter_nan, freq = "AS", nb_nan = 2)
        
        # Plot the results
        fig = plt.figure()
        result.boxplot(fontsize = 10, rot = 45)
        
        if saveFile:
            result.to_csv(path2SaveFiles+"cooling.csv")
                
        if saveFigure:
            fig.savefig(path2SaveFigures+"DJU_cooling.png")
            
    return result


def nbHeatWaveDays(dfDailyExt, path2SaveFiles, path2SaveFigures, thresholdDuration = 1, 
                   thresholdNight = 20, thresholdDay = 34, saveFile = True, saveFigure = True):
    """Calculates the number of heat way days. A heat wave day is counted
    under two conditions:
        - daily temperature condition: each of the minimum and maximum temperature
        should be above a minimum ('thresholdNight') and maximum temperature
        ('thresholdDay') thresholds
        - duration condition: a day respecting the previous condition is counted
        as a heatwave day only if at least the 'thresholdDuration' previous
        days also respect the previous condition
    
	Parameters
	_ _ _ _ _ _ _ _ _ _ 

			dfDailyExt : DataFrame
				The temperature data that has to be processed (should be a 
                multiindex dataframe with two levels of columns: the first
                is the name of the station, the second should contain the
                daily "MIN" and "MAX")
            thresholdDuration : integer, default 1
                The number of days before the current day that should have 
                met the daily temperature condition in order to count the 
                current day as a heat wave day
            thresholdNight : float, default 20
                Temperature used as threshold for night-time (minimum temperature)
            thresholdDay : "String", default "10-01"
                Temperature used as threshold for day-time (maximum temperature)
            path2SaveFiles : string
                Name of the URL where to save the resulting DataFrame if 'saveFile' = True
            path2SaveFigures : string
                Name of the URL where to save the resulting Figure if 'saveFigure' = True
            saveFile : boolean, default True
                Whether or not the results should be saved in a file
            saveFigure : boolean, default True
                Whether or not thefigure should be saved

	Returns 
	_ _ _ _ _ _ _ _ _ _ 

			results : pd.DataFrame
                DataFrame containing the number of heatwave day for each year
                and each station (column of the initial data)"""
    # Identify the days respecting the two threshold temperature
    df_selection = (dfDailyExt.xs("MIN", axis = 1, level = 1) > thresholdNight) &\
                    (dfDailyExt.xs("MAX", axis = 1, level = 1) > thresholdDay)
    df_buff = df_selection.copy()
    # Identify consecutive days respecting the two threshold temperature
    for i in range(1, thresholdDuration+1):
        df_buff = df_buff.multiply(df_selection.shift(1))
      
    # Calculates the total number of consecutive heat wave days per year 
    # (setting 0 for years where no data exist...)
    result_without_nan = df_buff.resample("AS").sum()
    
    # Reset nan values that have been lost with the previous analysis
    df_filter_nan = (dfDailyExt.xs("MIN", axis = 1, level = 1).isna()) |\
                (dfDailyExt.xs("MAX", axis = 1, level = 1).isna())
    result = df_set_nan(result_without_nan, df_buff, df_filter_nan, freq = "AS", nb_nan = 2)
    
    # Display figure
    fig = plt.figure()
    result.boxplot(fontsize = 10, rot = 45)
    
    if saveFile:
        result.to_csv(path2SaveFiles+"nb_heat_wave_days.csv")
    
    if saveFigure:
        fig.savefig(path2SaveFigures+"nb_heat_wave_days.csv")
    
    return result

def df_set_nan(result_without_nan, df_initial, df_nan_to_filter, freq = "AS", nb_nan = 2):
    """ The resample function does not get rid of nan neither the potential filtering induced
    by threshold comparison. Thus the results may display a value for certain year / month
    while there is no data for this year / month. Thus this function set the result
    year / month to nan when needed
    
	Parameters
	_ _ _ _ _ _ _ _ _ _ 

			result_without_nan : DataFrame
				The result without any nan while it should contain
            df_initial : DataFrame
                The raw data before the resampling
            df_nan_to_filter : DataFrame
                The raw data with each value replaced by a boolean (True when nan)
            thresholdDay : "String", default "10-01"
                Temperature used as threshold for day-time (maximum temperature)
            freq : pd.offsets, "AS"
                Offsets corresponding to the resampling
            nb_nan : integer, default 2
                The minimum number of nan to consider that a month (or year) should be
                set to nan

	Returns 
	_ _ _ _ _ _ _ _ _ _ 

			results : pd.DataFrame
                DataFrame containing the results with values set to nan"""
    df_initial[df_nan_to_filter] = np.nan
    g = {c: df_initial[c].groupby(pd.Grouper(freq = freq)) for c in df_initial.columns}
    under_two_nan = pd.concat({c: g[c].apply(lambda x: pd.isnull(x).sum()) >= nb_nan
                     for c in g.keys()}, axis = 1)
    result_without_nan[under_two_nan] = np.nan
    
    return result_without_nan