import pandas as pd
import os
import hvplot.pandas

ALL_POLLUTANTS = ["pm10","nox","no","no2","pm2.5","pm1","o3"]
DICT_DF_STATIONS_ID = {'d':1,'w':2,'s':3,'n':4,'e':5,'z':6}
DICT_DF_STATIONS_NAME = {'d':"Graz DonBosco",'w':"Graz West",'s':"Graz South","n":'Graz North','e':"Graz East",'z':"Zagreb"}
        
def get_station_name_by_id(id) -> str:
    for key,value in DICT_DF_STATIONS_ID.items():
        if value==id:
            return DICT_DF_STATIONS_NAME[key]
    return "ERROR NOT FOUND!"

def get_station_name_by_indice(indice):
    return DICT_DF_STATIONS_NAME.get(indice,"ERROR NOT FOUND!")

def get_li_station_names_indices(li_stations):
    return str([DICT_DF_STATIONS_NAME.get(x,x) for x in li_stations])

def get_station_indice_by_id(id) -> str:
    for key,value in DICT_DF_STATIONS_ID.items():
        if value==id:
            return key
    return "ERROR NOT FOUND!"

def get_station_id_by_name(name):
    return DICT_DF_STATIONS_ID[name]


def get_station_id_by_name(name):
    return DICT_DF_STATIONS_ID[name]


def read_df(path)-> pd.DataFrame:
    return pd.read_csv(path,
        header=0,
        index_col=0,
        infer_datetime_format=True,
        parse_dates=[0]) #to set index to datetime

def read_all(path,DICT_DF_STATIONS,DICT_DF_STATIONS_ID,use_lags=False)-> pd.DataFrame:
    for file in os.scandir(path=path):
        name = file.name
        key = name.split('.')[0]
        DICT_DF_STATIONS[key] = read_df(path+"/"+name)
        DICT_DF_STATIONS[key]["id"] = DICT_DF_STATIONS_ID[key]

    # drop lag columns
    if not use_lags:
        for station in DICT_DF_STATIONS:
            if "pm10Lag" in DICT_DF_STATIONS[station].columns:
                DICT_DF_STATIONS[station].drop('pm10Lag',axis=1,inplace=True)
                print("DELETE done for station '"+get_station_name_by_indice(station)+"'")
    return DICT_DF_STATIONS



def find_intersection(lists)-> list:
    intersection = set(lists[0])
    for li in lists[1:]:
        intersection = intersection.intersection(li)
    return list(intersection)

def get_colname_intersection(dict_stations, li_station_names):
    li_col_names = []
    for station_name in li_station_names:
        li_col_names.append(list(dict_stations[station_name].keys()))
    return find_intersection(lists=li_col_names)


def create_global_model(dict_stations,li_station_combination)->pd.DataFrame:
     # list of dfs that will be concatenated to one
    li_dfs = []
    # all features (met+poll) that are the same
    dict_sel_stations  = {k: dict_stations[k] for k in li_station_combination}
    li_intersections = get_colname_intersection(dict_sel_stations,li_station_combination)
    for station in li_station_combination:
        li_dfs.append(dict_stations[station][li_intersections])
    return pd.concat(li_dfs,axis=0)
    
    
def split_pollutant_meteorological(df,li_all_pollutants=ALL_POLLUTANTS, li_sel_pollutants=[])-> (pd.DataFrame,pd.DataFrame):
    """
    returns df_features, df_pollutants
    """ 
       
    if li_sel_pollutants == []:
        li_sel_pollutants = li_all_pollutants
        
    df_features = df.loc[:, ~df.columns.isin(li_all_pollutants)].copy()
    df_pollutants = df.loc[:, df.columns.isin(li_sel_pollutants)].copy()
        
    if len(df_pollutants.columns)==0:
        print("Pollutant[s] '"+ str(li_sel_pollutants) +"' not in dataframe")
        sys.exit()
    
    return df_features, df_pollutants

def split_train_test_by_year(df,pollutant_to_predict,split_ratio,scaler=None, output=True) -> (pd.DataFrame,pd.DataFrame, pd.DataFrame,pd.DataFrame, dict):
    """ Calculates training and testing data based on the split_ratio
    Training and tresting data are spliited according to their year (only in whole years)
    returns df_train_X, df_train_y, df_test_X, df_test_y, dict_info
    """
    dict_info = {"train":[], "test":[]}
    li_years =list(set(df.index.year))
    li_years.sort()
    split_index = int(len(li_years)*split_ratio)
    real_split_ratio = split_index/len(li_years)
    if split_index >= len(li_years):
        if output:
            print("Training data = Testing data: whole dataframe")
            print("Training samples: ",len(df))
        df_train_X, df_train_Y = split_pollutant_meteorological(df,li_sel_pollutants=[pollutant_to_predict])
        dict_info["train"] = li_years
        dict_info["test"] =  li_years
        return df_train_X, df_train_Y, df_train_X, df_train_Y, dict_info
    years_training = li_years[:split_index]
    years_testing = li_years[split_index:]
    
    
    df_train = df[df.index<str(li_years[split_index])]
    df_test = df[df.index>=str(li_years[split_index])]
    
    # scale the data if scaling is needed:
    if scaler!=None: 
        # for scaling: only scale final dataframe: all features used + the one used pollutant
        # select all relevant columns (drop pollutants that are not used)
        li_poll_to_drop =  list(set(ALL_POLLUTANTS) - set([pollutant_to_predict]))
        if len(li_poll_to_drop)==0:
            print("SOMETHING WENT WRONG")
            return
        # select all data used 
        df_train = df_train.loc[:, ~df_train.columns.isin(li_poll_to_drop)].copy()
        df_test = df_test.loc[:, ~df_test.columns.isin(li_poll_to_drop)].copy()
        df_train, df_test = scale_data(scaler=scaler,df_train=df_train,df_test=df_test)
    
    real_split_ratio_samples = len(df_train)/(len(df_test) + len(df_train))
    
    df_train_X, df_train_y = split_pollutant_meteorological(df_train,li_sel_pollutants=[pollutant_to_predict])
    df_test_X, df_test_y = split_pollutant_meteorological(df_test,li_sel_pollutants=[pollutant_to_predict])
    
    dict_info["train"] = years_training
    dict_info["test"] =  years_testing
    
    if output:
        print("split_index:\t\t\t" + str(split_index)  + "\n"+
              "real_split_ratio [years]:\t" + str(real_split_ratio) +  "\n"+
              "years_training:\t\t\t" + str(years_training) + "\n"+
              "years_testing:\t\t\t" + str(years_testing) +  "\n"+
              "split_year:\t\t\t"+ str(li_years[split_index]) + "\n"+
              "training samples:\t\t"+ str(len(df_train)) + "\n"+
              "testing samples:\t\t" + str(len(df_test))+ "\n"+
             "real_split_ratio [samples]:{:10.3f}".format(real_split_ratio_samples)) 
    
    return df_train_X, df_train_y, df_test_X, df_test_y, dict_info


def split_train_test_by_station_ratio(df,pollutant_to_predict,li_stations,ratio)->(pd.DataFrame,pd.DataFrame,pd.DataFrame,pd.DataFrame,dict):
    """ Calculates training and testing data based on the split_ratio
    Training and tresting data are spliited according to the station
    returns df_train_X, df_train_y, df_test_X, df_test_y, dict_info
    """
    li_ids = [get_station_id_by_name(id_) for id_ in li_stations]
    
    dict_info = {"train":[],
                 "test":[]}
    
    print("Stations:\t\t\t"+get_li_station_names_indices(li_stations))

    # default
    dict_info["train"]=[get_station_indice_by_id(id_) for id_ in li_ids]
    dict_info["test"]=dict_info["train"]
    
    if len(li_ids)==1: #only one station
        print("ONLY ONE STATION: "+get_station_name_by_id(li_ids[0]))
        df_X, df_y = split_pollutant_meteorological(df=df,li_sel_pollutants=[pollutant_to_predict])
        
        return df_X,df_y,df_X,df_y
    index = int(ratio * len(li_ids))
    if index >= len(li_ids):
        print("Training data = Testing data -> whole DataFrame")
        print("Training samples: ",len(df))
        
        df_X, df_y = split_pollutant_meteorological(df=df,li_sel_pollutants=[pollutant_to_predict])
        return df_X,df_y,df_X,df_y,dict_info
    
    li_train = li_ids[:index]
    li_test = li_ids[index:]
    real_ratio = index/len(li_ids)
    
    df_train = df[df['id'].isin(li_train)]
    df_test = df[df['id'].isin(li_test)]
    
    df_train_X, df_train_y = split_pollutant_meteorological(df=df_train,li_sel_pollutants=[pollutant_to_predict])
    df_test_X, df_test_y = split_pollutant_meteorological(df=df_test,li_sel_pollutants=[pollutant_to_predict])
    
    dict_info["train"]=[get_station_indice_by_id(id_) for id_ in li_train]
    dict_info["test"]=[get_station_indice_by_id(id_) for id_ in li_test]
    
 
    print("split_index:\t\t\t" + str(index)  + "\n"+
          "ratio:\t\t\t\t" + str(ratio) +  "\n"+
          "real_ratio:\t\t\t" + str(real_ratio) +  "\n"+
          "Stations used for training: \t" + str([get_station_name_by_id(idx) for idx in li_train]) + "\n"+
          "Stations used for testing:\t" + str([get_station_name_by_id(idx) for idx in li_test]) + "\n"+
          "Training samples:\t\t"+ str(len(df_train)) + "\n"+
          "Testing samples:\t\t" + str(len(df_test))+ "\n"
          "real_split_ratio [samples]:{:10.3f}".format(len(df_train)/len(df))) 
   
    return df_train_X, df_train_y, df_test_X, df_test_y,dict_info


def split_train_val_test_by_station_fluent_cut(df,tr=0.8,te=0.1,val=0.1,pollutant="pm10")->(pd.DataFrame,pd.DataFrame,pd.DataFrame,pd.DataFrame,pd.DataFrame,pd.DataFrame,dict):
    """
    Splits training data of one station into train/test/val set
    Makes a cut on the given split ratio, does not check whether this happens during a year! Cut could be anywhere
    returns: df_train_X, df_train_y, df_test_X, df_test_y, df_val_X, df_val_y, dict_info
    """
    
    dict_info = {"train":None,
                 "test":None,
                 "val":None}
        
    index_tr = int(tr*len(df))
    index_te = int(te*len(df))
    index_val = int(val*len(df))
    
    if(index_tr+index_te+index_val) != len(df):
        index_tr+=1
    
    index_te = index_te + index_tr
    
    df_train = df[:index_tr]
    df_test = df[index_tr:index_te]
    df_val = df[index_te:]
    
    li_stations = [get_station_name_by_id(station) for station in list(set(df["id"]))]
    print("Station: \t\t\t"+str(li_stations))
    print("Samples used for training:\t"+str(len(df_train)))
    print("Samples used for testing:\t"+str(len(df_test)))
    print("Samples used for validation:\t"+str(len(df_val)))
    
    print("Real training ratio [%]:   {:10.3f}".format(len(df_train)/len(df))+ "\n" +
         "Real testing ratio [%]:    {:10.3f}".format(len(df_test)/len(df))+ "\n" +
         "Real validation ratio [%]: {:10.3f}".format(len(df_val)/len(df))) 
    
    df_train_X,df_train_y = split_pollutant_meteorological(df=df_train,li_sel_pollutants=[pollutant])
    df_test_X,df_test_y = split_pollutant_meteorological(df=df_test,li_sel_pollutants=[pollutant])
    df_val_X,df_val_y = split_pollutant_meteorological(df=df_val,li_sel_pollutants=[pollutant])
    
    dict_info["train"]=len(df_train)
    dict_info["test"]=len(df_test)
    dict_info["val"]=len(df_val)
    
    return df_train_X, df_train_y, df_test_X, df_test_y, df_val_X, df_val_y, dict_info


def split_train_val_test_by_year(dict_all_stations,station,tr=0.8,te=0.1,val=0.1,pollutant="pm10", output=True)->(pd.DataFrame,pd.DataFrame,pd.DataFrame,pd.DataFrame,pd.DataFrame,pd.DataFrame,dict):
    """
    Splits training data of one station into train/test/val set
    returns: df_train_X, df_train_y, df_test_X, df_test_y, df_val_X, df_val_y
    """
    dict_info = {"train":None,
                 "test":None,
                 "val":None}
    
    df = dict_all_stations[station].copy()
    
    li_years = list(set(df.index.year))
    li_years.sort()
    index_tr = int(tr*len(li_years))
    index_te = int(te*len(li_years))
    index_val = int(val*len(li_years))
    
    if(index_tr+index_te+index_val) != len(df):
        index_tr+=1
    
    index_te = index_te + index_tr
    
    li_years_train = li_years[:index_tr]
    li_years_test = li_years[index_tr:index_te]
    li_years_val = li_years[index_te:]
    
    df_train = df[df.index.year.isin(li_years_train)]
    df_test = df[df.index.year.isin(li_years_test)]
    df_val = df[df.index.year.isin(li_years_val)]
    
    if output:
        print("Station: \t\t\t"+get_station_name_by_indice(station))
        print("Years used for training:\t"+str(li_years_train))
        print("Years used for testing:\t\t"+str(li_years_test))
        print("Years used for validation:\t"+str(li_years_val))

        print("Real training ratio [%]:   {:10.3f}".format(len(df_train)/len(df))+ "\n" +
             "Real testing ratio [%]:    {:10.3f}".format(len(df_test)/len(df))+ "\n" +
             "Real validation ratio [%]: {:10.3f}".format(len(df_val)/len(df))) 
    
    df_train_X,df_train_y = split_pollutant_meteorological(df=df_train,li_sel_pollutants=[pollutant])
    df_test_X,df_test_y = split_pollutant_meteorological(df=df_test,li_sel_pollutants=[pollutant])
    df_val_X,df_val_y = split_pollutant_meteorological(df=df_val,li_sel_pollutants=[pollutant])
    
    
    dict_info["train"]=len(df_train)
    dict_info["test"]=len(df_test)
    dict_info["val"]=len(df_val)
    
    return df_train_X, df_train_y, df_test_X, df_test_y, df_val_X, df_val_y, dict_info

def split_train_test_by_station(df,pollutant_to_predict,li_stations,station_test=None, station_validation=None, use_validation = True, output=True)->(pd.DataFrame,pd.DataFrame,pd.DataFrame,pd.DataFrame, dict):
    """
    returns df_train_X, df_train_y, df_test_X, df_test_y, df_val_X, df_val_y, dict_info
    """
    dict_info = {"train":[], "test":[],"val":[]}
    
    # if station_test is none then select last one as test data:
    if station_test==None:
        station_test = li_stations[-2]
    # none just means that the prelast statio is used as validation data
    if station_validation==None and use_validation:
        station_validation = li_stations[-1]
    
    li_stations_train = li_stations.copy() 
    li_stations_train.remove(station_test)
    
    # only if validation is desired do something!
    if use_validation:
        li_stations_train.remove(station_validation)
        
        df_val = df[df['id'] == get_station_id_by_name(station_validation)]
        df_val_X, df_val_y = split_pollutant_meteorological(df=df_val,li_sel_pollutants=[pollutant_to_predict])
    else:
        df_val = []
        df_val_X = None
        df_val_y = None
        station_validation = "-"
    
    if len(li_stations_train)==0:
        li_stations_train = li_stations
        print("Single data set detected. No split possible. Calling 'split_train_test_by_year'")
        return split_train_test_by_year(df,pollutant_to_predict,0.8)
    
    df_train =  df[df['id'].isin([get_station_id_by_name(elem) for elem in  li_stations_train])] 
    df_train_X, df_train_y = split_pollutant_meteorological(df=df_train,li_sel_pollutants=[pollutant_to_predict])

    df_test = df[df['id'] == get_station_id_by_name(station_test)]
    df_test_X, df_test_y = split_pollutant_meteorological(df=df_test,li_sel_pollutants=[pollutant_to_predict])
    
    if output:
        print("Stations:\t\t\t"+get_li_station_names_indices(li_stations)+ "\n"+
              "Stations used for training: \t" + str([get_station_name_by_indice(idx) for idx in li_stations_train]) + "\n"+
              "Station used for testing:\t" + str(get_station_name_by_indice(station_test)) + "\n"+
              "Station used for validation:\t" + str(get_station_name_by_indice(station_validation)) + "\n"+
              "Training samples:\t\t"+ str(len(df_train)) + "\n"+
              "Testing samples:\t\t" + str(len(df_test)) +"\n" +
              "Validation samples:\t\t"+ str(len(df_val)))
    
    dict_info["train"] = li_stations_train.copy()
    dict_info["test"] = station_test
    dict_info["test"] = station_test
    
    return df_train_X, df_train_y, df_test_X, df_test_y, df_val_X, df_val_y, dict_info
    
def create_title(dict_info, test_text=" Test: ")->str:
    train = get_li_station_names_indices(dict_info["train"])
    test = get_li_station_names_indices(dict_info["test"])
    return "Train: "+str(train)+test_text+str(test)

def create_prediction_plot(df, period, is_sub=False, plot_kind="line", add_dots=False, title="", ylabel="\u03BCg/m³"):
    """ Plotss the predictions
    add_dots -> to add dots to the line 
    """
    plot = None
    if period is not None:
        title_pre="Daily mean prediction: "
        if period=="M":
            title_pre="Monthly mean prediction: "
        elif period=="W":
            title_pre="Weekly mean prediction: "
        plot = df.resample(period).mean().hvplot(use_index=True, value_label=' ',width=1200, subplots=is_sub, kind=plot_kind,title=title_pre+title, ylabel=ylabel,xlabel="Date")
    else: 
        plot= df.hvplot(use_index=True, value_label=' ',width=1200, subplots=is_sub, kind=plot_kind, title=title)
    if add_dots:
        if not plot_kind == "scatter":
            return plot * create_prediction_plot(df=df, period=period, is_sub=False, plot_kind="scatter", add_dots=False, title=title)
    return plot


def use_traffic_data(dict_stations,use_traffic_continous = False, use_traffic_bins = False):
    if not use_traffic_continous:
        for station in dict_stations:
            if "traffic" in dict_stations[station].columns:
                dict_stations[station].drop('traffic',axis=1,inplace=True)
                print("DELETE [traffic] done for station '"+get_station_name_by_indice(station)+"'")
    if not use_traffic_bins:
        for station in dict_stations:
            if "trafficClass" in dict_stations[station].columns:
                dict_stations[station].drop('trafficClass',axis=1,inplace=True)
                print("DELETE [trafficClass] done for station '"+get_station_name_by_indice(station)+"'")
    return dict_stations

def use_traffic_lags(dict_stations, use_traffic_lags = True, only_use_lags = False):
    if only_use_lags:
        for station in dict_stations:
            if "traffic" in dict_stations[station].columns:
                dict_stations[station].drop('traffic',axis=1,inplace=True)
                print("DELETE [traffic] done for station '"+get_station_name_by_indice(station)+"'")
    # drop lag columns
    if not use_traffic_lags:
        for station in dict_stations:
            li_lags = ["trafficLag1","trafficLag2","trafficLag3","trafficLag4"]
            for traffic_lag in li_lags:
                if traffic_lag in dict_stations[station].columns:
                    dict_stations[station].drop(traffic_lag ,axis=1,inplace=True)
                    print("DELETE"+ traffic_lag + "done for station '"+get_station_name_by_indice(station)+"'")
    return dict_stations

def calc_exceeding_days(df_y,predictions, max_pm10=50):
    
    df_result = pd.DataFrame()
    df_result = df_y.copy()
    df_result["pm10_pred"] = predictions
    
    # first and second column select without col name
    li_colnames = df_result.columns
    
    li_years = list(set(df_y.index.year))
    li_years.sort()
    
    li_pm10_exceeding = [] 
    li_pm10_exceeding_pred = []
    
    for year in li_years:
        li_pm10_exceeding.append(len(df_result[(df_result.index.year == year) & (df_result[li_colnames[0]] > max_pm10)]))
        li_pm10_exceeding_pred.append(len(df_result[(df_result.index.year == year) & (df_result[li_colnames[1]] > max_pm10)]))
    
    
    data = {'pm10_exc': li_pm10_exceeding, 'pm10_exc_pre': li_pm10_exceeding_pred}

    df = pd.DataFrame(data, index=li_years)
    
    return df