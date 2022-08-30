
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.holtwinters import SimpleExpSmoothing, ExponentialSmoothing
import pandas as pd
import cus_func as cf
import numpy as np
import matplotlib.pyplot as plt


# df = pd.read_excel(r'D:\V\项目文件\202203 DP_FC\data\按月数据_历史5年sell in_已处理过_60701246.xlsx')
df_07 = pd.read_csv(r'D:\V\项目文件\202203 DP_FC\data\sku_region__WEEK_202207_alltime.csv')

df_07['DN_date'] = pd.DatetimeIndex(df_07['DN_date'], dtype="datetime64[ns]", freq=None)
df_07['New_Material_Number'] = df_07['New_Material_Number'].astype(str)
print(df_07.head())

df_sku = pd.read_excel(r'D:\V\项目文件\202203 DP_FC\data\主数据.xlsx')

keylist1 = ['New_Material_Number','Region','Cost_center']
keylist2 = ['New_Material_Number','Region']
keylist3 = ['New_Material_Number']
keylist4 = ['Pack','Region']
keylist5 = ['Pack','Cost_Center']

df_07['key1'] = ''
df_07['key1'] = df_07['key1'].map(str).str.cat(df_07[keylist2], sep='', na_rep='')
class_list = list(df_07['key1'].unique())

list = []
list_re = []

model = 'additive'
seasonal = 'add'

df = df_07[df_07['DN_date'] < '20220401' ]
df_47 = df_07[df_07['DN_date'] >= '20220401' ]

for class_i in class_list:

    df_i = df[df['key1'] == class_i]
    df_i = df_i.reset_index()

    Material_Number = df_i['New_Material_Number'][0]
    # Cost_Center = df_i['Cost_center'][0]
    Region = df_i['Region'][0]

    share_info = df_i.set_index(df_i['DN_date'])
    data = share_info['QTY(DN)']

    if data.count() >= 52:

        decompose_result = seasonal_decompose(data, model=model, period=52, extrapolate_trend='freq')

        tr = decompose_result.trend
        df_temp = pd.DataFrame(tr)
        df_temp['seasonal'] = decompose_result.seasonal
        df_temp['resid'] = decompose_result.resid
        df_temp['observed'] = decompose_result.observed
        df_temp['New_Material_Number'] = Material_Number
        # df_temp['costcenter'] = Cost_Center
        df_temp['Region'] = Region
        df_temp['model'] = model
        list.append(df_temp)

        # fig = decompose_result.plot()
        # filename = class_i + '_additive'
        # path = 'D:\\V\\项目文件\\202203 DP_FC\\data\\holtwinter_pic\\'  + filename
        # fig.savefig( path )

        fit1 = ExponentialSmoothing(data, seasonal_periods=52, trend='add', seasonal=seasonal).fit()

        his_fit = fit1.fittedvalues
        pred = fit1.forecast(18)
        his_fit = his_fit.reset_index()
        pred = pred.reset_index()
        his_fit.columns = ['DN_date', 'QTY(DN)']
        pred.columns = ['DN_date', 'QTY(DN)']
        df_re_temp = pd.concat([his_fit, pred])
        df_re_temp['New_Material_Number'] = Material_Number
        df_re_temp['Region'] = Region
        df_re_temp['model'] = seasonal
        list_re.append(df_re_temp)

        del df_temp, tr, decompose_result, df_re_temp

df_tr = pd.concat(list,axis=0)
df_re = pd.concat(list_re,axis=0)

df_re['key2'] = ''
df_re['DN_date'] = pd.to_datetime(df_re['DN_date'])
df_re['DN_date'] = df_re['DN_date'].astype(str)
# df_re['key2'] = df_re['DN_date'] + df_re['Material_Number']
df_re['key2'] = df_re['DN_date'] + df_re['New_Material_Number'] + df_re['Region']

df_07['key2'] = ''
df_07['DN_date'] = pd.to_datetime(df_07['DN_date'])
df_07['DN_date'] = df_07['DN_date'].astype(str)
df_07['key2'] = df_07['DN_date'] + df_07['New_Material_Number'] + df_07['Region']

df_re_mer = pd.merge(df_re,df_07 ,how = 'outer', left_on = 'key2', right_on = 'key2')
df_re_mer['key'] = ''
df_re_mer['key'] = df_re_mer['DN_date_x'] + df_re_mer['New_Material_Number_x'] + df_re_mer['Region_x']
del df_re_mer['Unnamed: 0'],df_re_mer['DN_date_y'],\
    df_re_mer['New_Material_Number_y'], df_re_mer['Region_y']

df_re_mer.rename(columns={'New_Material_Number_x': 'New_Material_Number',
                           'DN_date_x': 'DN_date',
                          'Region_x': 'Region',
                          'QTY(DN)_y ': 'Actual',
                          'QTY(DN)_X': 'Fit/Predict',}, inplace=True)

df_sku['New_Material_Number'] = df_sku['New_Material_Number'].astype(str)
cf.map_one (df_sku, df_re_mer, 'New_Material_Number', 'Pack')

df_re_mer.to_excel(r'D:\V\项目文件\202203 DP_FC\data\fsct_es_sku_region_WEEK_0819.xlsx')

#del df_re_mer['Unnamed:0'], df_re_mer['DN_date_y']

df_tr.to_excel(r'D:\V\项目文件\202203 DP_FC\data\fsct_es_sku_region_WEEK_holtwinter分解_0819.xlsx')

df_tr['key2'] = ''
df_tr['DN_date'] = pd.to_datetime(df_tr['DN_date'])
df_tr['DN_date'] = df_tr['DN_date'].astype(str)
df_tr['key2'] = df['DN_date'] + df['New_Material_Number'] + df['Region']
df_all = pd.merge(df_re_mer,df_tr ,how = 'outer', left_on = 'key2', right_on = 'key2')


# fig = decompose_result.plot()
# fig.show()

