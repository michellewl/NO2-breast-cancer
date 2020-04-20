# Code provided by Geoff Ma

import requests
import json
import pandas as pd

# The key api thing you'll need is this. Conveniently, it throws data in CSV format, which pandas can deal with directly
# 
# 
# `/Data/Site/Wide/SiteCode={SiteCode}/StartDate={StartDate}/EndDate={EndDate}/csv	`
# 
# This returns raw data based on 'SiteCode', 'StartDate', 'EndDate'. Default time period is 'hourly'. Data returned in CSV format
# 

# ### Get List of sites
# 
# However, you're probably wondering where to get the site codes for the above.

##first things first; let's get our sites list:

london_sites = requests.get("http://api.erg.kcl.ac.uk/AirQuality/Information/MonitoringSites/GroupName=London/Json")

london_sites_df = pd.DataFrame( london_sites.json()['Sites']['Site'] )

##double check our sites are good:
london_sites_df.head(5)

##for now, let's select current sites that haven't got a dateclosed at all.....
london_sites_active_df = london_sites_df[london_sites_df['@DateClosed'] == ""]

all_sites_active_list = london_sites_active_df['@SiteCode'].to_list()


# **Warning: Many of these site codes are not currently active - you should make sure that @dateClosed is not current in your date range**


# ### Example to pull data from a single site

# pandas is clever enough to do this all for you! Can you do this in R?
pd.read_csv('http://api.erg.kcl.ac.uk/AirQuality/Data/Site/Wide/SiteCode=BG1/StartDate=2019-01-01/EndDate=2019-01-02/csv').head(5)

# **Now let's say you want NO2 data from all the sites**
# 
# Luckily, there's also an option to query sites by species code/.
# I haven't verified that this will get data from every single station.

no2_df = pd.DataFrame()

for site in all_sites_active_list:
    cur_df = pd.read_csv('http://api.erg.kcl.ac.uk/AirQuality/Data/SiteSpecies/SiteCode=%s/SpeciesCode=NO2/StartDate=2019-01-01/EndDate=2019-01-02/csv'%site)
    cur_df.index = cur_df['MeasurementDateGMT'] 
    cur_df = cur_df.drop(columns =['MeasurementDateGMT']  )

    try:
        no2_df = no2_df.join(cur_df, how = 'outer')
    except KeyboardInterrupt:
        break
    except:
        print(site)


import matplotlib.pyplot as plt
plt.figure(figsize = (10,5))
for col in no2_df.columns.to_list():
    plt.plot(no2_df.index, no2_df[col] )
plt.show()


# **Performance remarks**
# This scripe has a 'for loop', querying one site at a time. It would be quicker to do a parallel query - I've done similar things in the past with 'grequest'
