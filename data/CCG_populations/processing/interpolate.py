import re
import pandas as pd
import os
import numpy as np

folder = os.path.dirname(os.path.dirname(os.path.realpath(__file__))) # folder where data is saved
filepath = os.path.join(folder, "london_females_2002-18.csv")

annual_df = pd.read_csv(filepath)
annual_df.year = annual_df.year.astype(str) + ["-06-30"]*len(annual_df) # Mid-year populations
annual_df.year = pd.to_datetime(annual_df.year, format="%Y-%m-%d")
annual_df.set_index("year", inplace=True)
# annual_df.set_index(["year", "area_code", "ccg", "age_group"], inplace=True)
# annual_df.sort_index(level="year", sort_remaining=True, inplace=True)
print(annual_df.columns)
#print(annual_df.index)

#print(annual_df.population)

ccg = annual_df.ccg.unique()[0]
age_group = annual_df.age_group.unique()[0]

full_df = pd.DataFrame()

for ccg in annual_df.ccg.unique():
    monthly_df = pd.DataFrame()
    for age_group in annual_df.age_group.unique():
        area_code = annual_df.loc[annual_df.ccg == ccg, "area_code"].unique()[0]
        print(area_code, ccg, age_group)
        if monthly_df.empty:
            monthly_df = annual_df.loc[(annual_df.ccg == ccg) & (annual_df.age_group == age_group), "population"].resample("M").asfreq()
            monthly_df = monthly_df.interpolate(method="linear")
            monthly_df = pd.DataFrame(monthly_df.round(0).astype(int))

            monthly_df["ccg_code"] = [area_code]*len(monthly_df)
            monthly_df["ccg_name"] = [ccg]*len(monthly_df)
            monthly_df[age_group] = monthly_df.population
            monthly_df.drop(columns="population", inplace=True)
            monthly_df = monthly_df.reset_index().rename(columns={"year": "date"}).set_index("date")
        else:
            monthly_df[age_group] = annual_df.loc[(annual_df.ccg == ccg) & (annual_df.age_group == age_group), "population"].resample("M").asfreq().interpolate(method="linear").round(0).astype(int)
        # print(monthly_df)
    if full_df.empty:
        full_df = monthly_df
    else:
        full_df = full_df.append(monthly_df)
    # print(full_df)

print(full_df)

save_filename = "london_female_pop_monthly_2002-06_2018-06.csv"
full_df.to_csv(os.path.join(folder, save_filename))

print("Saved to csv.")
