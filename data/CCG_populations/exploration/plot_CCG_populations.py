import pandas as pd
import os
import seaborn as sns
sns.set(style="darkgrid")

print("Saving plots to:")

folder = os.path.dirname(os.path.dirname(os.path.realpath(__file__))) # folder where data is saved
filename = [file for file in os.listdir(folder) if file == "london_all_years.csv"][0]

df = pd.read_csv(os.path.join(folder, filename))

age_groupings = df.age_group.unique().tolist()

for group in age_groupings:
    plot_df = df.loc[df.age_group == group]

    g = sns.relplot(x="year", y="population", hue="ccg", kind="line", data=plot_df, legend=False, height=5, aspect=3)
    group_name = group.replace("_", " ")
    g.fig.suptitle(f"Female population ({group_name}) of London CCGs")
    # plt.show()

    plot_filename = os.path.join(os.path.dirname(os.path.realpath(__file__)), f"CCG_fpopulation_{group}.png")
    g.savefig(plot_filename)
    print(plot_filename)