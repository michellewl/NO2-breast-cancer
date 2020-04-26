### Batch downloading
Batch download is necessary due to KCL API server error if too much data is called at once. This batch download script divides the 230 LAQN sites into 10 equally sized batches. At the moment, the batch number needs to be manually changed and the script re-run, but this process could be automated using a shell script in future. The script is currently set up for NO2 data only but the species can be specified. 

### Troubleshooting
Any errors in joining each site to a batch dataframe are usually because the timestamp column of a particular site is not formatted the same as the rest of the sites. The troubleshooting script fixes this. Site codes should be input manually (the batch download script prints a list of problem sites). A new batch csv file is produced containing the fixed "problem sites".

### Joining batches
This script joins all downloaded batches together and outputs a csv file with all LAQN sites. Any sites with no data are dropped.

### Meta data
This downloads the London site info from KCL using the API. This meta data is called directly using the KCL API when required in other scripts, but a downloaded file can be useful for inspection and takes up little space.
