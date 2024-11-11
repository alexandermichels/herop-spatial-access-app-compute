# HEROP Spatial Accessibility App through CyberGIS-Compute

Check out the lab: https://healthyregions.org/

Compute plug: https://cybergis.github.io/cybergis-compute-python-sdk/


## Running the Model with Compute

The [UsingTheModel.ipynb](UsingTheModel.ipynb) should work on any of the CyberGIS-Compute supported science gateways including [CyberGISX Hub](https://cybergisxhub.cigi.illinois.edu/). 



## Running the Model Directly

You should run the model on Keeling, but you can do it anywhere if you have the correct data and change the paths. Instructions for running Jupyter on Keeling are here: https://github.com/mgrover1/keeling-crash-course

For the full effect, I recommend running the model using the `test_all_params.sbatch` script which mimics how CyberGIS-Compute works by running within a Singularity container and passing everything as an environmental variable. 

To convert changes made in the Jupyter to a script use `jupyter nbconvert --to script` in the terminal. Then you need to make a couple changes:

* Comment out any lines setting environmental variables to ensure that the parameter sweep works. E.g.:

```
os.environ["result_folder"] = "result"
os.environ['param_mobility_mode'] = "BICYCLE"
os.environ['param_population_type'] = "ZIP"
os.environ['param_max_travel_time'] = "30"
os.environ['param_access_measure'] = "ALL"
os.environ['param_supply_filename'] = "supply/ContinentalHospitals.shp"
os.environ['param_supply_capacity'] = "BEDS"
os.environ['param_supply_latlon_or_id'] = "ID"
os.environ['param_supply_lat'] = ""
os.environ['param_supply_lon'] = ""
os.environ['param_supply_id'] = "ZIP"
```

* The `HEROP_DATA_DIR` path depends on if you're running locally on Keeling or thorugh Compute on keeling:
```
# data folder depends on running in container vs. directly on Keeling
HEROP_DATA_DIR = "/data/keeling/a/michels9/common/michels9/herop_access_data"  # directly on keeling
# HEROP_DATA_DIR = "/job/herop_access_data"  # path we map that directory to in the container
```
* Comment out any line using matplotlib/folium as those aren't in the container (but I can add them lmk)
* If there are particular parameters/data you want to use for our sanity check `test_all_params.sbatch` file it is just a simple script looping through options so you can add your stuff to the arrays we loop through.