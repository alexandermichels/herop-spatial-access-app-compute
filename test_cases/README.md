# Test Cases

This directory holds test cases in the form of directories labeled "case<x>". The goal is to quickly test the app by running through each test case.

## Case Directory Structure

Each test case must have a parameters.env file that will be loaded by the Docker container and corresponding data.

## False Alarms

You can safely ignore the following errors/warnings (generally):

```
ERROR 1: PROJ: proj_create_from_database: Open of /opt/conda/share/proj failed
```

```
/opt/conda/lib/python3.9/site-packages/access/fca.py:161: UserWarning: some tracts may be unaccounted for in supply_cost
  warnings.warn("some tracts may be unaccounted for in supply_cost", stacklevel=1)
```

## TODO

-[] Allow for more complex test cases, specified by case.
-[] Singularity/Apptainer version.