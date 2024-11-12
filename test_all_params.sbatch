#!/bin/bash

#SBATCH --job-name=HEROPtest
#SBATCH --mem=40G
#SBATCH --time=120:00:00
#SBATCH --mail-type=FAIL
#SBATCH --mail-type=END
#SBATCH --mail-user=michels9@illinois.edu

declare -a mobility_options=("DRIVING" "WALKING" "BICYCLE")
declare -a supply_file=("supply/ContinentalHospitals.shp" "supply/chi_doc_pysal.csv" "supply/chi_hospitals_pysal.csv")

# just use these throughout
export param_max_travel_time="90"
export param_access_measure="ALL"  # this will run all of them no need to loop

for mob in "${mobility_options[@]}"; do
  for sf in "${supply_file[@]}"; do
    export param_mobility_mode=$mob
    export param_supply_filename=$sf
    if  [[ $sf == "supply/ContinentalHospitals.shp" ]]; then
      export param_population_type="ZIP"
      export param_supply_capacity="BEDS"
      export param_supply_latlon_or_id="ID"
      export param_supply_id="ZIP"
      export param_supply_lat=""
      export param_supply_lon=""
    elif [[ $sf == "supply/chi_doc_pysal.csv" ]]; then
      export param_population_type="TRACT"
      export param_supply_capacity="doc"
      export param_supply_latlon_or_id="ID"
      export param_supply_id="geoid"
      export param_supply_lat=""
      export param_supply_lon=""
    elif [[ $sf == "supply/chi_hospitals_pysal.csv" ]]; then
      export param_population_type="TRACT"
      export param_supply_capacity=""  # just count num hospitals
      export param_supply_latlon_or_id="LATLON"  # use the geometry field
      export param_supply_id=""
      export param_supply_lat=""
      export param_supply_lon=""
    else
      echo "You didn't pass in a valid supply name"
    fi
    printf "Running:\n  * MOBILITY_MODE=${param_mobility_mode}\n  * POPULATION_TYPE=${param_population_type}\n  * MAX_TRAVEL_TIME=${param_access_measure}\n  * SUPPLY_FILE=${param_supply_filename}\n\n"
    executable_folder="/job/executable" data_folder="/job/data" result_folder="/job/result" srun --unbuffered --mpi=pmi2 singularity exec --bind /data/keeling/a/michels9/herop-spatial-access-app-compute:/job/executable,/data/keeling/a/michels9/herop-spatial-access-app-compute:/job/result,/data/cigi/scratch/cigi-gisolve/compute_shared:/compute_shared,/data/keeling/a/michels9/common/michels9/herop_access_data:/job/herop_access_data /data/keeling/a/cigi-gisolve/simages/pysal-access.simg bash -c "cd /job/executable && python3 AccessComputeJob.py"
  done
done

