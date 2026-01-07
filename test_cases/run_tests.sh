#!/bin/bash
GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m' # No Color

printred () {
    echo -e "${RED}$1${NC}"
}

# source the global test variables
source TESTENV.env
cd $WORKINGDIR
mkdir -p ./exec
cp ../AccessComputeJob.py ./exec/

docker pull $DOCKERIMAGE

# collect the test cases
tcases=(case*/)
cnum=0
for tcase in "${tcases[@]}"; do
    echo "Executing case ${cnum}/${#tcases[@]}: ${tcase}..."
    echo $param_mobility_mode
    case_result_folder="${result_folder}${tcase}"
    mkdir -p $case_result_folder
    chmod -R 777 $case_result_folder  # makes the directory writable to everyone.
    # cleanup case result folder before we add to it
    rm ./$case_result_folder*
    # echo $case_result_folder

    # volume mounts are designed to mimic CyberGIS-Compute
    docker run --env result_folder="/job/result/" --read-only -v ./exec:/job/executable --read-only -v ./$tcase:/job/data -v ./$case_result_folder:/job/result/:rw --read-only -v $herop_data_folder:/job/herop_access_data -w /job/executable $DOCKERIMAGE bash -c "source /job/data/params.env && python3 AccessComputeJob.py" > $case_result_folder/out.log

    # check for the output CSV
    ## TODO will need to check for either CSV or gpkg later!
    expected_output="${case_result_folder}/access_result.csv"
    if [ ! -f $expected_output ]; then
        printred "${cnum}/${#tcases[@]} (${tcase}): Output not found"
    fi

    (( cnum++ ))
done