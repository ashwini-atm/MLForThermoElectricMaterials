#!/bin/bash

#DATASET=df_300_magpie_drop_vif_std.csv TEMPERATURE=300 PROPERTY='seebeck_coefficient' MRL=mrl.csv ESTIMATOR=random_forest python script_RF_GA_HT.py

for property in "seebeck_coefficient" "electrical_conductivity" "thermal_conductivity" "power_factor" "ZT"
do
	for temp in 300 400 700
	do
		for data in "magpie_drop_vif_ff_std.csv" "magpie_drop_vif_std.csv" "magpie_ff.csv"
		do
			echo "Doing for dataset: ""df_"$temp"_"$data" temperature: "$temp" property: "$property
			DATASET="df_"$temp"_"$data TEMPERATURE=$temp PROPERTY=$property MRL=mrl.csv ESTIMATOR=random_forest python script_RF_GA_HT.py
		done
	done
done

