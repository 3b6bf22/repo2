N_list=(9 17 21 33 65 129)
for i in "${!N_list[@]}"; do
    N=${N_list[$i]}
    echo "$N $i"
done