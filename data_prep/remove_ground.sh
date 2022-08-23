inter_dir_2018="$1/*[0-9].pcd"
inter_dir_2016="$2/*[0-9].pcd"

echo "Starting to remove ground for 2016"
for file in ${inter_dir_2016};
do
    file=${file%.*}
    echo $file
    pcd_file="${file}.pcd"
    las_file="${file}.las"
    pdal translate $pcd_file $las_file
    pdal translate $las_file $las_file --json remove_ground.json
    pdal translate $las_file $pcd_file
    rm $las_file
done
echo "Finished for 2016"

echo "Starting to remove ground for 2018"
for file in $inter_dir_2018;
do  
    file=${file%.*}
    echo $file
    pcd_file="${file}.pcd"
    las_file="${file}.las"
    pdal translate $pcd_file $las_file
    pdal translate $las_file $las_file --json remove_ground.json
    pdal translate $las_file $pcd_file
    rm $las_file
done
echo "Finished for 2018"