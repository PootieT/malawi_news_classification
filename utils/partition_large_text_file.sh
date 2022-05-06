
#
# Usage: ./split_file_process_template.sh $lg_pair $num_partition
# split the language pair .train data into multiple equal size partitions
# for distributed job submission in clusters

set -e

input_path=$1  # input language pair
num_partition=$2  # number of equal size partitions to create

total_lc=$(wc -l $input_path | cut -d ' ' -f1)
lc_per_split=$(( $total_lc / $num_partition + 1 ))
echo "total line count="$total_lc", lc_per_split="$lc_per_split

split -l $lc_per_split -d $input_path $input_path
