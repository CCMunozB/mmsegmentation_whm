name_t1nii=$1
name_flrnii=$2
name_WMHs=$3
name_label=$4
model_name=$5

python tools_wmh/exec.py ${name_t1nii} ${name_flrnii} tools_wmh/output/${name_WMHs} ${name_label} ${model_name}