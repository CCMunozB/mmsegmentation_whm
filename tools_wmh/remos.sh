name_t1nii=$1
name_flrnii=$2
name_WMHs=$3

name_t1=${name_t1nii%.nii}
name_flr=${name_flrnii%.nii}

sh tools_wmh/runROBEX.sh tools_wmh/input/${name_t1} tools_wmh/output/${name_t1}_rbx.nii tools_wmh/output/${name_t1}_rbx_mask.nii  
python tools_wmh/exec.py tools_wmh/output/${name_t1}_rbx.nii tools_wmh/output/${name_t1}_rbx_mask.nii tools_wmh/input/${name_flrnii} tools_wmh/output/${name_WMHs}