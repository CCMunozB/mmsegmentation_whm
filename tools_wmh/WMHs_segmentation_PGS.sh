name_t1nii=$1
name_flrnii=$2
name_WMHs=$3

name_t1=${name_t1nii%".nii.gz"}
name_flr=${name_flrnii%".nii.gz"}

sh tools_wmh/runROBEX.sh /home/electroscian/ownCloud/Competencia/Datos_ROBEX/T1/${name_t1nii} tools_wmh/output/${name_t1}_rbx.nii tools_wmh/output/${name_t1}_rbx_mask.nii  
python tools_wmh/WMHs_segmentation_PGS.py tools_wmh/output/${name_t1}_rbx.nii tools_wmh/output/${name_t1}_rbx_mask.nii /home/electroscian/ownCloud/Competencia/Datos_ROBEX/FLAIR/${name_flrnii} tools_wmh/output/PGS/${name_WMHs}