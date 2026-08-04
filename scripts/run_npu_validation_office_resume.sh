#!/usr/bin/env bash
set -euo pipefail

repo=/home/amd/project/sam3-tracker-rocm
release=/home/amd/project/npu_iron/releases/sam3-vit-p14-m1536-power-v1
out=$repo/results/validation_p14_power_v1_20260804
python=/home/amd/miniforge3/envs/rocm7p13-sam3/bin/python
preload=/opt/rocm-7.2.0/lib/libmigraphx_c.so.3:/opt/rocm-7.2.0/lib/migraphx/lib/libmigraphx.so.2016000.0
input=assets/office_hallway_two_way.mp4
output=$out/08_office_hallway_two_way__p14_mask.mp4
log=$out/logs/08_office_hallway_two_way.log

check_dstate(){ local f; f=$(ps -eo stat,wchan:32,comm,args | awk '$1 ~ /^D/ && /amdxdna/ {print}'); [[ -z $f ]] || { printf '%s\n' "$f" >&2; exit 3; }; }
[[ -d $out ]]
[[ $(awk -F '\t' 'NR>1&&$4=="PASS"{n++}END{print n+0}' "$out/status.tsv") == 7 ]]
[[ -s $repo/$input ]]
[[ ! -e $output ]] || { echo "refusing to overwrite: $output" >&2; exit 2; }
check_dstate
bash "$release/scripts/verify_release.sh" >"$out/release_verify_office.log"
cd "$repo"
set +e
timeout --signal=TERM --kill-after=20s 1200s \
  env LD_PRELOAD="$preload" "$python" demo_npu_parallel.py \
    --video "$input" --text floor wall \
    --checkpoint model/sam3 --onnx-dir onnx_files_504 --imgsz 504 \
    --output "$output" >"$log" 2>&1
rc=$?
set -e
check_dstate
[[ $rc == 0 && -s $output ]] || { echo "office validation failed rc=$rc log=$log" >&2; exit "${rc:-1}"; }
frames=$(ffprobe -v error -select_streams v:0 -show_entries stream=nb_frames -of default=nw=1:nk=1 "$output")
duration=$(ffprobe -v error -show_entries format=duration -of default=nw=1:nk=1 "$output")
bytes=$(stat -c %s "$output")
printf '%s\t%s\t%s\tPASS\t%s\t%s\t%s\t%s\n' \
  08 "$input" 'floor|wall' "$output" "$frames" "$duration" "$bytes" >>"$out/status.tsv"
printf '%s\n' \
  'assets/indoor_to_outdoor.mp4 | prompts=floor,wall,sidewalk,lawn | source video unavailable; existing copies already contain old masks/HUD' \
  >"$out/missing_inputs.txt"
cd "$out"
find . -maxdepth 1 -type f -name '*.mp4' -print0 | sort -z | xargs -0 sha256sum >VIDEOS.sha256
cd "$repo"
sha256sum \
  assets/blackswan.mp4 assets/parkour.mp4 assets/sidewalk_running_man.mp4 \
  assets/sideway_lawn.mp4 assets/two_person_dog_lawn.mp4 \
  assets/pexels_baseball_field_drone.mp4 assets/gettyimages-2171845186-640_adpp.mp4 \
  assets/office_hallway_two_way.mp4 >"$out/INPUTS.sha256"
echo "office_output=$output frames=$frames duration=$duration bytes=$bytes"
echo "video_passes=$(awk -F '\t' 'NR>1&&$4=="PASS"{n++}END{print n+0}' "$out/status.tsv")"
echo "NPU_VALIDATION_OFFICE_RESUME=PASS"
