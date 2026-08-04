#!/usr/bin/env bash
set -euo pipefail

repo=/home/amd/project/sam3-tracker-rocm
release=/home/amd/project/npu_iron/releases/sam3-vit-p14-m1536-power-v1
out=$repo/results/validation_p14_power_v1_20260804
python=/home/amd/miniforge3/envs/rocm7p13-sam3/bin/python
preload=/opt/rocm-7.2.0/lib/libmigraphx_c.so.3:/opt/rocm-7.2.0/lib/migraphx/lib/libmigraphx.so.2016000.0

[[ ! -e $out ]] || { echo "refusing to overwrite: $out" >&2; exit 2; }
mkdir -p "$out/logs"

check_dstate(){
  local found
  found=$(ps -eo stat,wchan:32,comm,args | awk '$1 ~ /^D/ && /amdxdna/ {print}')
  [[ -z $found ]] || { printf '%s\n' "$found" >&2; exit 3; }
}

cd "$repo"
check_dstate
bash "$release/scripts/verify_release.sh" >"$out/release_verify.log"
printf 'index\tvideo\tprompts\tstatus\toutput\tframes\tduration_s\tbytes\n' >"$out/status.tsv"

failures=0
run_one(){
  local index=$1 input=$2 stem=$3 prompt_spec=$4
  local output="$out/${index}_${stem}__p14_mask.mp4"
  local log="$out/logs/${index}_${stem}.log"
  local prompts
  IFS='|' read -r -a prompts <<<"$prompt_spec"
  echo "=== [$index] $input prompts=${prompts[*]} ==="
  set +e
  timeout --signal=TERM --kill-after=20s 1200s \
    env LD_PRELOAD="$preload" "$python" demo_npu_parallel.py \
      --video "$input" --text "${prompts[@]}" \
      --checkpoint model/sam3 --onnx-dir onnx_files_504 --imgsz 504 \
      --output "$output" >"$log" 2>&1
  rc=$?
  set -e
  check_dstate
  if [[ $rc == 0 && -s $output ]]; then
    local frames duration bytes
    frames=$(ffprobe -v error -select_streams v:0 -show_entries stream=nb_frames \
      -of default=nw=1:nk=1 "$output" 2>/dev/null || echo unknown)
    duration=$(ffprobe -v error -show_entries format=duration \
      -of default=nw=1:nk=1 "$output" 2>/dev/null || echo unknown)
    bytes=$(stat -c %s "$output")
    printf '%s\t%s\t%s\tPASS\t%s\t%s\t%s\t%s\n' \
      "$index" "$input" "$prompt_spec" "$output" "$frames" "$duration" "$bytes" \
      >>"$out/status.tsv"
    echo "PASS output=$output frames=$frames duration=$duration"
  else
    printf '%s\t%s\t%s\tFAIL(rc=%s)\t%s\t-\t-\t-\n' \
      "$index" "$input" "$prompt_spec" "$rc" "$output" >>"$out/status.tsv"
    echo "FAIL rc=$rc log=$log" >&2
    failures=$((failures+1))
  fi
}

run_one 01 assets/blackswan.mp4 blackswan 'swan'
run_one 02 assets/parkour.mp4 parkour 'people'
run_one 03 assets/sidewalk_running_man.mp4 sidewalk_running_man 'sidewalk|lawn'
run_one 04 assets/sideway_lawn.mp4 sideway_lawn 'sidewalk|lawn'
run_one 05 assets/two_person_dog_lawn.mp4 two_person_dog_lawn 'people|dog|lawn'
run_one 06 assets/pexels_baseball_field_drone.mp4 pexels_baseball_field_drone 'lawn'
run_one 07 assets/gettyimages-2171845186-640_adpp.mp4 gettyimages_dog 'dog'
run_one 08 assets/office_hallway_two_way.mp4 office_hallway_two_way 'floor|wall'

printf '%s\n' \
  'assets/indoor_to_outdoor.mp4 | prompts=floor,wall,sidewalk,lawn | source video unavailable; existing copies already contain old masks/HUD' \
  >"$out/missing_inputs.txt"

cd "$out"
find . -maxdepth 1 -type f -name '*.mp4' -print0 | sort -z | xargs -0 sha256sum >VIDEOS.sha256
cd "$repo"
sha256sum \
  assets/blackswan.mp4 \
  assets/parkour.mp4 \
  assets/sidewalk_running_man.mp4 \
  assets/sideway_lawn.mp4 \
  assets/two_person_dog_lawn.mp4 \
  assets/pexels_baseball_field_drone.mp4 \
  assets/gettyimages-2171845186-640_adpp.mp4 \
  assets/office_hallway_two_way.mp4 \
  >"$out/INPUTS.sha256"
cd "$out"
check_dstate
echo "output_dir=$out"
echo "video_passes=$(awk -F '\t' 'NR>1&&$4=="PASS"{n++} END{print n+0}' status.tsv)"
echo "video_failures=$failures"
[[ $failures == 0 ]]
echo "NPU_VALIDATION_VIDEOS=PASS"
