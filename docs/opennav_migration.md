# OpenNav migration contract

Reference upstream snapshot:
`open-navigation/opennav_amd_semantic_navigation@5e82f4a458f9e97ddd92e2cf10feb2c3552295d3`.

## Runtime and model setup

OpenNav should use the same flow as this repository:

1. Download the pinned precompiled MIGraphX tar and ORT wheel.
2. Assemble the ROCm 7.14 Docker environment without compiling runtime sources.
3. Obtain the SAM3 checkpoint separately.
4. Run the checked-in 504px model export/compile pipeline locally.

Do not restore the upstream package's ROCm 7.2 / MIGraphX 2.16 host fallback.
The initial port uses `SAM3Live`, full detection on every consumed frame,
`bootstrap_frames=0`, and the generated fixed decoder.

## ROS integration rules

- Vendor the complete `tracker` package below `opennav_sam3_inference`; its
  internal imports are package-relative.
- Use continuous capture, owned latest-frame capacity one, and one ordered
  inference owner. Do not add N+1 backbone/preprocessing work.
- Preserve the source image header and keep exposure time separate from host
  arrival/completion time. Query TF at exposure time.
- Close and join the old pipeline before prompt or tracking reset. Never publish
  a late result using a newer class map.
- Tracker-only output is not valid negative/free-space evidence. Do not enable
  hybrid navigation until this provenance reaches the costmap consumer.

After porting, validate a real ROS build/launch, RGB/depth/TF rosbag replay,
prompt changes during inference, reset/shutdown, missing TF, and stale-result
rejection. The local video smoke does not replace these ROS tests.
