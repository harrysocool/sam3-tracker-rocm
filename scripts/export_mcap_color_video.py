#!/usr/bin/env python3
"""Export a timestamp-sampled raw color MP4 from a ROS2 MCAP bag."""

import argparse
import subprocess
from pathlib import Path

import numpy as np
from mcap.reader import make_reader
from mcap_ros2.decoder import DecoderFactory


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mcap", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--topic", default="/sensors/camera_0/color/image")
    p.add_argument("--fps", type=float, default=30.0)
    p.add_argument("--frames", type=int, default=1330)
    args = p.parse_args()
    if args.output.exists():
        raise RuntimeError(f"refusing to overwrite: {args.output}")

    proc = None
    count = 0
    next_time_ns = None
    period_ns = int(round(1_000_000_000 / args.fps))
    with args.mcap.open("rb") as f:
        reader = make_reader(f, decoder_factories=[DecoderFactory()])
        for _schema, _channel, message, ros_msg in reader.iter_decoded_messages(
            topics=[args.topic]
        ):
            stamp = message.log_time
            if next_time_ns is not None and stamp < next_time_ns:
                continue
            h, w = ros_msg.height, ros_msg.width
            row_bytes = int(ros_msg.step)
            raw = np.frombuffer(ros_msg.data, dtype=np.uint8).reshape(h, row_bytes)
            pixels = raw[:, : w * 3].reshape(h, w, 3)
            if ros_msg.encoding == "rgb8":
                frame = pixels[:, :, ::-1]
            elif ros_msg.encoding == "bgr8":
                frame = pixels
            else:
                raise RuntimeError(f"unsupported encoding: {ros_msg.encoding}")
            if proc is None:
                args.output.parent.mkdir(parents=True, exist_ok=True)
                proc = subprocess.Popen(
                    [
                        "ffmpeg", "-loglevel", "error", "-y",
                        "-f", "rawvideo", "-pix_fmt", "bgr24",
                        "-s", f"{w}x{h}", "-r", str(args.fps), "-i", "pipe:0",
                        "-c:v", "libx264", "-crf", "18", "-preset", "fast",
                        str(args.output),
                    ],
                    stdin=subprocess.PIPE,
                )
                next_time_ns = stamp
            proc.stdin.write(np.ascontiguousarray(frame).tobytes())
            count += 1
            next_time_ns += period_ns
            if count >= args.frames:
                break

    if proc is None:
        raise RuntimeError("no matching image messages")
    proc.stdin.close()
    rc = proc.wait()
    if rc != 0 or count != args.frames:
        raise RuntimeError(f"ffmpeg rc={rc}, exported={count}, expected={args.frames}")
    print(f"output={args.output} frames={count} fps={args.fps}")
    print("MCAP_COLOR_EXPORT=PASS")


if __name__ == "__main__":
    main()
