# Asset video → canonical text prompts for delivery testing
# Source: git history, test matrix logs, customer run logs (2026-06-24)
#
# Format per entry:
#   video: path relative to project root
#   prompts: list of text prompts (multi-prompt = separate SAM3 detection classes)
#
# Run with demo_live.py:
#   python demo_live.py --video <video> --text <prompts...> --imgsz 504 --mig --onnx-dir onnx_files_504

VIDEO_PROMPTS = [
    {
        "video": "assets/blackswan.mp4",
        "prompts": ["swan"],
        "notes": "Single thing-class, ideal tracker test",
    },
    {
        "video": "assets/parkour.mp4",
        "prompts": ["people"],
        "notes": "Fast motion, single thing-class",
    },
    {
        "video": "assets/sidewalk_running_man.mp4",
        "prompts": ["sidewalk", "lawn"],
        "notes": "Two stuff-classes, running person stress test",
    },
    {
        "video": "assets/sideway_lawn.mp4",
        "prompts": ["sidewalk", "lawn"],
        "notes": "Two stuff-classes, moderate motion",
    },
    {
        "video": "assets/two_person_dog_lawn.mp4",
        "prompts": ["people", "dog", "lawn"],
        "notes": "Three prompts (2 thing + 1 stuff), multi-object",
    },
    {
        "video": "assets/pexels_baseball_field_drone.mp4",
        "prompts": ["lawn"],
        "notes": "Drone view, stuff-class (lawn)",
    },
    {
        "video": "assets/office_hallway_two_way.mp4",
        "prompts": ["floor", "wall"],
        "notes": "Steve / Nav2 use case, two stuff-classes",
    },
    {
        "video": "assets/indoor_to_outdoor.mp4",
        "prompts": ["floor", "wall", "sidewalk", "lawn"],
        "notes": "Scene transition, four stuff-classes",
    },
    {
        "video": "assets/gettyimages-2171845186-640_adpp.mp4",
        "prompts": ["dog"],
        "notes": "White dog on sidewalk, thing-class",
    },
]
