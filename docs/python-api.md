# Python API

[Overview](../README.md) · [Configuration](configuration.md) · [Video](video.md)

Run examples from the repository root after installation. Configuration paths
are explicit caller inputs.

## Images and arrays

```python
from PIL import Image
from luma_fatal import Config, apply_filter

config = Config.load("configs/fattal-1.yaml")
with Image.open("assets/lantern-garden-original.png") as image:
    result = apply_filter(image, config=config)
```

`apply_filter` returns a Pillow image for Pillow input or a NumPy array for array
input. Arrays must be `uint8` with shape `(height, width, 3)` and already have the
intended RGB color interpretation. Arrays carry no ICC or EXIF metadata.
The `fattal1` and `fattal2` convenience functions select the variant directly.
A configuration with `filter: both` requires an explicit `variant=1` or `variant=2`
when calling `apply_filter`.

In-memory results do not use configured file encoding or create run records.
Opaque SDR RGB is the validated scope. Straight RGBA is an explicit experimental
extension, described in the [scope notes](method.md#scope).

## Image files

Use the file adapter to apply configured encoding and save a metadata sidecar:

```python
from luma_fatal.images import process_image

record = process_image(
    "assets/lantern-garden-original.png",
    "outputs/api.jpg",
    config_path="configs/fattal-1.yaml",
)
```

`config_path` loads settings when `config` is omitted. When both are supplied,
`config` controls processing and `config_path` records the underlying file.
The output extension must match the configured encoding. A file adapter processes
one filter per call.

The record includes resolved settings, dimensions, color conversion details,
dependency versions, and input/output hashes.

## Videos and decoded frames

```python
from luma_fatal.video import process_video

record = process_video(
    "local_inputs/input.mp4",
    "outputs/api.mp4",
    config=Config.load("configs/fattal-1.yaml"),
)
```

Video settings come from `config.video`. Passing `video_config=VideoConfig(...)`
explicitly overrides them. See the [video guide](video.md) for accepted inputs.

For your own decoder and encoder, use `process_frames`:

```python
from luma_fatal import process_frames

for filtered in process_frames(decoded_rgb_frames, config=config):
    write_frame(filtered)
```

Here `decoded_rgb_frames` and `write_frame` belong to the caller. Frames must be
RGB8 arrays. The iterator processes them lazily in input order and holds no
temporal state. Container timing, color interpretation, audio, and encoding remain
with the caller. Use one filter per stream.
