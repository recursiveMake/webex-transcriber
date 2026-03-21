# Meeting Transcriber

Transcribes WebEx meeting recordings with speaker identification and generates a structured summary — entirely on-device, no cloud APIs required.

Speaker attribution works by watching the video itself: WebEx displays a green microphone icon next to whichever participant is speaking. This is more reliable than audio-based diarisation, which struggles with similar voices, accents, and conference call audio quality.

**Output** — two Markdown files written alongside the video:

| File | Contents |
|---|---|
| `<meeting>_transcript.md` | Timestamped, speaker-labelled transcript |
| `<meeting>_summary.md` | Structured summary with inline citations |

---

## Requirements

| Dependency | Purpose | Install |
|---|---|---|
| **Python 3.11+** | Runtime | [python.org](https://python.org) |
| **ffmpeg** | Video/audio extraction | `brew install ffmpeg` |
| **Ollama** | Local LLM for summaries | [ollama.com](https://ollama.com) |
| **Apple Silicon Mac** | MLX/Metal acceleration | M1 or later |

> The pipeline runs on Intel Macs (falls back to CPU for Whisper and EasyOCR) but is significantly slower. Apple Silicon is the intended target.

### Pull an Ollama model

The default summary model is `gemma3:1b` (small, fast). For better summaries pull a larger model:

```bash
ollama pull gemma3:1b          # default — fast, ~1 GB
ollama pull llama3.2           # recommended — better quality, ~2 GB
ollama pull llama3.1:8b        # best quality, ~5 GB
```

---

## Installation

```bash
git clone <repo>
cd meeting

python3.11 -m venv .venv
source .venv/bin/activate

pip install -e .
```

---

## Quick Start

Make sure Ollama is running (`ollama serve` or the app is open), then:

```bash
meeting-transcriber weekly_sync.mp4
```

This produces `weekly_sync_transcript.md` and `weekly_sync_summary.md` next to the video.

---

## CLI Reference

```
meeting-transcriber [OPTIONS] VIDEO
```

| Option | Default | Description |
|---|---|---|
| `-o, --output-dir PATH` | Same dir as video | Where to write the two output files |
| `-m, --whisper-model` | `large-v3` | Whisper model size (see table below) |
| `--ollama-model TEXT` | `gemma3:1b` | Ollama model for the summary |
| `--frame-interval FLOAT` | `1.0` | Seconds between sampled frames for speaker detection |
| `--keep-work-dir` | off | Retain the temporary work directory (audio WAV, extracted frames) after processing |
| `-v, --verbose` | off | Enable DEBUG logging (per-frame speaker activity, OCR cache hits) |
| `-q, --quiet` | off | Suppress INFO logs; show warnings and errors only |
| `--list-ollama-models` | — | List locally available Ollama models and exit |

### Whisper model options

| Model | Size | Speed (M3 Pro) | Notes |
|---|---|---|---|
| `tiny` | 39 M | ~10× real-time | Useful for testing; word error rate is high |
| `base` | 74 M | ~7× real-time | Acceptable for clear audio |
| `small` | 244 M | ~4× real-time | Good balance for development |
| `medium` | 769 M | ~2× real-time | Recommended minimum for production use |
| `large-v2` | 1.5 GB | ~1× real-time | High accuracy |
| **`large-v3`** | **1.5 GB** | **~1× real-time** | **Default — best accuracy** |

Models are downloaded from Hugging Face on first use and cached locally. `large-v3` is ~3 GB on disk.

---

## Useful Configurations

### Fast development pass

When experimenting or verifying a new recording, use a smaller model and wider frame interval to get results quickly:

```bash
meeting-transcriber meeting.mp4 \
  --whisper-model small \
  --frame-interval 2.0 \
  --ollama-model gemma3:1b
```

### Production quality

```bash
meeting-transcriber meeting.mp4 \
  --whisper-model large-v3 \
  --ollama-model llama3.2 \
  -o ./transcripts/
```

### Debug speaker detection

If speakers are being misidentified or labelled "Unknown", run with verbose logging to see per-frame detection and OCR results:

```bash
meeting-transcriber meeting.mp4 -v 2>debug.log
```

Look for lines like:
- `OCR identified new participant: 'Alice Chen' (conf=0.87)` — confirms a name was read
- `Layout drift at 12.0s: tile count changed 3 → 4` — someone joined or left
- `No blobs found inside known layout tiles` — all-muted frame, expected during screen share

### Quiet mode for scripting

```bash
meeting-transcriber meeting.mp4 -q -o /output && echo "Done"
```

---

## Output Format

### Transcript (`_transcript.md`)

```markdown
# Meeting Transcript

*Source: weekly_sync.mp4*

---

**[00:00:08] Alice Chen**
Welcome everyone, let's get started with the agenda.

**[00:00:21] Bob Torres**
Thanks Alice. I wanted to flag that the API migration is complete.

**[00:00:35] Alice Chen**
That's great news. Let's make that official.
```

### Summary (`_summary.md`)

```markdown
# Meeting Summary

*Source: weekly_sync.mp4*

---

## Key Decisions
- Bob Torres owns the API migration sign-off [Bob Torres, 00:00:21]
- Deployment is scheduled for Thursday pending QA sign-off [Alice Chen, 00:02:14]

## Action Items
- **Bob Torres**: Submit migration report by end of week
- **Carol Kim**: Complete QA checklist by Wednesday [Carol Kim, 00:04:33]

## Topics Discussed
- API migration status and completion
- QA timeline and deployment gate
- Upcoming sprint planning
```

---

## Design Decisions

### Why visual speaker identification instead of audio diarisation

Audio diarisation models (pyannote, NeMo, etc.) work by finding acoustic differences between speakers. They degrade quickly when voices are similar, when participants are on different microphones, or when the audio is compressed (as WebEx recordings are). Their error rates are hard to predict and difficult to correct.

WebEx already solves this problem at the source: it knows who is speaking and displays a green microphone icon next to that participant's thumbnail. Parsing this visual signal gives ground-truth speaker identity with no acoustic modelling required.

The trade-off is that the approach is specific to WebEx's UI. The colour thresholds and tile inference logic would need adjustment for other platforms (Zoom, Teams, Meet), though the architecture is the same.

### Layout detection without user input

Rather than asking the user to define where the participant panel is, the detector finds it automatically:

1. Scan sampled frames for bright-green blobs matching WebEx's mic-active colour (HSV hue 40–90, saturation and value ≥ 120).
2. Cluster blob positions across frames. A position that appears in only one frame is noise; one that recurs is a real participant tile.
3. Infer each tile's bounding box by expanding upward from the mic icon position using a heuristic scaled to the frame resolution.
4. OCR the bottom strip of each tile to read the participant name. Names are cached by tile position — after the first successful read, subsequent frames require only a fast lookup.

A `LayoutTracker` watches for tile count changes or position shifts exceeding 80 pixels and re-runs detection. This handles late joiners, early leavers, and layout changes caused by screen sharing.

### Timestamp alignment across two models

Whisper and the visual speaker timeline are produced by independent processes with different timing characteristics:

- **Whisper** timestamps come from audio analysis and are accurate to ~100 ms.
- **Frame sampling** at 1 fps means each speaker detection has up to ±1 s of quantisation error.
- **WebEx display lag**: the green mic icon appears roughly 200–400 ms after a speaker starts and clears 100–300 ms after they stop.

Three mechanisms compensate:

1. **Temporal smoothing** — a 3-frame majority-vote window removes single-frame mic-icon flickers (e.g. from codec artefacts) before span construction. A speaker must dominate more than half of the surrounding frames to register as a change.

2. **Span padding** — every detected speaker span is expanded by 300 ms at its start and 200 ms at its end to absorb the visual lag. Adjacent expanded spans are then re-clamped to prevent overlaps.

3. **Boundary tolerance** — when assigning a Whisper segment to a speaker, the lookup window is widened by 400 ms at each boundary. Matches inside the tolerance zone are discounted to 25% weight so they only influence the result if no stronger candidate exists.

### Parallelism

```
┌─ ffmpeg audio extraction ──────────────────────► mlx-whisper (Metal) ──┐
│                                                                          ├──► align ──► Ollama ──► write
└─ ffmpeg frame extraction ──► speaker detection (MPS + frame prefetch) ──┘
```

Audio and frame extraction run in two threads simultaneously (both are ffmpeg subprocesses that are mostly I/O and codec work). Once extraction finishes, Whisper transcription and speaker detection run concurrently in two threads — they use different hardware (Whisper uses the Neural Engine and GPU via MLX; EasyOCR uses MPS; OpenCV uses NEON SIMD), so they don't compete.

Within speaker detection, a background thread pre-loads JPEG frames from disk into a bounded queue (depth 8) while the main thread processes them, hiding disk I/O latency behind computation.

EasyOCR is expensive (~500 ms per call) but is only ever called once per unique tile position across the entire video — subsequent frames are O(1) cache lookups. For a 1-hour meeting with 10 participants, that is at most 10 OCR calls regardless of meeting length.

---

## Expected Performance

All figures measured on an Apple Silicon M-series MacBook Pro. Times are approximate and depend on meeting content, number of participants, and available memory.

### Per-stage timing (60-minute meeting, 10 participants)

| Stage | Time | Notes |
|---|---|---|
| Audio + frame extraction | ~30 s | Both run in parallel; dominated by frame extraction |
| Whisper transcription (`large-v3`) | ~50–70 min | Runs concurrently with speaker detection |
| Speaker detection (3,600 frames) | ~2–5 min | Frame read: ~3 s; blob detection: ~4 s; OCR: ≤10 calls |
| Timeline + alignment | < 5 s | CPU-bound, fast |
| Ollama summary (`gemma3:1b`) | ~30–90 s | Depends on transcript length and model |
| File write | < 1 s | |

**Total wall-clock time** for a 60-minute meeting with `large-v3` + `gemma3:1b`: approximately **55–75 minutes**, dominated by Whisper. Whisper runs at roughly real-time speed on Apple Silicon for `large-v3`.

### Choosing a faster configuration

| Scenario | Whisper model | Frame interval | Estimated total time (60-min meeting) |
|---|---|---|---|
| Maximum quality | `large-v3` | 1.0 s | ~60–75 min |
| Recommended | `medium` | 1.0 s | ~25–35 min |
| Fast draft | `small` | 2.0 s | ~12–18 min |
| Quick check | `tiny` | 2.0 s | ~5–8 min |

Frame interval has a small effect on total time (affects speaker detection only, not Whisper). Whisper model size is the dominant variable.

### Memory

| Component | Peak RAM |
|---|---|
| `large-v3` weights | ~3.5 GB |
| EasyOCR model | ~500 MB |
| Frame buffer (depth 8 × 1080p JPEG) | ~40 MB |
| Total peak | ~4.5–5 GB |

`medium` reduces peak RAM to approximately 3 GB total.

---

## Troubleshooting

**Speakers all labelled "Unknown"**
The green mic blobs were not detected. Run with `-v` and check for `No blobs found` messages. Possible causes: the recording was exported with a WebEx layout that places thumbnails differently (e.g., full-screen speaker view with no strip), or the mic icon is obscured. Try inspecting a frame manually with `--keep-work-dir`.

**Names garbled or concatenated**
EasyOCR read the name strip incorrectly. Confidence scores are logged at INFO level. The name cache keeps the highest-confidence read, so accuracy tends to improve as the meeting progresses. If names are consistently wrong, the frame resolution or compression level may be too low for OCR — check the source recording quality.

**Whisper model download is slow**
Models are fetched from Hugging Face on first use. Set `HF_HUB_CACHE` to a fast local drive if the default cache location is on a network share.

**Ollama is not running**
The summary stage raises a `RuntimeError` with instructions. Start Ollama (`ollama serve`) or open the Ollama app before running. Use `--list-ollama-models` to confirm the server is reachable and the chosen model is pulled.

**Screen share breaks speaker detection**
When a participant shares their screen, WebEx often rearranges or minimises the participant strip. The layout tracker detects position shifts > 80 px and re-calibrates. Segments during extended screen shares with no visible thumbnails will be attributed to "Unknown". This is expected.
