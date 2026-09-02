# ASR benchmark summary (long audio, Deepgram only)

**File:** `Test_2_Long_audio.wav` (~11.8 min / 707s, 64.7 MB @ 48 kHz mono)

## Model comparison (Fireworks skipped)

| Model | Latency | WER | CER | Verdict |
|-------|---------|-----|-----|---------|
| **nova-3** | 251s (~4.2 min) | 0.226 | 0.103 | **Keep (baseline)** |
| nova-2 | 259s (~4.3 min) | 0.242 | 0.119 | Do not switch — slower and worse quality |

**Recommendation:** Stay on `ASR_MODEL=nova-3`. Nova-2 does not reduce STEP1 time on long consults.

## Infra wins

- **ASR_PREP_WAV=true** resamples 48 kHz → 16 kHz mono via stdlib (no ffmpeg required): **64.7 MB → 21.6 MB** upload (~3× smaller).
- **ASR_PARALLEL_CHUNKS** (optional): split long audio into parallel Deepgram requests; enable only after running `scripts/benchmark_parallel_asr.py` on your audio.

## Your ~14 min / ~2 min ASR case

~2 min ASR on ~14 min audio is already ~7× faster than realtime. Further cuts require **parallel chunk ASR** (`ASR_PARALLEL_CHUNKS=true`), not nova-2.
