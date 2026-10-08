#!/bin/sh
# Assemble the Phase 10 demo video from browser captures + PIL cards.
# Run from the repo root: sh demo-video/assemble.sh
# Inputs: /tmp/cap-*.webm (git-ignored source captures, see SCRIPT.md).
# Outputs: demo-video/trustlayer-demo.mp4 + demo-video/demo.gif
set -eu
cd "$(dirname "$0")"
B=build
mkdir -p "$B"
FF="ffmpeg -hide_banner -loglevel error -y"
NORM="scale=1280:720:force_original_aspect_ratio=decrease,pad=1280:720:(ow-iw)/2:(oh-ih)/2,fps=30"

# 1. Normalise captures to 1280x720p30 + trim heads/tails/waits.
$FF -i /tmp/cap-ask.webm -vf "$NORM,trim=2:24,setpts=PTS-STARTPTS" \
  -c:v libx264 -preset veryfast -crf 22 -an "$B/ask.mp4"
# answer: submit click ~4s, render ~177s; keep click, skip the wait, keep render.
$FF -i /tmp/cap-login.webm -vf "$NORM,trim=2:6,setpts=PTS-STARTPTS" \
  -c:v libx264 -preset veryfast -crf 22 -an "$B/answer-click.mp4"
$FF -ss 178 -i /tmp/cap-login.webm -vf "$NORM,trim=0:14,setpts=PTS-STARTPTS" \
  -c:v libx264 -preset veryfast -crf 22 -an "$B/answer-render.mp4"
$FF -i /tmp/cap-result.webm -vf "$NORM,trim=65:91,setpts=PTS-STARTPTS" \
  -c:v libx264 -preset veryfast -crf 22 -an "$B/classify.mp4"
$FF -i /tmp/cap-block.webm -vf "$NORM,trim=20:40,setpts=PTS-STARTPTS" \
  -c:v libx264 -preset veryfast -crf 22 -an "$B/block.mp4"
$FF -i /tmp/cap-allow.webm -vf "$NORM,trim=4:24,setpts=PTS-STARTPTS" \
  -c:v libx264 -preset veryfast -crf 22 -an "$B/allow.mp4"

# 2. Caption overlays: bottom bar over the last 6s of each captioned segment.
cap() { # $1=segment $2=card; overlay window computed from segment duration
  dur=$(ffprobe -v error -show_entries format=duration -of csv=p=0 "$B/$1.mp4")
  start=$(python3 -c "print(max(0.0, float('$dur') - 6.0))")
  $FF -i "$B/$1.mp4" -i "$B/$2.png" \
    -filter_complex "[0:v][1:v]overlay=0:H-64:enable='gte(t,$start)'" \
    -c:v libx264 -preset veryfast -crf 22 -an "$B/$1-cap.mp4"
}
cap ask cap1
cap answer-render cap3
cap classify cap4
cap block cap5
cap allow cap6

# 3. Title/end cards as 3s / 4s clips.
$FF -loop 1 -framerate 30 -t 3 -i "$B/title.png" \
  -c:v libx264 -preset veryfast -crf 22 -pix_fmt yuv420p -an "$B/title.mp4"
$FF -loop 1 -framerate 30 -t 4 -i "$B/end.png" \
  -c:v libx264 -preset veryfast -crf 22 -pix_fmt yuv420p -an "$B/end.mp4"

# 4. Concat + GIF preview (ask segment only, 640px, ~8s).
printf "file '%s'\n" title.mp4 ask-cap.mp4 answer-click.mp4 answer-render-cap.mp4 \
  classify-cap.mp4 block-cap.mp4 allow-cap.mp4 end.mp4 > "$B/list.txt"
$FF -f concat -safe 0 -i "$B/list.txt" -c:v libx264 -preset veryfast -crf 22 \
  -pix_fmt yuv420p -an trustlayer-demo.mp4
$FF -i "$B/ask-cap.mp4" -vf "fps=10,scale=640:-1:flags=lanczos,trim=2:10,setpts=PTS-STARTPTS" \
  demo.gif
ffprobe -v error -show_entries format=duration,size -of default=noprint_wrappers=1 trustlayer-demo.mp4
ls -la trustlayer-demo.mp4 demo.gif
