# usage: bash setup.sh
CHANNEL_URL="${CHANNEL_URL:-https://www.youtube.com/@wesawthatpod}"

set -e
D="$HOME/comedy-clips"; mkdir -p "$D/logs" && cd "$D"
command -v brew >/dev/null || { echo "Install Homebrew first: https://brew.sh"; exit 1; }
brew install ffmpeg python@3.12 deno >/dev/null   # deno: JS runtime yt-dlp needs for YouTube
ffmpeg -hide_banner -filters 2>/dev/null | grep -q " subtitles " || echo "WARNING: this ffmpeg lacks libass (captions) - run: brew reinstall ffmpeg"
"$(brew --prefix python@3.12)/bin/python3.12" -m venv .venv
.venv/bin/pip install -q --upgrade pip
.venv/bin/pip install -q mlx-whisper anthropic yt-dlp google-api-python-client google-auth-oauthlib

cat > clips.py <<'PYEOF'
#!/usr/bin/env python3
"""
comedy-clips: turn long-form comedy uploads into two videos per joke - a tight vertical Short
and the full joke as a normal 16:9 video (with thumbnail), both captioned.

  python clips.py run              # weekly job: new long-form uploads -> clips + schedule sheet
  python clips.py video <url|id>   # process one specific video now (ignores 'done' state)
  python clips.py status           # what's been processed, where the schedule has got to
  python clips.py auth             # one-off: sign in to the YouTube channel uploads go to
  python clips.py upload           # daily job: upload clips due in the next 48h as scheduled Shorts

Each run writes to out/<date>/:
  NN_<slug>_short.mp4   1080x1920 Short (hook to punchline), captions burned in
  NN_<slug>_long.mp4    1920x1080 full joke, captions burned in, + NN_<slug>_long.jpg thumbnail
  schedule.csv          title, description, publish slot (UK time), format, source timestamps
"""
import csv, datetime as dt, json, os, re, subprocess, sys, textwrap
from pathlib import Path
from zoneinfo import ZoneInfo

HOME = Path(__file__).resolve().parent
CFG_PATH, STATE_PATH = HOME / "config.json", HOME / "state.json"
WORK, OUT = HOME / "work", HOME / "out"
UK = ZoneInfo("Europe/London")

DEFAULTS = {
    "channel_url": "",                  # e.g. https://www.youtube.com/@yourchannel
    "min_source_minutes": 5,            # ignore anything shorter (your own Shorts etc.)
    "videos_per_run": 2,                # newest first, then works back through the catalogue
    "whisper_model": "medium.en",       # mlx-community name or full HF repo; small.en is faster, a bit worse on crowd voices
    "claude_model": "claude-sonnet-5-5",
    "min_clip_seconds": 15,
    "max_clip_seconds": 170,            # Shorts cap is 180s; leave margin
    "min_score": 6,                     # Claude rates each bit 1-10; below this is skipped
    "frame_zoom": 1.0,                  # 1.0 = full width of the two-shot; 1.2 trims 8% off each side
    "show_title": True,                 # clip title in the top band
    "slots_per_day": ["12:30", "19:30"],  # Shorts
    "short_max_seconds": 55,            # the Short cut; Shorts are ranked on how many watch to the end
    "long_slots_per_day": ["17:00"],    # full-length 16:9 clips; these count towards watch hours, Shorts don't
    "credit": "",                       # whose footage it is, e.g. "We Saw That (@wesawthatpod)"
    "contact_email": "",                # where comedians can ask for a credit change or takedown
    "ytdlp_extra": [],                  # e.g. ["--cookies-from-browser", "chrome"] if YouTube blocks downloads
    "upload_hours_ahead": 48,           # upload this far before each slot; leaves time to pull a clip in Studio
    "max_late_per_run": 1,              # clips whose slot has passed go out immediately, this many per run
    "mask_swears_in_captions": True,    # on-screen text counts for ad suitability; audio is never changed
}

# ---------------------------------------------------------------- helpers
def log(*a):
    print(dt.datetime.now().strftime("%H:%M:%S"), *a, flush=True)

def sh(cmd, **kw):
    return subprocess.run(cmd, check=True, text=True, capture_output=True, **kw).stdout

def load(path, default):
    return json.loads(path.read_text()) if path.exists() else default

def save(path, obj):
    tmp = path.with_suffix(".tmp"); tmp.write_text(json.dumps(obj, indent=2)); tmp.replace(path)

def slug(s, n=40):
    return re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-")[:n] or "clip"

def load_env():
    env = HOME / ".env"
    if env.exists():
        for line in env.read_text().splitlines():
            if "=" in line and not line.strip().startswith("#"):
                k, v = line.split("=", 1); os.environ.setdefault(k.strip(), v.strip().strip('"'))

def cfg():
    c = {**DEFAULTS, **load(CFG_PATH, {})}
    return c

# ---------------------------------------------------------------- 1. find + download
def list_uploads(c):
    url = c["channel_url"].rstrip("/")
    if not url:
        sys.exit("Set channel_url in config.json")
    if not url.endswith("/videos"):
        url += "/videos"          # the Videos tab excludes Shorts and lives
    out = sh(["yt-dlp", "--flat-playlist", "--print", "%(id)s\t%(duration)s\t%(title)s", *c["ytdlp_extra"], url])
    vids = []
    for line in out.splitlines():
        vid, dur, title = (line.split("\t", 2) + ["", ""])[:3]
        try: dur = float(dur)
        except ValueError: dur = 0
        if dur >= c["min_source_minutes"] * 60:
            vids.append({"id": vid, "title": title, "duration": dur})
    return vids  # newest first

def download(vid, c):
    d = WORK / vid; d.mkdir(parents=True, exist_ok=True)
    src = d / "source.mp4"
    if not src.exists():
        log(f"downloading {vid}")
        sh(["yt-dlp", "-f", "bv*[height<=1080][ext=mp4]+ba[ext=m4a]/b[height<=1080]/b",
            "--merge-output-format", "mp4", "-o", str(src), *c["ytdlp_extra"],
            f"https://www.youtube.com/watch?v={vid}"])
    return src

# ---------------------------------------------------------------- 2. transcribe
def transcribe(src, c):
    tj = src.parent / "transcript.json"
    if tj.exists():
        return json.loads(tj.read_text())
    log(f"transcribing with whisper {c['whisper_model']} on the GPU (slowest step)")
    import mlx_whisper
    repo = c["whisper_model"]
    if "/" not in repo:
        repo = f"mlx-community/whisper-{repo}-mlx"   # e.g. medium.en -> mlx-community/whisper-medium.en-mlx
    res = mlx_whisper.transcribe(str(src), path_or_hf_repo=repo, word_timestamps=True,
                                 condition_on_previous_text=False,
                                 hallucination_silence_threshold=2.0)  # skips junk text in long laughs
    out = []
    for s in res["segments"]:
        out.append({"start": round(s["start"], 2), "end": round(s["end"], 2), "text": s["text"].strip(),
                    "words": [{"s": round(w["start"], 2), "e": round(w["end"], 2), "w": w["word"].strip()}
                              for w in s.get("words", []) if w["word"].strip()]})
    tj.write_text(json.dumps(out))
    return out

# ---------------------------------------------------------------- 3. find the jokes
PROMPT = """You are editing a live comedy show (two hosts, lots of crowdwork) into YouTube Shorts.
Below is the full transcript, one line per segment: [start-end seconds] text.
Laughter is not transcribed; a gap between lines after a punchline usually means the room is laughing.

Pick out every self-contained bit that would work as a standalone Short for someone who has not seen the show:
- Start where a cold viewer can follow it (the question to the audience member, the setup) - not mid-thought.
- End right after the punchline or the final tag that gets the laugh. Do not run on into the next topic.
- Between {min_s} and {max_s} seconds. Tight is better: most great bits are 25-75 seconds.
- Bits must not overlap.
- Score each 1-10 on how well it stands alone and how hard it lands. Be honest; filler is fine to score low.
- Title: the Shorts title. Its only job is to make someone scrolling stop and tap.
  - 30-60 characters. Front-load the most surprising, specific detail (a place, a job, a number, the outrageous claim).
  - Open a curiosity gap: tease the setup, never give away the punchline.
  - Sound like a person, not a headline: a wild quote from the bit, or a reaction ("He said THIS to his girlfriend's nan").
  - At most one word in CAPS, no emojis, no hashtags, no "you won't believe" / "gone wrong" filler.
  - It must be true to the clip - no bait the clip doesn't pay off.
- Description: 1-2 short sentences (under 200 characters) that set the scene in different words from the title, ending with a question that gets people commenting. Do not repeat the title.
- Title and description must be clean: no swear words (even masked), nothing sexual, violent or insulting.
- Hashtags: 3-5 hashtags specific to this bit (e.g. #crowdwork #dating #london), not #shorts.
- Hook: the first line a viewer hears, quoted from the transcript, so the editor can check the start point.

Return ONLY JSON: {{"bits": [{{"start": 123.4, "end": 171.0, "score": 8, "title": "...", "description": "...", "hashtags": ["#crowdwork"], "hook": "...", "why": "one line"}}]}}

Video title: {title}

TRANSCRIPT
{transcript}"""

def ask_claude(prompt, c):
    """One streamed request; returns the JSON object in the answer, or exits without saving anything."""
    import anthropic
    ws = os.environ.get("ANTHROPIC_WORKSPACE_ID")   # needed when the API key isn't scoped to a workspace
    client = anthropic.Anthropic(default_headers={"anthropic-workspace-id": ws} if ws else None)
    # stream: thinking + 20-odd bits can run well past 8k tokens
    with client.messages.stream(model=c["claude_model"], max_tokens=64000,
                                messages=[{"role": "user", "content": prompt}]) as stream:
        msg = stream.get_final_message()
    if msg.stop_reason != "end_turn":   # never cache a truncated or refused answer
        sys.exit(f"Claude stopped early ({msg.stop_reason}); nothing saved, rerun to retry")
    text = "".join(b.text for b in msg.content if getattr(b, "type", "") == "text")
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        sys.exit("Claude's answer had no JSON; nothing saved, rerun to retry")
    return json.loads(m.group(0))

def find_bits(segs, title, c):
    lines = "\n".join(f"[{s['start']:.1f}-{s['end']:.1f}] {s['text']}" for s in segs)
    return ask_claude(PROMPT.format(min_s=c["min_clip_seconds"], max_s=c["max_clip_seconds"],
                                    title=title, transcript=lines), c)["bits"]

TITLE_PROMPT = """You write YouTube Shorts titles for clips from a live comedy show. Each clip's transcript is below.

The best titles are lines the comedian actually says - or a tightened version of one - picked because it is the most outrageous, quotable or relatable thing in the clip. They sound like a person talking, in their voice and slang, not like an editor describing the clip.

Good (said on stage):
  I think every politician should be autistic
  My biggest fear is being exposed on TikTok as a gym creep
  Bro, what are you WEARING?
  Man's not trying to see 12 Years a Slave with Liam Neeson
Bad (narrator voice, explains instead of grabbing):
  Every politician should be autistic, he argues
  Civil servant gets asked the minimum wage live
  He's ripped but wears flowers. Should we roast him?

For each clip:
- Write 5 options, mostly real lines from the transcript. Trim filler ("like", "you know", repeated words) and fix obvious transcription errors. At most one option may be a non-quote, and only if nothing in the clip is quotable.
- Never give away the punchline: the title is the setup or the bold claim, the clip is the payoff.
- 25-60 characters. No emojis, no hashtags, no quotation marks around the title. At most one word in CAPS.
- Keep it clean and advertiser-friendly: no swear words (not even masked with asterisks), nothing sexual, no violence or weapons, no drugs, no slurs, and nothing insulting about race, nationality, religion, disability or sexuality. If the funniest line breaks this, pick a different line or reword it. Keep harmless slang like bro, innit, mandem.
- Then pick the one most likely to make someone scrolling stop and tap.

Also write, under the same clean rules:
- description: 1-2 short sentences (under 200 characters) that set the scene in different words from the title and end with a question that gets people commenting. Do not give away the punchline.
- hashtags: 3-5 hashtags specific to the clip (e.g. #crowdwork #dating #london), not #shorts.

And pick the Short cut:
- short: the best 20-{short_max} second stretch inside the clip for a Short people watch to the end. Start on the line that hooks a cold viewer instantly (cut slow setup the joke survives without), end right after the biggest laugh. Use the [start-end] times given. If the whole clip is already under {short_max} seconds, use the whole clip.

Return ONLY JSON: {{"titles": [{{"clip": 1, "options": ["...", "..."], "best": "...", "description": "...", "hashtags": ["#crowdwork"], "short": {{"start": 123.4, "end": 160.2}}}}]}}

CLIPS
{clips}"""

# backstop for the "keep it clean" rule; also catches asterisk-masked versions (f*ck)
RUDE = re.compile(r"\b(f[\*u]ck\w*|sh[\*i]t\w*|b[\*i]tch\w*|d[\*i]ck\w*|p[\*u]ssy|c[\*u]nt\w*|cock\w*|"
                  r"wank\w*|twat\w*|prick\w*|bollocks|arse(hole)?|ass(hole)?|n[\*i]gg\w*|slut\w*|whore\w*|"
                  r"sex\w*|porn\w*|behead\w*|stab\w*|kill\w*|nazi\w*|rape\w*)\b", re.I)

def title_bits(bits, segs, c):
    """Second pass on each clip's own transcript: titles (its best line beats an editor's summary),
    description, hashtags, and the window for the Short cut."""
    clips = []
    for i, b in enumerate(bits, 1):
        said = "\n".join(f"[{s['start']:.1f}-{s['end']:.1f}] {s['text']}"
                         for s in segs if s["end"] > b["start"] and s["start"] < b["end"])
        clips.append(f"--- clip {i} (working title: {b['title']})\n{said}")
    prompt = TITLE_PROMPT.format(clips="\n\n".join(clips), short_max=c["short_max_seconds"])
    for t in ask_claude(prompt, c)["titles"]:
        b = bits[int(t["clip"]) - 1]
        clean = [o for o in [t["best"], *t["options"]] if not RUDE.search(o)]
        b["title_options"], b["title"] = t["options"], (clean[0] if clean else t["best"])
        b["description"], b["hashtags"] = t.get("description", b.get("description", "")), t.get("hashtags", [])
        short = t.get("short") or {}
        b["short_start"], b["short_end"] = short.get("start", b["start"]), short.get("end", b["end"])
        if not clean or RUDE.search(b["description"]):
            log(f"  WARNING: check wording on '{b['title']}' before uploading")

def snap(bit, segs, c):
    """Snap to segment boundaries and keep the laugh after the punchline."""
    starts = [s["start"] for s in segs]
    # start: the segment start nearest to (and not after) Claude's start, minus a breath
    cand = [x for x in starts if x <= bit["start"] + 0.5]
    start = max(cand) if cand else bit["start"]
    start = max(0.0, start - 0.25)
    # end: end of the segment containing the punchline, then ride the laugh until the
    # next speech starts (capped at +4s, min +1s)
    ends = [s for s in segs if s["start"] <= bit["end"] + 0.5]
    seg_end = ends[-1]["end"] if ends else bit["end"]
    nxt = [s["start"] for s in segs if s["start"] > seg_end]
    gap_end = nxt[0] if nxt else seg_end + 2.5
    end = min(max(seg_end + 1.0, gap_end - 0.15), seg_end + 4.0)
    end = min(end, start + c["max_clip_seconds"])
    return round(start, 2), round(end, 2)

# ---------------------------------------------------------------- 4. render
def ass_time(t):
    t = max(0, t); h = int(t // 3600); m = int(t % 3600 // 60); s = t % 60
    return f"{h}:{m:02d}:{s:05.2f}"

def ass_escape(s):
    return s.replace("\\", "").replace("{", "(").replace("}", ")")

SWEARS = re.compile(r"\b(motherf\w*|f+u+c*k\w*|sh+i+t\w*|bitch\w*|dick\w*|pussy\w*|cunt\w*|cock\w*|"
                    r"wank\w*|twat\w*|prick\w*|bollocks|arse(hole)?|n[i1]gg\w*)\b", re.I)

def mask_swears(w, c):
    """Captions only (the audio is untouched): fuck -> f**k."""
    if not c["mask_swears_in_captions"]:
        return w
    return SWEARS.sub(lambda m: m.group(0)[0] + "*" * (len(m.group(0)) - 2) + m.group(0)[-1], w)

def ass_head(W, H, cap_size, title_size, title_align, title_margin):
    # brand: yellow &H000AD6FF, ink &H00100E0E (ASS colours are &HAABBGGRR); Title is a solid box (BorderStyle 3)
    return f"""[Script Info]
ScriptType: v4.00+
PlayResX: {W}
PlayResY: {H}
WrapStyle: 0

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Cap,Arial Black,{cap_size},&H00FFFFFF,&H00FFFFFF,&H00000000,&H64000000,0,0,0,0,100,100,0,0,1,7,3,5,60,60,0,1
Style: Title,Arial Black,{title_size},&H00100E0E,&H00100E0E,&H000AD6FF,&H000AD6FF,0,0,0,0,100,100,0,0,3,22,0,{title_align},90,90,{title_margin},1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""

def build_ass(words, start, end, title, c, path, fmt="short"):
    """Big chunked captions (2-3 words at a time, spoken word highlighted). Shorts also get the title box;
    the 16:9 version doesn't need it on screen because its thumbnail carries the title."""
    if fmt == "short":
        W, H = 1080, 1920
        vid_h = int(W / 16 * 9 * c["frame_zoom"])
        cap_y = (H + vid_h) // 2 + 180          # just below the video frame
        head = ass_head(W, H, 84, 72, 8, 190)
    else:
        W, H = 1920, 1080
        cap_y = H - 140                          # lower third, over the audience
        head = ass_head(W, H, 72, 72, 8, 60)
    cx = W // 2
    ev = []
    if fmt == "short" and c["show_title"] and title:
        wrapped = "\\N".join(textwrap.wrap(ass_escape(title), 22)[:3])
        ev.append(f"Dialogue: 0,{ass_time(0)},{ass_time(end - start)},Title,,0,0,0,,{wrapped}")
    ws = [w for w in words if w["e"] > start and w["s"] < end]
    chunks, cur = [], []
    for w in ws:
        cur.append(w)
        txt = " ".join(x["w"] for x in cur)
        if len(cur) >= 3 or len(txt) > 16 or re.search(r"[.?!,]$", w["w"]):
            chunks.append(cur); cur = []
    if cur: chunks.append(cur)
    for i, ch in enumerate(chunks):
        s = ch[0]["s"] - start
        e = (chunks[i + 1][0]["s"] - start) if i + 1 < len(chunks) else ch[-1]["e"] - start + 0.4
        e = min(e, ch[-1]["e"] - start + 0.6)       # don't hold a caption through a laugh
        e = max(e, s + 0.2)
        for j in range(len(ch)):                    # current word lights up in brand yellow
            ws = s if j == 0 else ch[j]["s"] - start
            we = e if j == len(ch) - 1 else min(ch[j + 1]["s"] - start, e)
            if we <= ws:
                continue
            words_ = [ass_escape(mask_swears(x["w"], c).upper()) for x in ch]
            words_[j] = f"{{\\c&H0AD6FF&}}{words_[j]}{{\\c&HFFFFFF&}}"
            ev.append(f"Dialogue: 1,{ass_time(ws)},{ass_time(we)},Cap,,0,0,0,,{{\\pos({cx},{cap_y})}}{' '.join(words_)}")
    path.write_text(head + "\n".join(ev) + "\n")

def render(src, start, end, ass, out, c, fmt="short"):
    z = c["frame_zoom"]
    if fmt == "short":
        # blurred, zoomed copy fills 9:16; sharp 16:9 frame sits in the middle
        vf = (f"[0:v]split[a][b];"
              f"[a]scale=-2:1920,crop=1080:1920,boxblur=30:3,eq=brightness=-0.12[bg];"
              f"[b]scale={int(1080*z)}:-2,crop=1080:ih[fg];"
              f"[bg][fg]overlay=(W-w)/2:(H-h)/2,subtitles='{ass.as_posix()}'[v]")
    else:
        vf = (f"[0:v]scale=1920:1080:force_original_aspect_ratio=decrease,"
              f"pad=1920:1080:(ow-iw)/2:(oh-ih)/2,setsar=1,subtitles='{ass.as_posix()}'[v]")
    sh(["ffmpeg", "-y", "-ss", f"{start:.2f}", "-to", f"{end:.2f}", "-i", str(src),
        "-filter_complex", vf, "-map", "[v]", "-map", "0:a",
        "-af", "loudnorm=I=-14:TP=-1.5:LRA=11,afade=t=in:d=0.08",
        "-c:v", "libx264", "-preset", "medium", "-crf", "20", "-pix_fmt", "yuv420p",
        "-r", "30", "-c:a", "aac", "-b:a", "160k", "-movflags", "+faststart", str(out)])

def thumbnail(src, t, title, ass, out):
    """1280x720 thumbnail for the 16:9 version: a punchy frame, the title in a big yellow box."""
    lines = "\\N".join(textwrap.wrap(ass_escape(title.upper()), 20)[:3])
    ass.write_text(ass_head(1280, 720, 60, 76, 1, 40)
                   + f"Dialogue: 0,{ass_time(0)},{ass_time(5)},Title,,0,0,0,,{lines}\n")
    sh(["ffmpeg", "-y", "-ss", f"{t:.2f}", "-i", str(src), "-frames:v", "1",
        "-vf", f"scale=1280:720:force_original_aspect_ratio=increase,crop=1280:720,"
               f"eq=saturation=1.3:contrast=1.1,subtitles='{ass.as_posix()}'",
        "-q:v", "2", str(out)])

# ---------------------------------------------------------------- 5. schedule
def next_slots(state, c, n, fmt="short"):
    """Continue from wherever last week's schedule ended (or tomorrow). Shorts and 16:9 have their own slots."""
    times = [dt.time(*map(int, t.split(":"))) for t in c["slots_per_day" if fmt == "short" else "long_slots_per_day"]]
    last = state.get("last_slot" if fmt == "short" else "last_slot_long")
    now = dt.datetime.now(UK)
    cur = dt.datetime.fromisoformat(last).astimezone(UK) if last else now
    if cur < now: cur = now
    slots, day = [], cur.date()
    while len(slots) < n:
        for t in times:
            s = dt.datetime.combine(day, t, UK)
            if s > cur and len(slots) < n:
                slots.append(s)
        day += dt.timedelta(days=1)
    return slots

# ---------------------------------------------------------------- pipeline
def process(v, c, state, run_dir, rows):
    vid = v["id"]
    src = download(vid, c)
    segs = transcribe(src, c)
    bits_path = src.parent / "bits.json"
    if bits_path.exists():
        bits = json.loads(bits_path.read_text())
    else:
        log("finding the bits with Claude")
        bits = find_bits(segs, v["title"], c)
        bits_path.write_text(json.dumps(bits, indent=2))
    words = [w for s in segs for w in s["words"]]
    keep = sorted([b for b in bits if b.get("score", 0) >= c["min_score"]], key=lambda b: b["start"])
    log(f"{vid}: {len(bits)} bits found, {len(keep)} at score {c['min_score']}+")
    if keep and not all("short_start" in b for b in keep):
        log("writing titles and picking Short cuts with Claude")
        title_bits(keep, segs, c)          # updates the dicts in bits too
        bits_path.write_text(json.dumps(bits, indent=2))
    for b in keep:
        start, end = snap(b, segs, c)
        if end - start < c["min_clip_seconds"]:
            continue
        # the Short: Claude's hook-to-punchline window, snapped the same way and kept inside the full clip
        ss, se = snap({"start": b["short_start"], "end": b["short_end"]}, segs, c)
        ss, se = max(ss, start), min(se, end, max(ss, start) + c["short_max_seconds"] + 4)
        if se - ss < c["min_clip_seconds"]:
            ss, se = start, min(end, start + c["short_max_seconds"])
        n = len({r["name"] for r in rows}) + 1
        name = f"{n:02d}_{slug(b['title'])}"
        common = {"name": name, "score": b["score"], "title": b["title"],
                  "alt_titles": " | ".join(o for o in b.get("title_options", []) if o != b["title"]),
                  "source_video": v["title"], "hook": b.get("hook", ""), "why": b.get("why", "")}
        desc = b.get("description") or b["title"]
        full_show = f"Full show: https://youtu.be/{vid}?t={int(start)}"
        tags = " ".join(b.get("hashtags", []))

        ass = WORK / vid / f"{name}_short.ass"
        build_ass(words, ss, se, b["title"], c, ass, "short")
        log(f"  rendering {name} short ({se-ss:.0f}s) + long ({end-start:.0f}s), score {b['score']}")
        render(src, ss, se, ass, run_dir / f"{name}_short.mp4", c, "short")
        rows.append({**common, "format": "short", "file": f"{name}_short.mp4", "thumbnail": "",
                     "description": f"{desc}\n\n{full_show}\n\n#shorts {tags}\n\n{footer(c)}",
                     "source_start": f"{int(ss//60)}:{int(ss%60):02d}", "length_s": round(se - ss)})

        ass = WORK / vid / f"{name}_long.ass"
        build_ass(words, start, end, b["title"], c, ass, "long")
        render(src, start, end, ass, run_dir / f"{name}_long.mp4", c, "long")
        thumbnail(src, start + (end - start) * 0.4, b["title"], WORK / vid / f"{name}_thumb.ass",
                  run_dir / f"{name}_long.jpg")
        rows.append({**common, "format": "long", "file": f"{name}_long.mp4", "thumbnail": f"{name}_long.jpg",
                     "description": f"{desc}\n\n{full_show}\n\n{tags}\n\n{footer(c)}",
                     "source_start": f"{int(start//60)}:{int(start%60):02d}", "length_s": round(end - start)})

def footer(c):
    """Credit + rights notice + takedown contact, on every description."""
    who = c["credit"] or "the comedians and the original show"
    lines = [f"Credit: {who}. All rights to this footage belong to them. If you enjoyed this, "
             f"watch the full show at the link above and support them.",
             "Clips are shared to promote the comedians and their shows. Views expressed are the "
             "comedians' own, performed as comedy."]
    if c["contact_email"]:
        lines.append(f"Comedians and rights holders: to change a credit or have a clip removed, email "
                     f"{c['contact_email']} and we'll sort it within 24 hours.")
    return "\n".join(lines)

def write_schedule(rows, run_dir, state, c):
    if not rows:
        return
    out = []
    for fmt, key in (("short", "last_slot"), ("long", "last_slot_long")):
        # best first, but never two clips from the same source back-to-back if avoidable
        pool = sorted([r for r in rows if r["format"] == fmt], key=lambda r: -r["score"])
        if not pool:
            continue
        ordered = []
        while pool:
            pick = next((r for r in pool if not ordered or r["source_video"] != ordered[-1]["source_video"]), pool[0])
            ordered.append(pick); pool.remove(pick)
        slots = next_slots(state, c, len(ordered), fmt)
        for r, s in zip(ordered, slots):
            r["publish_uk"], r["_when"] = s.strftime("%a %d %b %H:%M"), s
        state[key] = slots[-1].isoformat()
        out += ordered
    out.sort(key=lambda r: r["_when"])
    cols = ["publish_uk", "format", "file", "thumbnail", "title", "alt_titles", "description", "score",
            "length_s", "source_video", "source_start", "hook", "why"]
    with open(run_dir / "schedule.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore"); w.writeheader(); w.writerows(out)

# ---------------------------------------------------------------- 6. upload
SCOPES = ["https://www.googleapis.com/auth/youtube.upload"]
CLIENT_SECRET, TOKEN = HOME / "client_secret.json", HOME / "token.json"

def youtube(interactive=False):
    from google.oauth2.credentials import Credentials
    from google.auth.transport.requests import Request
    from google_auth_oauthlib.flow import InstalledAppFlow
    from googleapiclient.discovery import build
    creds = Credentials.from_authorized_user_file(str(TOKEN), SCOPES) if TOKEN.exists() else None
    if creds and creds.expired and creds.refresh_token:
        creds.refresh(Request())
    elif not creds or not creds.valid:
        if not CLIENT_SECRET.exists():
            sys.exit(f"Not set up for uploads: put the OAuth client file at {CLIENT_SECRET}, then run: python clips.py auth")
        if not interactive:               # scheduled runs can't open a browser
            sys.exit("Not signed in to YouTube: run  python clips.py auth")
        creds = InstalledAppFlow.from_client_secrets_file(str(CLIENT_SECRET), SCOPES).run_local_server(port=0)
    TOKEN.write_text(creds.to_json())
    return build("youtube", "v3", credentials=creds)

def slot_time(run_dir, publish_uk):
    """'Mon 19 Oct 12:30' + the run folder's date -> aware datetime (handles the year rolling over)."""
    d = dt.date.fromisoformat(run_dir.name)
    t = dt.datetime.strptime(f"{publish_uk} {d.year}", "%a %d %b %H:%M %Y")
    if t.date() < d:
        t = t.replace(year=d.year + 1)
    return t.replace(tzinfo=UK)

def upload(c, state):
    from googleapiclient.errors import HttpError
    from googleapiclient.http import MediaFileUpload
    done = state.setdefault("uploaded", {})          # "<run date>/<file>" -> YouTube video id
    now = dt.datetime.now(UK)
    horizon = now + dt.timedelta(hours=c["upload_hours_ahead"])
    yt, n, late = None, 0, 0
    for sched in sorted(OUT.glob("*/schedule.csv")):
        run_dir = sched.parent
        if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", run_dir.name):   # skip archived copies like 2026-10-19-v1
            continue
        with open(sched, newline="") as f:
            rows = list(csv.DictReader(f))
        for r in sorted(rows, key=lambda r: slot_time(run_dir, r["publish_uk"])):
            key = f"{run_dir.name}/{r['file']}"
            when = slot_time(run_dir, r["publish_uk"])
            if key in done or when > horizon or not (run_dir / r["file"]).exists():
                continue
            status = {"selfDeclaredMadeForKids": False}
            if when > now + dt.timedelta(minutes=15):
                status.update(privacyStatus="private",     # YouTube flips it public at publishAt
                              publishAt=when.astimezone(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"))
            elif late < c["max_late_per_run"]:
                status["privacyStatus"] = "public"; late += 1
            else:
                continue                                    # missed slot; goes out on a later run
            tags = [w.lstrip("#") for w in r["description"].split() if w.startswith("#")]
            body = {"snippet": {"title": r["title"][:100], "description": r["description"],
                                "tags": tags, "categoryId": "23",       # 23 = Comedy
                                "defaultLanguage": "en", "defaultAudioLanguage": "en-GB"},
                    "status": status}
            yt = yt or youtube()
            log(f"uploading {key}")
            try:
                req = yt.videos().insert(part="snippet,status", body=body,
                                         media_body=MediaFileUpload(str(run_dir / r["file"]), chunksize=-1, resumable=True))
                resp = None
                while resp is None:
                    _, resp = req.next_chunk()
            except HttpError as e:
                if "quotaExceeded" in str(e) or "uploadLimitExceeded" in str(e):
                    log("YouTube upload quota used up for today; the rest go tomorrow"); return n
                raise
            done[key] = resp["id"]; save(STATE_PATH, state); n += 1
            if r.get("thumbnail") and (run_dir / r["thumbnail"]).exists():
                try:
                    yt.thumbnails().set(videoId=resp["id"], media_body=MediaFileUpload(str(run_dir / r["thumbnail"]))).execute()
                except HttpError as e:     # custom thumbnails need a phone-verified channel
                    log(f"  thumbnail not set ({e.status_code}); verify the channel's phone number in YouTube Studio")
            log(f"  -> https://youtu.be/{resp['id']}  " + (f"scheduled {r['publish_uk']}" if "publishAt" in status else "public now"))
    return n

def notify(msg):
    if sys.platform == "darwin":
        subprocess.run(["osascript", "-e", f'display notification "{msg}" with title "Comedy clips"'])

def main():
    load_env()
    c, state = cfg(), load(STATE_PATH, {"done": []})
    cmd = sys.argv[1] if len(sys.argv) > 1 else "run"
    if cmd == "status":
        print(json.dumps(state, indent=2)); return
    if cmd == "auth":
        youtube(interactive=True); log("signed in; uploads will go to the channel you picked"); return
    if cmd == "upload":
        n = upload(c, state)
        log(f"done: {n} uploaded")
        if n: notify(f"{n} clips uploaded to YouTube")
        return
    run_dir = OUT / dt.date.today().isoformat(); run_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    if cmd == "video":
        ref = sys.argv[2]
        vid = re.search(r"(?:v=|youtu\.be/|shorts/|live/)?([A-Za-z0-9_-]{11})(?:\W|$)", ref).group(1)
        tf = WORK / vid / "title.txt"     # cached: saves a YouTube request on reruns
        if tf.exists():
            title = tf.read_text().strip()
        else:
            title = sh(["yt-dlp", "--print", "%(title)s", *c["ytdlp_extra"], f"https://youtu.be/{vid}"]).strip()
            tf.parent.mkdir(parents=True, exist_ok=True); tf.write_text(title)
        todo = [{"id": vid, "title": title}]
    else:
        todo = [v for v in list_uploads(c) if v["id"] not in state["done"]][: c["videos_per_run"]]
        if not todo:
            log("nothing new"); notify("No new long-form videos this week"); return
    for v in todo:
        log(f"== {v['title']}")
        process(v, c, state, run_dir, rows)
        if v["id"] not in state["done"]:
            state["done"].append(v["id"])
        save(STATE_PATH, state)
    write_schedule(rows, run_dir, state, c)
    save(STATE_PATH, state)
    log(f"done: {len(rows)} clips in {run_dir}")
    notify(f"{len(rows)} clips ready to schedule")
    if sys.platform == "darwin" and rows:
        subprocess.run(["open", str(run_dir)])

if __name__ == "__main__":
    main()
PYEOF

[ -f config.json ] || cat > config.json <<EOF
{
  "channel_url": "$CHANNEL_URL",
  "videos_per_run": 2,
  "min_score": 6,
  "slots_per_day": ["12:30", "19:30"]
}
EOF

cat > weekly.sh <<'EOF'
#!/bin/zsh
cd "$HOME/comedy-clips"
export PATH="$HOME/comedy-clips/.venv/bin:/opt/homebrew/bin:/usr/local/bin:$PATH"
pip install -q -U yt-dlp            # YouTube changes often; keep the downloader current
echo "==== $(date)"
caffeinate -i python clips.py run
EOF
chmod +x weekly.sh

cat > upload.sh <<'EOF'
#!/bin/zsh
cd "$HOME/comedy-clips"
export PATH="$HOME/comedy-clips/.venv/bin:/opt/homebrew/bin:/usr/local/bin:$PATH"
echo "==== $(date)"
python clips.py upload
EOF
chmod +x upload.sh

PL="$HOME/Library/LaunchAgents/com.comedyclips.weekly.plist"
cat > "$PL" <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
  <key>Label</key><string>com.comedyclips.weekly</string>
  <key>ProgramArguments</key><array><string>$D/weekly.sh</string></array>
  <key>StartCalendarInterval</key><dict><key>Weekday</key><integer>0</integer><key>Hour</key><integer>7</integer><key>Minute</key><integer>0</integer></dict>
  <key>StandardOutPath</key><string>$D/logs/run.log</string>
  <key>StandardErrorPath</key><string>$D/logs/run.log</string>
</dict></plist>
EOF
launchctl bootout gui/$(id -u) "$PL" 2>/dev/null || true
launchctl bootstrap gui/$(id -u) "$PL"
PU="$HOME/Library/LaunchAgents/com.comedyclips.upload.plist"
cat > "$PU" <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
  <key>Label</key><string>com.comedyclips.upload</string>
  <key>ProgramArguments</key><array><string>$D/upload.sh</string></array>
  <key>StartCalendarInterval</key><dict><key>Hour</key><integer>8</integer><key>Minute</key><integer>0</integer></dict>
  <key>StandardOutPath</key><string>$D/logs/upload.log</string>
  <key>StandardErrorPath</key><string>$D/logs/upload.log</string>
</dict></plist>
EOF
launchctl bootout gui/$(id -u) "$PU" 2>/dev/null || true
launchctl bootstrap gui/$(id -u) "$PU"
echo "Installed. Clips made Sundays 07:00, uploads daily 08:00. Next: add your API key, then client_secret.json + python clips.py auth."
