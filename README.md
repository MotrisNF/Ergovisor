*This project has been created as part of the 42 curriculum by saperez-.*

# RAG against the machine: workplace ergonomics from video

The system watches people working, from a video file or a live camera,
and checks their work against the Spanish occupational risk prevention
rules. For every risk it finds, it retrieves the passage of the
regulation or technical guide behind it, and a language model can write
a formal report that cites those passages.

Everything runs on the computer: the video and the images never leave it,
and nobody is recognised (people are "Persona 1", "Persona 2"...). The
reports are anonymous, as the data protection rules (RGPD) ask of
measures of people at work: they keep the day of the session but not its
time, and how often and how long each thing happened, never when (no
timeline, no second of the session). Only the optional written report
sends those measures (numbers and words, no images) to Gemini.

The documentation of the original RAG project (the vLLM question
answering system this one started from) is in
[original_docs/](original_docs/README.md).

## Description

What is measured, and with which method:

| Risk | Method | What the video gives |
|---|---|---|
| Postures | REBA (NTP 601, INSST) | Trunk, neck, arms and legs, frame by frame; moments of risk with their causes |
| Lifting and lowering loads | Revised NIOSH equation (NTP 477) and INSST guide (2024) | Each two-handed lift: heights, distances, twist, frequency; the weight limit and, with the load given, the lifting index |
| Pushing and pulling | INSST guide (2024), ISO/TR 12295 force limits | Each two-handed push or pull: distance, hand height, trunk posture |
| Screen work (only with `SCREEN=1` / `--screen`) | RD 488/1997 and its INSST guide (2021) | Eye-screen distance, neck posture, blinks, visual rests (20-20-20) and breaks |

How it works:

1. **People and skeletons.** MediaPipe Pose gives a 3D skeleton of each
   person. With one person the model sees the whole image; when a second
   person comes in, a people detector (YOLOX-Tiny, run with OpenCV) finds
   each one and the pose model measures each in their own part of the
   image. Up to 3 people per camera (`MAX_PEOPLE` in
   `src/posture/people.py`; `analyze --people N`).
2. **Measures.** Each person is analysed apart: REBA, lifts, pushes, and,
   when asked for (`make app SCREEN=1`, `analyze --screen`) and the
   camera is a laptop's (face and shoulders, no hips), the screen work
   with MediaPipe's face model.
3. **Regulation.** Each kind of risk is searched in an index of the
   prevention corpus (`data/raw/prl/`: Law 31/1995, RD 487/1997,
   RD 488/1997, the INSST guides on manual handling and screens, NTP 477
   and NTP 601) with the project's own BM25 retriever.
4. **Report.** On request (or by itself when a session ends with
   something found), Gemini writes the ergonomic assessment report from
   the measures and those passages, with what a risk assessment must
   record (RD 39/1997, art. 7): the job, the risks, the people affected,
   the result, and the preventive measures planned by priority (art. 9).
   The level and priority of each risk are worked out by the program from
   each method's scale (`src/posture/assessment.py`), not by the model;
   every citation is checked against the passages given.

Every threshold the regulation does not fix is marked "Ours" in the code,
with the calibration it comes from: see
[calibracion_videos.md](calibracion_videos.md) for the calibration videos
and their results. To keep false alarms down, a movement only counts as a
lift or a push when it is made with both hands holding one object (the
NIOSH equation and the guide's push-pull methods are for two hands), and
postures are only scored when the whole body is in the picture and near
enough to measure it (standing, at least 40 % of the image's height).

## Instructions

### Requirements

- Linux or macOS, Python 3.10+, [uv](https://docs.astral.sh/uv/).
- A webcam, a USB camera or an IP camera (a phone with IP Webcam works).
- Optional: `ffmpeg`, so the drawn and recorded videos play in any
  browser; a Gemini API key for the written reports: copy `.env-example`
  to `.env` and put it there (`.env` is never committed; a
  `GEMINI_API_KEY` exported in the shell takes precedence).

### Installation

```console
$ make install
$ uv run -m src index --raw_dir data/raw/prl --index_dir data/processed/prl
```

The pose, face and people models (6-31 MB each) are downloaded the first
time they are used and kept in `~/.cache/rag-posture/`.

### The application

```console
$ make app                 # http://127.0.0.1:8000, this computer only
$ make app LLM=hydrogen    # the best written report (about a minute)
$ make app RECORD=1        # allow recording the cameras' original video
$ make app SCREEN=1        # also analyse the work at a screen
```

Add a camera, an IP address or a video; up to four at once, each in the
background. Each one saves its report when it ends; the reports, their
regulation, the written report (.docx or printed) and the summaries of a
period are in "Informes".

### Command line

```console
$ uv run -m src analyze --video work.mp4 --load_kg 8 --overlay drawn.mp4
$ uv run -m src analyze --camera 0 --show --regulation_dir data/processed/prl
$ uv run -m src regulation data/output/posture/work.json
```

`uv run -m src analyze --help` lists the task data the video does not
show (load, grip, task duration, sex and age, height, push forces).

### Tests

```console
$ make test     # pytest
$ make lint     # flake8 and mypy
```

## Known limitations

- It is an automatic analysis: it does not replace the risk assessment by
  competent technical staff, and every report says so.
- The video does not show the load's weight nor the push force: they are
  given by the person, or the report says they could not be assessed
  (without the load, a lift gets its weight limit but no level of risk).
  Nor how long the task goes on: unless given, a short task (1 h or
  less) is assumed and the report says so.
- Measures no body can take (a stature under 140 cm or over 210 cm, the
  hands out of reach, the trunk bent over 135°) are the model's error and
  are not scored.
- Some REBA points cannot be seen on a skeleton (a supported arm, wrist
  twist, shoulders raised) and are never added.
- People are numbered by where they are: when two cross or one leaves and
  another comes in, their numbers can swap.
- Seen side-on, the depth (towards the camera) is the least certain
  measure: sideways bends are not scored then.

## Resources

### References

- INSST: NTP 601 (REBA), NTP 477 (NIOSH equation), *Guía técnica para la
  evaluación y prevención de los riesgos relativos a la manipulación
  manual de cargas* (2024), *Guía técnica ... pantallas de
  visualización* (2021).
- Real Decreto 487/1997 and 488/1997; Ley 31/1995 de Prevención de
  Riesgos Laborales.
- Hignett and McAtamney (2000), *Rapid Entire Body Assessment*; Waters
  et al. (1993), *Revised NIOSH equation for the design and evaluation
  of manual lifting tasks*.
- [MediaPipe Pose and Face Landmarker](https://ai.google.dev/edge/mediapipe/solutions/guide);
  [YOLOX](https://github.com/Megvii-BaseDetection/YOLOX) (Apache-2.0).

### Use of AI

I used an AI assistant (Claude) throughout the project as a pair
programmer: it wrote most of the posture analysis, the web interface and
the tests at my request, proposed the designs that I chose between, and
checked the normative values against the official sources in the corpus.
I recorded the calibration videos, decided every trade-off (what counts
as a lift, how false alarms weigh against missed risks, what is shown to
whom) and reviewed the results against what really happened in each
video.
