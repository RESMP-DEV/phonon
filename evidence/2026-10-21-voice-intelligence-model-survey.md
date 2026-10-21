# Voice-intelligence model survey, 2026-10-21

## Scope

This receipt records a read-only parent survey coordinated through five scoped CCR
workers, plus parent-side live probes. It covers speech-to-speech, speech-to-text,
text-to-speech, and audio-in LLM candidates relevant to Phonon, with emphasis on
models whose language-model component can perform the same kind of correction and
reasoning work as the existing text-only teacher lane. No source code, runtime
configuration, or default engine was changed.

The full worker reports remain ignored under
`build/voice-intel-workers/`; this receipt is the durable aggregate record. It is
sanitized: no secret value, private clip, private transcript, frozen-slice row ID,
owner dictionary value, or screen capture is included.

## Method and live evidence

The parent pinned `main` at `2cea9940467940ef35729ee6dcafef40d00647ab` with a clean
tree before documentation receipt creation. Workers covered local access, xAI/Grok,
hosted APIs, open weights, and Phonon fit. Each finished successfully; the local
access worker was interrupted once and resumed from its saved session before
completion.

Primary parent-side reads included the current Artificial Analysis
speech-to-speech and speech-to-text pages, xAI voice and speech-to-text
documentation, Alibaba Model Studio model catalog/Qwen Omni documentation, and
OpenRouter's model catalog. Live access probes used non-secret account/model
endpoints where credentials existed.

The OpenRouter credential was the only immediately live hosted audio-in inference
route. The parent sent only the synthetic, repository-owned
`assets/startup.wav` fixture to it:

| Probe | Served model | Outcome | Measured elapsed | Reported cost | Notable usage |
| --- | --- | --- | ---:| ---:| --- |
| audio-to-text transcription request | `qwen/qwen3.8-omni-flash` | returned a sensible transcript | 152.31 s | USD 0.0000465003 | 68 reasoning tokens |
| audio-to-text transcription request | `openai/gpt-audio-mini` | returned a slightly different transcript | 150.87 s | USD 0.0000279180 | 11 audio input tokens, no reasoning tokens |

Those requests establish real audio-in capability and billing, but their latency is
far outside interactive dictation. They do not constitute a Phonon quality or
latency gate. No private audio was sent.

## Leaderboard snapshot

Artificial Analysis ranked Grok Voice 3.0 first on its composite Speech-to-Speech
Index: 86.9 overall, 99% Big Bench Audio speech reasoning, 97.0% conversational
dynamics, 71.6% agentic performance, 1064 Arena Elo, 94.8% task success, 0.81 s
time-to-first-audio, and USD 4.80 per input-audio hour on its fixed benchmark
subset. Other leading native speech-to-speech candidates were:

| Candidate | AA speech-to-speech result |
| --- | --- |
| OpenAI GPT-Live-1 (Astra, medium) | 83.2 composite; 90% speech reasoning; 74.5% agentic; 87.4% task success; 1.34 s TTFA |
| Google Gemini 3.8 Live Extended Thinking (High) | 82.6 composite; 98% speech reasoning; 89.1% task success; 1.35 s TTFA |
| Grok Voice 2.0 | 81.3 composite; 97% speech reasoning; 94.6% task success; 0.70 s TTFA |
| OpenAI GPT-Realtime-2.1 High | 75.2 composite; 96% speech reasoning; 91.5% task success; 1.21 s TTFA |
| Google Gemini 3.8 Live | 76.0 composite; 92% speech reasoning; 93.2% task success; 1.18 s TTFA; USD 0.84/hour |
| Eleven Realtime v1 Max | no composite; 99% speech reasoning; 87.1% task success; 0.57 s TTFA |
| Qwen Audio 3.0 Realtime Plus | 69.5 composite; 99% speech reasoning; 77.8% task success; 1.54 s TTFA |

StepAudio 3 Realtime reported 100% speech reasoning and 98.9% conversational
dynamics but 8.83 s TTFA and lacked enough composite components. These numbers are
cross-protocol observations, not evidence on Phonon's private slice.

On AA speech-to-text, Grok Voice Transcribe 2.0 reported 2.3% WER, 186.2x median
speed factor, and USD 1.67 per 1000 minutes. Nearby models included Voxtral Small
Transcribe 3 at 2.1%, MAI-Transcribe-2 at 2.0%, ElevenLabs Scribe v2 at 2.2%,
StepAudio 3 ASR at 2.5%, and Gemini 3.5 Transcribe at 2.6%. Again, this is not a
Phonon-slice result.

## xAI/Grok contract findings

xAI documents `grok-voice-3.0` / `grok-voice-latest` speech-to-speech over
WebSocket with audio and text events, tools, language hints, up to 100 keyterms,
and `reasoning.effort` settable to `high` or `none`. The STT endpoint is
`grok-voice-transcribe-2.0`, with batch REST, streaming WebSocket, word timings,
diarization, multichannel input, raw PCM/mulaw/alaw/Opus, keyterms, and Smart Turn
end-pointing. It is therefore materially more than an acoustic transcoder.

The strict audio-to-JSON path is still two-step: transcribe with Grok Voice, then
apply `json_schema` enforcement on a text model. A realtime Voice session can emit
JSON text under instructions, but xAI does not document schema enforcement there.
General Grok text models do not document audio input. The machine-readable xAI
OpenAPI file also omitted the voice endpoints even though the prose REST/WebSocket
reference documented them.

Privacy is the principal blocker. xAI's general API posture permits encrypted
30-day request/response storage for abuse unless Zero Data Retention is enabled,
while its Voice compliance page says realtime audio is never stored or used for
training. Until a named account proves the operative setting for every voice
surface, Phonon must take the stricter interpretation. Custom voice cloning and
server-side retained reference clips are out of bounds for Phonon's ordinary
dictation plane.

## Current access boundary

| Surface | Classification on this machine |
| --- | --- |
| OpenRouter | Live. Catalog exposed `qwen/qwen3.8-omni-flash`, `openai/gpt-audio`, `openai/gpt-audio-mini`, Mistral Voxtral Small, and a Nemotron Omni reasoning model; synthetic audio requests succeeded |
| Hugging Face | Live account identity and artifact access; no hosted inference claim |
| Alibaba Model Studio / Qwen Omni | Catalog is public, but local Alibaba OAuth credentials expired and no usable DashScope/Model Studio key was found |
| xAI/Grok | Local Grok OAuth token expired at 2026-10-21 05:09:56Z; refresh token exists, but direct API calls returned unauthenticated/403 |
| Google / Gemini | Stored OAuth artifacts exist, but the Gemini CLI reports an ineligible individual tier and its current OAuth token was expired for the parent catalog probe |
| Azure | Stored login artifacts exist, but no live voice-model catalog probe succeeded |
| Deepgram | Keychain key exists but `whoami`/models report an invalid key |
| Direct OpenAI | Codex CLI key exists but api.openai.com rejects it; the configured Codex provider is the local CCR endpoint |
| Kimi, AWS | Stored credentials rejected/expired |
| ElevenLabs, AssemblyAI, Groq | No credential found |

The best privacy contracts found by the hosted survey do not coincide with current
credential availability. AssemblyAI Dictation is the closest product match because
it returns verbatim text and an LLM response separately, supports streaming,
keyterms, and self-serve training opt-out/TTL. Deepgram Voice Agent exposes a
per-request `mip_opt_out=true` zero-retention mode. Google Gemini paid tiers
explicitly exclude model improvement but need configurable logging and paid-tier
verification. Each would require credentials and a recorded provider/data-scope
decision before any private audio request.

## Open-weight candidates

The open-weight worker verified published cards, weights, and source/inference
trees. Its three recommended live-evaluation candidates were:

1. `openbmb/MiniCPM-o-4_5`: Apache-2.0, about 9B total, Qwen3-8B language
   backbone, text/image/audio/video input, speech and text output, 40K context,
   full-duplex streaming claims, about 17.4 GB BF16 and realistic on one 3090 Ti.
2. `Qwen/Qwen3-Omni-30B-A3B-Instruct`: Apache-2.0, 30B total with about 3B
   active parameters, strong audio/video benchmark claims, 64K context, streaming
   Thinker-Talker design; AWQ/GGUF routes make an M4 Max test possible while B550
   can test the larger full-precision route.
3. `mistralai/Voxtral-Mini-4B-Realtime-2602`: Apache-2.0, 4B audio-to-text
   streaming model with configurable 80 ms to 2.4 s delay, 128K vocabulary, and
   MLX/MLX-Audio support; it maps directly to Phonon's streaming-ownership gate.

LFM2.5-Audio remains the existing local SALM lineage, but its measured stock-slice
0.1778 fair WER and current adapters remain far from the Aqua raw baseline. MOSS
Audio 8B and MOSS Transcribe-Diarize 0.9B are additional Apache-2.0 local
contenders. No model was loaded or executed during this survey.

## Phonon decision

Phonon's product path remains local and pinned. Hosted audio-in models are
teachers, evaluators, or explicitly selected experimental engines, never silent
provider settings. Any hosted private-audio run requires a recorded provider,
data-scope, retention, and consent decision. Until then only public or synthetic
audio is allowed. The decisive quality gate is still the full registered 500-row
slice against Aqua raw at 0.032912 fair WER, 0.055232 strict WER, and 0.674 exact,
with per-row win/tie/loss and generation/end-to-end timing. A subset cannot
promote a model.

For open local evaluation, Voxtral Mini Realtime is the streaming-first baseline;
MiniCPM-o-4.5 is the strongest local single-model general candidate; Qwen3-Omni is
the quality ceiling. For hosted research lanes, obtain separate xAI and
AssemblyAI credentials, verify their retention controls, and run synthetic access
probes before requesting any private-audio decision. OpenRouter's current audio
models are useful contract probes but not product-latency candidates.

## Worker receipts

| Piece | SHA-256 | Worker outcome |
| --- | --- | --- |
| local access | `8f79c0c3dd64b843417b68bf93bb4bc83dd19d50027833c15e7fd6d9d208169f` | completed after resume |
| xAI/Grok | `e211bc1fe7c99e62ac5073f3becf4ae5e8d2e0504c141d3467a12c94cc62af69` | completed |
| hosted APIs | `a35d48881f22985a8ca392a214ff3146ddce109a9a2bfe8f5510d1511158c584` | completed |
| open weights | `91872dad2ae621b5d1d6c47d8ef29c7fd1a8aada026e60c6652620f8287086e9` | completed |
| Phonon fit | `5e5babddf5f4cd3f6e60996ca159017e89bbb5aedd3a5a5c75a03c6db3874354` | completed |

The CCR worker run directories are private machine state under
`~/.cache/ccr-worker-runs/`; their `status.json` receipts record successful
completion, routes, output paths, and elapsed times.

## Non-claims

This survey did not run the registered Phonon slice, evaluate end-to-end insertion
latency, load an open-weight model, compare generated speech quality, or prove a
provider's production retention setting. Public and AA benchmarks are explicitly
cross-protocol. No personal dictation audio, owner screen content, private corpus
row, or secret value left the machine or entered Git.
