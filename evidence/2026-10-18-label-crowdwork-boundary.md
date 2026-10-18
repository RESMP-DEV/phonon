# Receipt: transcript crowdwork boundary

Date: 2026-10-18

## Finding

Amazon closed AWS Mechanical Turk effective 2026-09-30 and had already stopped
accepting new requesters before closure. It is therefore not a usable option
even before privacy is considered.

Current commercial alternatives include enterprise labeling vendors such as
Scale and freelance/research marketplaces such as Prolific, but they do not
remove the product boundary: Phonon review clips are the owner's personal
dictation audio and accepted text. Uploading clips plus teacher hypotheses to a
crowdworker pool would disclose the owner's voice and intended text to unknown
third parties. It also cannot resolve personal intent or preferred canonical
formatting.

## Current position

Human labels stay local. The scalable path is active triage:

1. use teacher disagreement to select only high-information segments;
2. present raw and variant hypotheses as one-click drafts;
3. approve the owner-edited canonical text locally;
4. rescore variants against those accepted labels;
5. promote preprocessing only if it improves the SALM gate without narrowing the
   speech band.

The current voice-isolation queue reduces owner review from all 27 native
capture segments to 12 highest-information segments. If even that is excessive,
the next implementation step is a five-segment session and explicit task
resumption, not a third-party upload.

## Non-claims

- No crowd vendor was engaged and no personal data was uploaded.
- Prolific, Scale, Mercor, and similar services were not priced or service-tested
  because the privacy boundary was dispositive for this corpus.
