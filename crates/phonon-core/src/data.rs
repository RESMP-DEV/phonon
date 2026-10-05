use anyhow::{bail, Context, Result};
pub use phonon_llm::PolishConfig;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::cmp::Reverse;
use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

pub const DATA_SCHEMA_VERSION: u32 = 1;
pub const TRAINING_CAPTURE_CONSENT_VERSION: u32 = 1;
pub const SCREENSHOT_FILE_NAME: &str = "screenshot.png";
pub const SCREENSHOT_MANIFEST_FILE_NAME: &str = "screenshot.json";
const DEFAULT_SCREENSHOT_RETENTION_SECONDS: u64 = 24 * 60 * 60;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct DictionaryEntry {
    pub phrase: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub replacement: Option<String>,
    #[serde(default)]
    pub spoken_forms: Vec<String>,
    #[serde(default)]
    pub source: String,
    #[serde(default)]
    pub starred: bool,
    #[serde(default)]
    pub usage_count: u64,
}

impl DictionaryEntry {
    pub fn canonical(&self) -> &str {
        self.replacement.as_deref().unwrap_or(&self.phrase)
    }

    pub fn forms(&self) -> impl Iterator<Item = &str> {
        std::iter::once(self.phrase.as_str()).chain(self.spoken_forms.iter().map(String::as_str))
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct DictionaryFile {
    pub schema_version: u32,
    pub updated_at_unix_ms: u128,
    pub entries: Vec<DictionaryEntry>,
}

impl Default for DictionaryFile {
    fn default() -> Self {
        Self {
            schema_version: DATA_SCHEMA_VERSION,
            updated_at_unix_ms: now_unix_ms(),
            entries: Vec::new(),
        }
    }
}

impl DictionaryFile {
    pub fn load() -> Result<Self> {
        let path = dictionary_path()?;
        if !path.is_file() {
            return Ok(Self::default());
        }
        let bytes = fs::read(&path).with_context(|| format!("read {}", path.display()))?;
        serde_json::from_slice(&bytes).with_context(|| format!("parse {}", path.display()))
    }

    pub fn save(&mut self) -> Result<()> {
        self.schema_version = DATA_SCHEMA_VERSION;
        self.updated_at_unix_ms = now_unix_ms();
        self.entries.sort_by(|a, b| {
            a.phrase
                .to_lowercase()
                .cmp(&b.phrase.to_lowercase())
                .then_with(|| a.replacement.cmp(&b.replacement))
        });
        write_json_atomic(&dictionary_path()?, self)
    }

    pub fn merge(&mut self, incoming: impl IntoIterator<Item = DictionaryEntry>) -> usize {
        let mut entries: BTreeMap<(String, Option<String>), DictionaryEntry> = self
            .entries
            .drain(..)
            .map(|entry| (dictionary_key(&entry), entry))
            .collect();
        let before = entries.len();
        for mut entry in incoming {
            entry.phrase = entry.phrase.trim().to_string();
            entry.replacement = entry
                .replacement
                .map(|value| value.trim().to_string())
                .filter(|value| !value.is_empty() && value != &entry.phrase);
            entry.spoken_forms = normalized_unique(entry.spoken_forms);
            if entry.phrase.is_empty() {
                continue;
            }
            let key = dictionary_key(&entry);
            entries
                .entry(key)
                .and_modify(|existing| {
                    existing.starred |= entry.starred;
                    existing.usage_count = existing.usage_count.max(entry.usage_count);
                    existing.spoken_forms.extend(entry.spoken_forms.clone());
                    existing.spoken_forms = normalized_unique(existing.spoken_forms.clone());
                    if existing.source.is_empty() {
                        existing.source.clone_from(&entry.source);
                    }
                })
                .or_insert(entry);
        }
        self.entries = entries.into_values().collect();
        self.entries.len().saturating_sub(before)
    }

    pub fn relevant_entries(&self, transcript: &str) -> Vec<&DictionaryEntry> {
        let normalized = normalize(transcript);
        let transcript_words: Vec<&str> = normalized.split_whitespace().collect();
        let mut matches: Vec<(&DictionaryEntry, u64)> = self
            .entries
            .iter()
            .filter_map(|entry| {
                let mut best = 0_u64;
                for form in entry.forms() {
                    let form = normalize(form);
                    if form.is_empty() {
                        continue;
                    }
                    if contains_phrase(&normalized, &form) {
                        best = best.max(1_000_000);
                        continue;
                    }
                    let form_words: Vec<&str> = form.split_whitespace().collect();
                    if form_words.len() == 1 {
                        for word in &transcript_words {
                            if plausibly_same_word(word, form_words[0]) {
                                best = best.max(500_000);
                            }
                        }
                        let form_key = phonetic_key(&form);
                        if form_key.len() >= 3 {
                            for window_len in 1..=3.min(transcript_words.len()) {
                                for window in transcript_words.windows(window_len) {
                                    if phonetic_key(&window.join(" ")) == form_key {
                                        best = best.max(750_000);
                                    }
                                }
                            }
                        }
                    }
                }
                (best > 0).then_some((entry, best + entry.usage_count))
            })
            .collect();
        matches.sort_by(|(a, a_score), (b, b_score)| {
            b_score
                .cmp(a_score)
                .then_with(|| b.starred.cmp(&a.starred))
                .then_with(|| a.phrase.cmp(&b.phrase))
        });
        matches.into_iter().map(|(entry, _)| entry).collect()
    }

    pub fn prepare_polish_input(&self, transcript: &str) -> String {
        self.prepare_polish_input_with_context(transcript, "")
    }

    pub fn prepare_polish_input_with_context(&self, transcript: &str, screen_text: &str) -> String {
        if self.entries.is_empty() {
            return transcript.to_string();
        }
        let likely_entries = self.relevant_entries(transcript);
        let normalized_screen = normalize(screen_text);
        let exact_corrected = self.apply_exact_replacements(transcript).text;
        let phonetic_matches = self.phonetic_matches(&exact_corrected);
        let mut pre_corrected = exact_corrected;
        for (spoken, canonical) in &phonetic_matches {
            pre_corrected = replace_ascii_phrase(&pre_corrected, spoken, canonical).0;
        }
        let terms = likely_entries
            .iter()
            .filter(|entry| entry.replacement.is_none())
            .map(|entry| entry.canonical())
            .collect::<Vec<_>>();
        let likely_terms = likely_entries
            .iter()
            .map(|entry| entry.canonical())
            .collect::<Vec<_>>();
        let replacements = likely_entries
            .iter()
            .filter_map(|entry| {
                entry
                    .replacement
                    .as_ref()
                    .map(|replacement| format!("{} => {replacement}", entry.phrase))
            })
            .collect::<Vec<_>>();
        let screen_terms = self
            .screen_confirmed_entries(transcript, screen_text)
            .iter()
            .filter(|entry| contains_phrase(&normalized_screen, &normalize(entry.canonical())))
            .map(|entry| entry.canonical())
            .collect::<Vec<_>>();
        let phonetic_matches = phonetic_matches
            .iter()
            .map(|(spoken, canonical)| format!("{spoken} => {canonical}"))
            .collect::<Vec<_>>();
        format!(
            "<phonon_dictionary>\ncanonical_terms: {}\nlikely_terms: {}\nscreen_confirmed_terms: {}\nphonetic_matches: {}\nexact_replacements: {}\n</phonon_dictionary>\n<transcript>\n{}\n</transcript>",
            terms.join(", "),
            likely_terms.join(", "),
            screen_terms.join(", "),
            phonetic_matches.join("; "),
            replacements.join("; "),
            pre_corrected.trim()
        )
    }

    /// Phonetic pre-correction is deliberately blind to meaning: it matches on a
    /// vowel-free consonant skeleton, so `please` and `pallas`, or `profile` and
    /// `prefill`, are indistinguishable. Guessing is only acceptable on tokens
    /// the speech model is unlikely to have got right in the first place, so an
    /// ordinary English word is never replaced. A term the user actually taught
    /// is an exact replacement and does not come through here.
    fn phonetic_matches(&self, transcript: &str) -> Vec<(String, String)> {
        // Key -> the canonical spelling plus the word counts it was indexed
        // under. Both matter. Indexing only single-word forms left every
        // multi-word term unreachable, and the counts bound how far a match may
        // stretch: with vowels dropped, `koo bloss is` keys the same as `koo
        // bloss`, so an unbounded window absorbs the next word and deletes it.
        let mut by_key: BTreeMap<String, Option<(String, BTreeSet<usize>)>> = BTreeMap::new();
        for entry in &self.entries {
            for form in entry.forms() {
                let form_words = normalize(form).split_whitespace().count();
                if form_words == 0 || form_words > MAX_PHONETIC_WINDOW {
                    // Longer than any window the scan below builds, so a key
                    // for it could never be matched.
                    continue;
                }
                let key = phonetic_key(form);
                if key.len() < 3 {
                    continue;
                }
                let canonical = entry.canonical().to_string();
                by_key
                    .entry(key)
                    .and_modify(|existing| {
                        match existing {
                            // `black well` and `blackwell` are one term written
                            // two ways, not an ambiguity. Keep whichever spells
                            // the capitals, since restoring them is the point.
                            Some((known, counts)) if same_term(known, &canonical) => {
                                counts.insert(form_words);
                                if capital_count(&canonical) > capital_count(known) {
                                    *known = canonical.clone();
                                }
                            }
                            // Two genuinely different terms sound alike; neither
                            // can be chosen safely, so drop the key entirely.
                            _ => *existing = None,
                        }
                    })
                    .or_insert_with(|| Some((canonical, BTreeSet::from([form_words]))));
            }
        }

        let normalized = normalize(transcript);
        let words = normalized.split_whitespace().collect::<Vec<_>>();
        let mut occupied = vec![false; words.len()];
        let mut matches = Vec::new();
        for window_len in (1..=MAX_PHONETIC_WINDOW.min(words.len())).rev() {
            for start in 0..=words.len() - window_len {
                if occupied[start..start + window_len]
                    .iter()
                    .any(|value| *value)
                {
                    continue;
                }
                let spoken = words[start..start + window_len].join(" ");
                let Some(Some((canonical, form_words))) = by_key.get(&phonetic_key(&spoken)) else {
                    continue;
                };
                // A recognizer splits or joins a term by one word, writing
                // `black well` for `Blackwell`. It does not fuse a term with
                // two of its neighbours, so a window that far from the taught
                // form is a key collision rather than a mishearing.
                if !form_words
                    .iter()
                    .any(|count| window_len.abs_diff(*count) <= 1)
                {
                    continue;
                }
                // A word the key cannot represent at all -- `a`, `I` -- is free
                // to absorb, and absorbing it deletes it from the transcript.
                // Every word in the window has to contribute a sound.
                if words[start..start + window_len]
                    .iter()
                    .any(|word| phonetic_key(word).is_empty())
                {
                    continue;
                }
                if normalize(canonical) == spoken {
                    continue;
                }
                if is_ordinary_english(&spoken) {
                    continue;
                }
                // A window longer than the taught form is either the recognizer
                // splitting the term into syllables, or the term reaching out and
                // swallowing the word beside it. Syllables are not words, so any
                // real English in an over-long window means the latter: `is not`
                // becoming `Sonnet`, or `my macbook` losing the `my`.
                let absorbing = form_words.iter().all(|count| window_len > *count);
                if absorbing
                    && words[start..start + window_len]
                        .iter()
                        .any(|word| is_real_word(word))
                {
                    continue;
                }
                occupied[start..start + window_len].fill(true);
                matches.push((spoken, canonical.clone()));
            }
        }
        matches
    }

    pub fn screen_confirmed_terms(&self, transcript: &str, screen_text: &str) -> Vec<String> {
        self.screen_confirmed_entries(transcript, screen_text)
            .into_iter()
            .map(|entry| entry.canonical().to_string())
            .collect()
    }

    fn screen_confirmed_entries(
        &self,
        transcript: &str,
        screen_text: &str,
    ) -> Vec<&DictionaryEntry> {
        let normalized_screen = normalize(screen_text);
        self.entries
            .iter()
            .filter(|entry| {
                contains_phrase(&normalized_screen, &normalize(entry.canonical()))
                    && entry
                        .forms()
                        .any(|form| contextually_similar_phrase(transcript, form))
            })
            .collect()
    }

    pub fn apply_exact_replacements(&self, text: &str) -> CorrectionResult {
        let mut output = text.to_string();
        let mut applied = Vec::new();
        for entry in &self.entries {
            let Some(replacement) = entry.replacement.as_deref() else {
                continue;
            };
            for form in entry.forms() {
                let (next, count) = replace_ascii_phrase(&output, form, replacement);
                if count > 0 {
                    output = next;
                    applied.push(AppliedCorrection {
                        from: form.to_string(),
                        to: replacement.to_string(),
                        count,
                    });
                }
            }
        }
        CorrectionResult {
            text: output,
            applied,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct AppliedCorrection {
    pub from: String,
    pub to: String,
    pub count: u64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CorrectionResult {
    pub text: String,
    pub applied: Vec<AppliedCorrection>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
pub struct LlmMetadata {
    pub latency_ms: f64,
    pub ttft_ms: f64,
    pub tokens_per_second: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct PixelDimensions {
    pub width: u32,
    pub height: u32,
}

/// A retained screen image is never a default by-product. Every field is part
/// of the consent contract: what was captured, from which display, how long it
/// may live, and which consent statement authorized it.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ScreenImageCapture {
    pub file: String,
    pub sha256: String,
    pub display_id: String,
    pub pixel_dimensions: PixelDimensions,
    pub scale_factor: f64,
    pub captured_at_unix_ms: u128,
    pub capture_origin: String,
    pub deletion_deadline_unix_ms: u128,
    pub consent_version: u32,
    pub consented_at_unix_ms: u128,
    /// Deliberately local-only: this records what the image may be used for,
    /// not a destination. Export remains a separate explicit user action.
    pub capability: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
struct ScreenImageManifest {
    schema_version: u32,
    recording_id: String,
    capture: ScreenImageCapture,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct RecordingMetadata {
    pub schema_version: u32,
    pub id: String,
    pub created_at_unix_ms: u128,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub recorded_at: Option<String>,
    pub source: String,
    pub audio_file: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub microphone: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub audio_duration_ms: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub speech_detected: Option<bool>,
    #[serde(default)]
    pub raw_transcript: String,
    #[serde(default)]
    pub final_transcript: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub intended_transcript: Option<String>,
    #[serde(default)]
    pub dictionary_corrections: Vec<AppliedCorrection>,
    #[serde(default)]
    pub screen_context_terms: Vec<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub llm: Option<LlmMetadata>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub screen_image: Option<ScreenImageCapture>,
}

impl RecordingMetadata {
    fn new(id: String, source: &str, audio_file: String) -> Self {
        Self {
            schema_version: DATA_SCHEMA_VERSION,
            id,
            created_at_unix_ms: now_unix_ms(),
            recorded_at: None,
            source: source.to_string(),
            audio_file,
            microphone: None,
            audio_duration_ms: None,
            speech_detected: None,
            raw_transcript: String::new(),
            final_transcript: String::new(),
            intended_transcript: None,
            dictionary_corrections: Vec::new(),
            screen_context_terms: Vec::new(),
            llm: None,
            screen_image: None,
        }
    }
}

#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
pub struct UsageStats {
    pub recordings: u64,
    pub words: u64,
    pub speaking_ms: u64,
    pub dictionary_fixes: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct SettingsFile {
    pub schema_version: u32,
    #[serde(default = "default_true")]
    pub streaming: bool,
    #[serde(default = "default_true")]
    pub local_history: bool,
    #[serde(default = "default_true")]
    pub screen_context: bool,
    #[serde(default = "default_microphone_priority")]
    pub microphone_priority: Vec<String>,
    #[serde(default = "default_true")]
    pub instant_mic: bool,
    #[serde(default = "default_shortcut_mode")]
    pub shortcut_mode: String,
    /// Bundle IDs of competing dictation apps the owner told the native app not
    /// to ask about again. Owned by the Swift bar; kept here so a rewrite of
    /// the file by the CLI does not drop it.
    #[serde(default)]
    pub competitor_quit_muted: Vec<String>,
    /// Put `profile/user.md` and `profile/vocab.md` in the correction prompt.
    #[serde(default = "default_profile_prefix")]
    pub profile_prefix: bool,
    #[serde(default)]
    pub training_capture_enabled: bool,
    #[serde(default)]
    pub include_screen_images: bool,
    #[serde(default = "default_screenshot_retention_seconds")]
    pub screenshot_retention_seconds: u64,
}

impl Default for SettingsFile {
    fn default() -> Self {
        Self {
            schema_version: DATA_SCHEMA_VERSION,
            streaming: true,
            local_history: true,
            screen_context: true,
            microphone_priority: default_microphone_priority(),
            instant_mic: true,
            shortcut_mode: default_shortcut_mode(),
            competitor_quit_muted: Vec::new(),
            profile_prefix: default_profile_prefix(),
            training_capture_enabled: false,
            include_screen_images: false,
            screenshot_retention_seconds: DEFAULT_SCREENSHOT_RETENTION_SECONDS,
        }
    }
}

/// Off until the profile block earns it. Measured 2026-08-30 with
/// `scripts/eval_profile_prefix.py` (24 labeled cases, 80 corpus recordings):
/// no latency cost warm, but 21/24 exact against 22/24 without it, and on the
/// corpus its vocabulary wins (Phonon, DEVLOG, Blackwell) came with two
/// dropped sentences and one wrong term. Users can turn it on in settings.json.
pub const DEFAULT_PROFILE_PREFIX: bool = false;

fn default_profile_prefix() -> bool {
    DEFAULT_PROFILE_PREFIX
}

fn default_screenshot_retention_seconds() -> u64 {
    DEFAULT_SCREENSHOT_RETENTION_SECONDS
}

fn default_true() -> bool {
    true
}

fn default_microphone_priority() -> Vec<String> {
    Vec::new()
}

fn default_shortcut_mode() -> String {
    "fn".into()
}

impl SettingsFile {
    /// Read without writing back. Missing file means defaults; the native app
    /// owns the file and this side must not rewrite it on every engine start.
    pub fn load() -> Result<Self> {
        let path = settings_path()?;
        if !path.is_file() {
            return Ok(Self::default());
        }
        Self::load_at(path)
    }

    pub fn load_or_create() -> Result<Self> {
        let path = settings_path()?;
        if path.is_file() {
            let bytes = fs::read(&path).with_context(|| format!("read {}", path.display()))?;
            let mut settings: Self = serde_json::from_slice(&bytes)
                .with_context(|| format!("parse {}", path.display()))?;
            settings.schema_version = DATA_SCHEMA_VERSION;
            write_json_atomic(&path, &settings)?;
            return Ok(settings);
        }
        let settings = Self::default();
        write_json_atomic(&path, &settings)?;
        Ok(settings)
    }

    pub fn load_at(path: PathBuf) -> Result<Self> {
        if !path.is_file() {
            return Ok(Self::default());
        }
        let bytes = fs::read(&path).with_context(|| format!("read {}", path.display()))?;
        serde_json::from_slice(&bytes).with_context(|| format!("parse {}", path.display()))
    }
}

pub fn app_support_dir() -> Result<PathBuf> {
    let home = std::env::var_os("HOME").context("HOME is not set")?;
    Ok(PathBuf::from(home)
        .join("Library")
        .join("Application Support")
        .join("Phonon"))
}

pub fn dictionary_path() -> Result<PathBuf> {
    Ok(app_support_dir()?.join("dictionary.json"))
}

pub fn settings_path() -> Result<PathBuf> {
    Ok(app_support_dir()?.join("settings.json"))
}

fn settings_path_at(app_support: &Path) -> PathBuf {
    app_support.join("settings.json")
}

pub fn corpus_dir() -> Result<PathBuf> {
    Ok(app_support_dir()?.join("Corpus"))
}

fn corpus_dir_at(app_support: &Path) -> PathBuf {
    app_support.join("Corpus")
}

fn recording_dir_at(app_support: &Path, id: &str) -> Result<PathBuf> {
    validate_recording_id(id)?;
    Ok(corpus_dir_at(app_support).join(id))
}

fn validate_recording_id(id: &str) -> Result<()> {
    if id.is_empty() || id.contains('/') || id.contains('\\') || id == "." || id == ".." {
        bail!("invalid recording id: {id:?}");
    }
    Ok(())
}

fn save_recording_at(app_support: &Path, metadata: &RecordingMetadata) -> Result<()> {
    let directory = recording_dir_at(app_support, &metadata.id)?;
    write_json_atomic(&directory.join("metadata.json"), metadata)
}

fn load_recording_at(path: &Path) -> Result<RecordingMetadata> {
    let bytes = fs::read(path).with_context(|| format!("read {}", path.display()))?;
    serde_json::from_slice(&bytes).with_context(|| format!("parse {}", path.display()))
}

fn screenshot_manifest_path(recording_directory: &Path) -> PathBuf {
    recording_directory.join(SCREENSHOT_MANIFEST_FILE_NAME)
}

fn save_screenshot_manifest(
    app_support: &Path,
    metadata: &RecordingMetadata,
    capture: &ScreenImageCapture,
) -> Result<()> {
    let manifest = ScreenImageManifest {
        schema_version: DATA_SCHEMA_VERSION,
        recording_id: metadata.id.clone(),
        capture: capture.clone(),
    };
    let directory = recording_dir_at(app_support, &metadata.id)?;
    write_json_atomic(&screenshot_manifest_path(&directory), &manifest)
}

fn load_screenshot_manifest(recording_directory: &Path) -> Result<ScreenImageManifest> {
    let bytes = fs::read(screenshot_manifest_path(recording_directory)).with_context(|| {
        format!(
            "read {}",
            screenshot_manifest_path(recording_directory).display()
        )
    })?;
    serde_json::from_slice(&bytes).with_context(|| {
        format!(
            "parse {}",
            screenshot_manifest_path(recording_directory).display()
        )
    })
}

fn sha256_hex(bytes: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(bytes);
    format!("{:x}", hasher.finalize())
}

fn atomic_copy(source: &Path, target: &Path) -> Result<()> {
    let parent = target
        .parent()
        .with_context(|| format!("target has no parent: {}", target.display()))?;
    fs::create_dir_all(parent)?;
    let temporary = target.with_extension(format!(
        "{}.tmp-{}-{}",
        target
            .extension()
            .and_then(|extension| extension.to_str())
            .unwrap_or("part"),
        std::process::id(),
        now_unix_ms()
    ));
    let result = fs::copy(source, &temporary).and_then(|_| fs::rename(&temporary, target));
    if result.is_err() {
        let _ = fs::remove_file(&temporary);
    }
    result.with_context(|| format!("copy {} to {}", source.display(), target.display()))
}
/// `profile/user.md` and `profile/vocab.md` live here (SPEC, onboarding).
pub fn profile_dir() -> Result<PathBuf> {
    Ok(app_support_dir()?.join("profile"))
}

/// The correction sidecar configuration users get: the profile when
/// `settings.json` enables it, plus developer environment overrides.
pub fn polish_config() -> Result<PolishConfig> {
    let settings = SettingsFile::load()?;
    let profile_dir = profile_dir()?;
    let config = PolishConfig {
        profile_dir: settings.profile_prefix.then(|| profile_dir.clone()),
        ..PolishConfig::default()
    };
    Ok(config.with_env_overrides(Some(&profile_dir)))
}

pub fn metadata_path_for_audio(audio_path: &Path) -> Result<PathBuf> {
    let parent = audio_path
        .parent()
        .with_context(|| format!("audio path has no parent: {}", audio_path.display()))?;
    Ok(parent.join("metadata.json"))
}

pub fn register_recording(audio_path: &Path, source: &str) -> Result<RecordingMetadata> {
    let parent = audio_path
        .parent()
        .with_context(|| format!("audio path has no parent: {}", audio_path.display()))?;
    fs::create_dir_all(parent)?;
    let id = parent
        .file_name()
        .and_then(|value| value.to_str())
        .context("recording directory has no UTF-8 id")?
        .to_string();
    let path = parent.join("metadata.json");
    if path.is_file() {
        return load_recording(&path);
    }
    let audio_file = audio_path
        .file_name()
        .and_then(|value| value.to_str())
        .unwrap_or("audio.wav")
        .to_string();
    let mut metadata = RecordingMetadata::new(id, source, audio_file);
    metadata.audio_duration_ms = wav_duration_ms(audio_path).ok();
    save_recording(&metadata)?;
    Ok(metadata)
}

pub fn import_recording(
    audio_path: &Path,
    id: &str,
    source: &str,
    final_transcript: &str,
    recorded_at: Option<String>,
) -> Result<(RecordingMetadata, bool)> {
    if id.is_empty() || id.contains('/') || id.contains('\\') || id == "." || id == ".." {
        bail!("invalid recording id: {id:?}");
    }
    if !audio_path.is_file() {
        bail!("audio file not found: {}", audio_path.display());
    }
    let directory = corpus_dir()?.join(id);
    let metadata_path = directory.join("metadata.json");
    if metadata_path.is_file() {
        return Ok((load_recording(&metadata_path)?, false));
    }
    fs::create_dir_all(&directory)?;
    let target = directory.join("audio.wav");
    fs::copy(audio_path, &target)
        .with_context(|| format!("copy {} to {}", audio_path.display(), target.display()))?;
    let mut metadata = RecordingMetadata::new(id.to_string(), source, "audio.wav".into());
    metadata.recorded_at = recorded_at;
    metadata.audio_duration_ms = wav_duration_ms(&target).ok();
    metadata.final_transcript = final_transcript.trim().to_string();
    save_recording(&metadata)?;
    Ok((metadata, true))
}

pub fn load_recording(path: &Path) -> Result<RecordingMetadata> {
    let bytes = fs::read(path).with_context(|| format!("read {}", path.display()))?;
    serde_json::from_slice(&bytes).with_context(|| format!("parse {}", path.display()))
}

pub fn save_recording(metadata: &RecordingMetadata) -> Result<()> {
    let path = corpus_dir()?.join(&metadata.id).join("metadata.json");
    write_json_atomic(&path, metadata)
}

pub fn set_speech_detected(id: &str, detected: bool) -> Result<()> {
    let mut metadata = load_recording_by_id(id)?;
    metadata.speech_detected = Some(detected);
    save_recording(&metadata)
}

pub fn list_recordings() -> Result<Vec<RecordingMetadata>> {
    let directory = corpus_dir()?;
    if !directory.is_dir() {
        return Ok(Vec::new());
    }
    let mut recordings = Vec::new();
    for entry in fs::read_dir(&directory)? {
        let entry = entry?;
        let path = entry.path().join("metadata.json");
        if path.is_file() {
            recordings.push(load_recording(&path)?);
        }
    }
    recordings.sort_by_key(|recording| Reverse(recording.created_at_unix_ms));
    Ok(recordings)
}

pub fn load_recording_by_id(id: &str) -> Result<RecordingMetadata> {
    let path = corpus_dir()?.join(id).join("metadata.json");
    if !path.is_file() {
        bail!("recording not found: {id}");
    }
    load_recording(&path)
}

pub fn usage_stats() -> Result<UsageStats> {
    let mut stats = UsageStats::default();
    for recording in list_recordings()? {
        stats.recordings += 1;
        let text = if recording.final_transcript.is_empty() {
            &recording.raw_transcript
        } else {
            &recording.final_transcript
        };
        stats.words += text.split_whitespace().count() as u64;
        stats.speaking_ms += recording.audio_duration_ms.unwrap_or(0);
        stats.dictionary_fixes += recording
            .dictionary_corrections
            .iter()
            .map(|correction| correction.count)
            .sum::<u64>();
    }
    Ok(stats)
}

pub fn set_intended_transcript(id: &str, intended: &str) -> Result<RecordingMetadata> {
    let mut metadata = load_recording_by_id(id)?;
    metadata.intended_transcript = Some(intended.trim().to_string());
    save_recording(&metadata)?;
    Ok(metadata)
}

pub fn update_recording(
    id: &str,
    update: impl FnOnce(&mut RecordingMetadata) -> Result<()>,
) -> Result<RecordingMetadata> {
    update_recording_at(&app_support_dir()?, id, update)
}

pub fn update_recording_at(
    app_support: &Path,
    id: &str,
    update: impl FnOnce(&mut RecordingMetadata) -> Result<()>,
) -> Result<RecordingMetadata> {
    let directory = recording_dir_at(app_support, id)?;
    let mut metadata = load_recording_at(&directory.join("metadata.json"))?;
    update(&mut metadata)?;
    save_recording_at(app_support, &metadata)?;
    Ok(metadata)
}

#[derive(Debug, Clone, PartialEq)]
pub struct ScreenImageCaptureRequest {
    pub display_id: String,
    pub pixel_width: u32,
    pub pixel_height: u32,
    pub scale_factor: f64,
    pub capture_origin: String,
    pub captured_at_unix_ms: Option<u128>,
    pub consented_at_unix_ms: Option<u128>,
    pub screen_capture_permission_granted: bool,
}

fn register_or_load_recording_at(
    app_support: &Path,
    audio_path: &Path,
    source: &str,
) -> Result<RecordingMetadata> {
    let parent = audio_path
        .parent()
        .with_context(|| format!("audio path has no parent: {}", audio_path.display()))?;
    fs::create_dir_all(parent)?;
    let id = parent
        .file_name()
        .and_then(|value| value.to_str())
        .context("recording directory has no UTF-8 id")?;
    validate_recording_id(id)?;
    let expected_parent = corpus_dir_at(app_support).join(id);
    if parent != expected_parent {
        bail!(
            "audio path is outside the corpus recording directory: {}",
            audio_path.display()
        );
    }
    let path = parent.join("metadata.json");
    if path.is_file() {
        return load_recording_at(&path);
    }
    let audio_file = audio_path
        .file_name()
        .and_then(|value| value.to_str())
        .unwrap_or("audio.wav")
        .to_string();
    let mut metadata = RecordingMetadata::new(id.to_string(), source, audio_file);
    metadata.audio_duration_ms = wav_duration_ms(audio_path).ok();
    save_recording_at(app_support, &metadata)?;
    Ok(metadata)
}

pub fn register_screen_image(
    audio_path: &Path,
    image_path: &Path,
    request: ScreenImageCaptureRequest,
    source: &str,
) -> Result<RecordingMetadata> {
    register_screen_image_at(&app_support_dir()?, audio_path, image_path, request, source)
}

pub fn register_screen_image_at(
    app_support: &Path,
    audio_path: &Path,
    image_path: &Path,
    request: ScreenImageCaptureRequest,
    source: &str,
) -> Result<RecordingMetadata> {
    let settings = SettingsFile::load_at(settings_path_at(app_support))?;
    require_screen_capture_scope(&settings)?;
    if !request.screen_capture_permission_granted {
        bail!("screen capture permission is not granted");
    }
    if request.display_id.trim().is_empty() || request.capture_origin.trim().is_empty() {
        bail!("screen-image provenance is incomplete");
    }
    if request.pixel_width == 0 || request.pixel_height == 0 || request.scale_factor <= 0.0 {
        bail!("screen-image dimensions are invalid");
    }
    if !audio_path.is_file() {
        bail!("audio file not found: {}", audio_path.display());
    }
    if !image_path.is_file() {
        bail!("screen image not found: {}", image_path.display());
    }
    let image = fs::read(image_path)
        .with_context(|| format!("read screen image {}", image_path.display()))?;
    if image.len() < 8 || image[..8] != [0x89, b'P', b'N', b'G', 0x0d, 0x0a, 0x1a, 0x0a] {
        bail!("screen image is not a PNG");
    }
    let sha256 = sha256_hex(&image);
    let now = now_unix_ms();
    let captured_at = request.captured_at_unix_ms.unwrap_or(now);
    let consented_at = request.consented_at_unix_ms.unwrap_or(now);
    let retention_ms = u128::from(settings.screenshot_retention_seconds) * 1_000;
    let deletion_deadline = captured_at
        .checked_add(retention_ms)
        .context("screenshot deletion deadline overflow")?;
    let mut metadata = register_or_load_recording_at(app_support, audio_path, source)?;
    let directory = corpus_dir_at(app_support).join(&metadata.id);
    let target = directory.join(SCREENSHOT_FILE_NAME);
    let manifest_path = screenshot_manifest_path(&directory);
    if metadata.screen_image.is_some() || target.exists() || manifest_path.exists() {
        bail!("recording {} already has a screen image", metadata.id);
    }
    let attach_result = atomic_copy(image_path, &target).and_then(|()| {
        let capture = ScreenImageCapture {
            file: SCREENSHOT_FILE_NAME.to_string(),
            sha256,
            display_id: request.display_id.trim().to_string(),
            pixel_dimensions: PixelDimensions {
                width: request.pixel_width,
                height: request.pixel_height,
            },
            scale_factor: request.scale_factor,
            captured_at_unix_ms: captured_at,
            capture_origin: request.capture_origin.trim().to_string(),
            deletion_deadline_unix_ms: deletion_deadline,
            consent_version: TRAINING_CAPTURE_CONSENT_VERSION,
            consented_at_unix_ms: consented_at,
            capability: "local_training_corpus_image".to_string(),
        };
        metadata.screen_image = Some(capture.clone());
        save_screenshot_manifest(app_support, &metadata, &capture)?;
        save_recording_at(app_support, &metadata)
    });
    if let Err(error) = attach_result {
        let _ = fs::remove_file(&target);
        let _ = fs::remove_file(&manifest_path);
        return Err(error);
    }
    Ok(metadata)
}

fn require_screen_capture_scope(settings: &SettingsFile) -> Result<()> {
    if !settings.training_capture_enabled {
        bail!("training capture is not enabled");
    }
    if !settings.include_screen_images {
        bail!("screen-image capture is not enabled");
    }
    if settings.screenshot_retention_seconds == 0 {
        bail!("screenshot retention is missing or zero");
    }
    Ok(())
}

fn verify_screen_image_at(
    directory: &Path,
    metadata: &RecordingMetadata,
) -> Result<ScreenImageCapture> {
    let capture = metadata
        .screen_image
        .clone()
        .context(format!("recording {} has no screen image", metadata.id))?;
    if capture.file != SCREENSHOT_FILE_NAME {
        bail!(
            "invalid screen-image file name for recording {}",
            metadata.id
        );
    }
    let image_path = directory.join(&capture.file);
    let image = fs::read(&image_path)
        .with_context(|| format!("read screen image {}", image_path.display()))?;
    if image.len() < 8 || image[..8] != [0x89, b'P', b'N', b'G', 0x0d, 0x0a, 0x1a, 0x0a] {
        bail!("screen image is not a PNG for recording {}", metadata.id);
    }
    let actual_hash = sha256_hex(&image);
    if actual_hash != capture.sha256 {
        bail!("screenshot hash mismatch for recording {}", metadata.id);
    }
    let manifest = load_screenshot_manifest(directory)
        .with_context(|| format!("verify screen-image manifest for {}", metadata.id))?;
    if manifest.schema_version != DATA_SCHEMA_VERSION
        || manifest.recording_id != metadata.id
        || manifest.capture != capture
    {
        bail!("screenshot manifest mismatch for recording {}", metadata.id);
    }
    Ok(capture)
}

pub fn delete_expired_screenshots(now_unix_ms: u128) -> Result<Vec<String>> {
    delete_expired_screenshots_at(&app_support_dir()?, now_unix_ms)
}

pub fn delete_expired_screenshots_at(app_support: &Path, now_unix_ms: u128) -> Result<Vec<String>> {
    let corpus = corpus_dir_at(app_support);
    if !corpus.is_dir() {
        return Ok(Vec::new());
    }
    let mut deleted = Vec::new();
    for entry in fs::read_dir(&corpus).with_context(|| format!("read {}", corpus.display()))? {
        let entry = entry?;
        let metadata_path = entry.path().join("metadata.json");
        if !metadata_path.is_file() {
            continue;
        }
        let mut metadata = load_recording_at(&metadata_path)?;
        let Some(capture) = metadata.screen_image.clone() else {
            continue;
        };
        if capture.deletion_deadline_unix_ms > now_unix_ms {
            continue;
        }
        verify_screen_image_at(&entry.path(), &metadata)?;
        let image_path = entry.path().join(&capture.file);
        let manifest_path = screenshot_manifest_path(&entry.path());
        fs::remove_file(&image_path).with_context(|| format!("delete {}", image_path.display()))?;
        fs::remove_file(&manifest_path)
            .with_context(|| format!("delete {}", manifest_path.display()))?;
        metadata.screen_image = None;
        if let Err(error) = save_recording_at(app_support, &metadata) {
            return Err(error.context(format!(
                "screenshot deleted, but metadata update failed for {}",
                metadata.id
            )));
        }
        deleted.push(metadata.id);
    }
    Ok(deleted)
}

pub fn extract_transcript_payload(output: &str) -> &str {
    let Some(start) = output.find("<transcript>") else {
        let trimmed = output.trim();
        if trimmed.contains("<phonon_dictionary")
            || trimmed.contains("</phonon_dictionary>")
            || trimmed.contains("canonical_terms:")
        {
            return "";
        }
        return trimmed;
    };
    let content_start = start + "<transcript>".len();
    let remaining = &output[content_start..];
    let end = remaining.find("</transcript>").unwrap_or(remaining.len());
    remaining[..end].trim()
}

pub fn safe_polish_output(
    dictionary: &DictionaryFile,
    source: &str,
    polished: &str,
) -> CorrectionResult {
    let polished = extract_transcript_payload(polished);
    let source_corrected = dictionary.apply_exact_replacements(source);
    let polished_corrected = dictionary.apply_exact_replacements(polished);
    let source_words = normalized_words(&source_corrected.text);
    let polished_words = normalized_words(&polished_corrected.text);
    let collapsed =
        polished.trim().is_empty() || (source_words.len() > 1 && polished_words.len() <= 1);
    let implausible_expansion = polished_words.len() > source_words.len().saturating_mul(2) + 8
        || polished.len() > source.len().saturating_mul(4) + 64;
    let meaning_only_deletion = source_words.len().saturating_sub(polished_words.len()) <= 2
        && deletes_meaningful_words_only(&source_words, &polished_words);
    if collapsed || implausible_expansion || meaning_only_deletion {
        source_corrected
    } else {
        polished_corrected
    }
}

fn normalized_words(text: &str) -> Vec<String> {
    text.split_whitespace()
        .map(|word| {
            word.chars()
                .filter(|character| character.is_alphanumeric())
                .flat_map(char::to_lowercase)
                .collect()
        })
        .filter(|word: &String| !word.is_empty())
        .collect()
}

fn deletes_meaningful_words_only(source: &[String], candidate: &[String]) -> bool {
    if candidate.len() >= source.len() {
        return false;
    }
    let mut candidate_index = 0;
    let mut removed = Vec::new();
    for (source_index, word) in source.iter().enumerate() {
        if candidate.get(candidate_index) == Some(word) {
            candidate_index += 1;
        } else {
            removed.push(source_index);
        }
    }
    if candidate_index != candidate.len() {
        return false;
    }
    removed.into_iter().any(|index| {
        let word = source[index].as_str();
        let filler = matches!(word, "uh" | "um" | "erm" | "hmm");
        let duplicate = index > 0 && source[index - 1] == source[index]
            || index + 1 < source.len() && source[index + 1] == source[index];
        !filler && !duplicate
    })
}

pub fn delete_recording(id: &str) -> Result<()> {
    delete_recording_at(&app_support_dir()?, id)
}

pub fn delete_recording_at(app_support: &Path, id: &str) -> Result<()> {
    validate_recording_id(id)?;
    let directory = corpus_dir_at(app_support).join(id);
    let metadata = directory.join("metadata.json");
    if !metadata.is_file() {
        bail!("recording not found: {id}");
    }
    let quarantine = corpus_dir_at(app_support).join(format!(
        ".deleting-{id}-{}-{}",
        std::process::id(),
        now_unix_ms()
    ));
    fs::rename(&directory, &quarantine)
        .with_context(|| format!("quarantine {}", directory.display()))?;
    if let Err(error) = fs::remove_dir_all(&quarantine) {
        let restore = fs::rename(&quarantine, &directory);
        if let Err(restore_error) = restore {
            return Err(anyhow::anyhow!(
                "delete failed: {error:#}; restore failed: {restore_error:#}"
            ))
            .context(format!("delete {}", directory.display()));
        }
        return Err(error).with_context(|| format!("delete {}", directory.display()));
    }
    Ok(())
}

#[derive(Debug, Clone, Default)]
pub struct CorpusExportOptions {
    pub include_screen_images: bool,
    pub only_consented: bool,
    pub include_text: bool,
}

#[derive(Debug, Clone, Serialize)]
struct CorpusExportItem {
    session_id: String,
    audio_sha256: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    screenshot_sha256: Option<String>,
    source: String,
    created_at_unix_ms: u128,
    #[serde(skip_serializing_if = "Option::is_none")]
    recorded_at: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    audio_duration_ms: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    consent_version: Option<u32>,
    provenance: serde_json::Value,
    #[serde(skip_serializing_if = "Option::is_none")]
    intended_transcript: Option<String>,
}

#[derive(Debug, Serialize)]
struct CorpusExportManifest {
    schema_version: u32,
    exported_at_unix_ms: u128,
    consent_version: u32,
    items: Vec<CorpusExportItem>,
}

pub fn export_corpus(out_dir: &Path, options: CorpusExportOptions) -> Result<PathBuf> {
    export_corpus_at(&app_support_dir()?, out_dir, options)
}

fn export_corpus_to_output(
    app_support: &Path,
    out_dir: &Path,
    options: CorpusExportOptions,
) -> Result<PathBuf> {
    let output = output_outside_app_support(app_support, out_dir)?;
    let mut items = Vec::new();
    for recording in list_recordings_at(app_support)? {
        if options.only_consented && recording.screen_image.is_none() {
            continue;
        }
        if recording.audio_file.is_empty()
            || recording.audio_file.contains('/')
            || recording.audio_file.contains('\\')
            || recording.audio_file == "."
            || recording.audio_file == ".."
        {
            bail!("invalid audio file name for recording {}", recording.id);
        }
        let audio_path = corpus_dir_at(app_support)
            .join(&recording.id)
            .join(&recording.audio_file);
        let audio = fs::read(&audio_path)
            .with_context(|| format!("read corpus audio {}", audio_path.display()))?;
        let audio_sha256 = sha256_hex(&audio);
        let recording_directory = corpus_dir_at(app_support).join(&recording.id);
        let screenshot = match recording.screen_image.clone() {
            Some(_) => Some(verify_screen_image_at(&recording_directory, &recording)?),
            None => None,
        };
        if let Some(capture) = &screenshot {
            if capture.deletion_deadline_unix_ms <= now_unix_ms() {
                bail!("screenshot is expired for recording {}", recording.id);
            }
            if options.include_screen_images {
                let image_path = recording_directory.join(&capture.file);
                atomic_copy(
                    &image_path,
                    &output
                        .join("screenshots")
                        .join(format!("{}.png", recording.id)),
                )?;
            }
        } else if options.include_screen_images {
            bail!(
                "screen images were requested, but recording {} has no consented screenshot",
                recording.id
            );
        }
        items.push(CorpusExportItem {
            session_id: recording.id.clone(),
            audio_sha256,
            screenshot_sha256: screenshot.as_ref().map(|capture| capture.sha256.clone()),
            source: recording.source.clone(),
            created_at_unix_ms: recording.created_at_unix_ms,
            recorded_at: recording.recorded_at.clone(),
            audio_duration_ms: recording.audio_duration_ms,
            consent_version: screenshot.as_ref().map(|capture| capture.consent_version),
            provenance: serde_json::json!({
                "screen_capture_origin": screenshot.as_ref().map(|capture| {
                    capture.capture_origin.clone()
                }),
                "display_id": screenshot.as_ref().map(|capture| capture.display_id.clone()),
                "pixel_dimensions": screenshot.as_ref().map(|capture| {
                    capture.pixel_dimensions.clone()
                }),
                "scale_factor": screenshot.as_ref().map(|capture| capture.scale_factor),
                "captured_at_unix_ms": screenshot.as_ref().map(|capture| {
                    capture.captured_at_unix_ms
                }),
                "deletion_deadline_unix_ms": screenshot.as_ref().map(|capture| {
                    capture.deletion_deadline_unix_ms
                }),
                "consented_at_unix_ms": screenshot.as_ref().map(|capture| {
                    capture.consented_at_unix_ms
                }),
                "capability": screenshot.as_ref().map(|capture| capture.capability.clone()),
                "screen_image_exported": options.include_screen_images && screenshot.is_some(),
            }),
            intended_transcript: options
                .include_text
                .then(|| recording.intended_transcript.clone())
                .flatten(),
        });
    }
    let manifest = CorpusExportManifest {
        schema_version: DATA_SCHEMA_VERSION,
        exported_at_unix_ms: now_unix_ms(),
        consent_version: TRAINING_CAPTURE_CONSENT_VERSION,
        items,
    };
    write_json_atomic(&output.join("manifest.json"), &manifest)?;
    Ok(output)
}

pub fn export_corpus_at(
    app_support: &Path,
    out_dir: &Path,
    options: CorpusExportOptions,
) -> Result<PathBuf> {
    let output = output_outside_app_support(app_support, out_dir)?;
    if let Err(error) = export_corpus_to_output(app_support, &output, options) {
        let _ = fs::remove_dir_all(&output);
        return Err(error);
    }
    Ok(output)
}

fn list_recordings_at(app_support: &Path) -> Result<Vec<RecordingMetadata>> {
    let directory = corpus_dir_at(app_support);
    if !directory.is_dir() {
        return Ok(Vec::new());
    }
    let mut recordings = Vec::new();
    for entry in fs::read_dir(&directory)? {
        let path = entry?.path().join("metadata.json");
        if path.is_file() {
            recordings.push(load_recording_at(&path)?);
        }
    }
    recordings.sort_by_key(|recording| Reverse(recording.created_at_unix_ms));
    Ok(recordings)
}

fn output_outside_app_support(app_support: &Path, requested: &Path) -> Result<PathBuf> {
    let support = app_support
        .canonicalize()
        .with_context(|| format!("resolve {}", app_support.display()))?;
    if requested.exists() {
        bail!(
            "export destination must not already exist: {}",
            requested.display()
        );
    }
    let parent = requested
        .parent()
        .with_context(|| format!("export path has no parent: {}", requested.display()))?;
    let parent = parent
        .canonicalize()
        .with_context(|| format!("resolve {}", parent.display()))?;
    let name = requested
        .file_name()
        .and_then(|value| value.to_str())
        .context("export directory has no UTF-8 name")?;
    let output = parent.join(name);
    if output.starts_with(&support) {
        bail!("export destination is inside Phonon's personal data directory");
    }
    Ok(output)
}

fn dictionary_key(entry: &DictionaryEntry) -> (String, Option<String>) {
    (
        entry.phrase.trim().to_lowercase(),
        entry.replacement.as_ref().map(|value| value.to_lowercase()),
    )
}

fn normalized_unique(values: Vec<String>) -> Vec<String> {
    let mut by_key = BTreeMap::new();
    for value in values {
        let value = value.trim().to_string();
        if !value.is_empty() {
            by_key.entry(value.to_lowercase()).or_insert(value);
        }
    }
    by_key.into_values().collect()
}

fn normalize(value: &str) -> String {
    value
        .chars()
        .map(|character| {
            if character.is_alphanumeric() {
                character.to_ascii_lowercase()
            } else {
                ' '
            }
        })
        .collect::<String>()
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
}

fn contains_phrase(text: &str, phrase: &str) -> bool {
    format!(" {text} ").contains(&format!(" {phrase} "))
}

fn plausibly_same_word(left: &str, right: &str) -> bool {
    if left == right {
        return true;
    }
    let longest = left.chars().count().max(right.chars().count());
    if longest < 5 || left.chars().next() != right.chars().next() {
        return false;
    }
    let allowed = if longest >= 6 { 2 } else { 1 };
    levenshtein(left, right) <= allowed
}

/// Webster's Second word list shipped with the app (`assets/english_words.txt`).
/// Loaded once; an install missing the file falls back to no phonetic guessing,
/// which is the safe direction — a missed term repair beats a corrupted word.
fn english_words() -> &'static std::collections::HashSet<String> {
    static WORDS: std::sync::OnceLock<std::collections::HashSet<String>> =
        std::sync::OnceLock::new();
    WORDS.get_or_init(|| {
        let path = crate::paths::project_root().join("assets/english_words.txt");
        fs::read_to_string(path)
            .map(|text| {
                text.lines()
                    .map(str::trim)
                    .filter(|line| !line.is_empty())
                    .map(str::to_string)
                    .collect()
            })
            .unwrap_or_default()
    })
}

/// Whether every word of `phrase` is ordinary English, allowing for the common
/// inflections the base word list omits.
fn is_ordinary_english(phrase: &str) -> bool {
    let words = english_words();
    if words.is_empty() {
        // No list available: treat everything as ordinary so nothing is guessed.
        return true;
    }
    let mut any = false;
    for word in phrase.split_whitespace() {
        any = true;
        if !is_ordinary_english_word(words, word) {
            return false;
        }
    }
    any
}

/// Whether a word is real enough that a term must not be allowed to swallow it.
/// Deliberately a plain lookup: the inflection guesses below exist to decide
/// whether a word may be *rewritten*, and they are far too loose to decide
/// whether one may be *deleted* -- they read `bloss` as English by way of `blo`.
fn is_real_word(word: &str) -> bool {
    if word.chars().count() <= 2 {
        return true;
    }
    let words = english_words();
    // With no list to consult, assume every word is real so nothing is dropped.
    words.is_empty() || words.contains(&word.to_lowercase())
}

fn is_ordinary_english_word(words: &std::collections::HashSet<String>, word: &str) -> bool {
    let word = word.to_lowercase();
    if word.chars().count() <= 2 {
        // The list leaves out `a`, `i`, `at` and their kin, which would
        // otherwise read as exotic and license a guess. A token this short
        // carries too little sound to overrule what the recognizer heard.
        return true;
    }
    if words.contains(&word) {
        return true;
    }
    for suffix in ["s", "es", "ed", "d", "ing", "ly", "er", "ers", "est"] {
        if let Some(stem) = word.strip_suffix(suffix) {
            if stem.len() < 3 {
                continue;
            }
            if words.contains(stem) {
                return true;
            }
            // running -> run, stopped -> stop
            if let Some(shortened) = stem.strip_suffix(stem.chars().last().unwrap_or(' ')) {
                if shortened.len() >= 3 && words.contains(shortened) {
                    return true;
                }
            }
            // making -> make, filed -> file
            if words.contains(&format!("{stem}e")) {
                return true;
            }
            // carries -> carry
            if let Some(without_i) = stem.strip_suffix('i') {
                if without_i.len() >= 3 && words.contains(&format!("{without_i}y")) {
                    return true;
                }
            }
        }
    }
    false
}

/// Whether two canonical spellings are the same term written differently, which
/// spacing and capitalization alone do not distinguish.
fn same_term(left: &str, right: &str) -> bool {
    let squash = |value: &str| {
        value
            .chars()
            .filter(|character| character.is_alphanumeric())
            .flat_map(char::to_lowercase)
            .collect::<String>()
    };
    squash(left) == squash(right)
}

fn capital_count(value: &str) -> usize {
    value
        .chars()
        .filter(|character| character.is_uppercase())
        .count()
}

/// Longest run of transcript words compared against one dictionary form. Three
/// covers the terms a recognizer actually splits apart (`Claude Agent SDK`)
/// without letting a match wander across a clause.
const MAX_PHONETIC_WINDOW: usize = 3;

fn phonetic_key(value: &str) -> String {
    let mut output = String::new();
    // A doubled letter inside a word is silent, so `bloss` reduces to `bls`. A
    // repeat across a word boundary is not: collapsing those made `gpt 5` and
    // `GPT-5.5` indistinguishable, and turned the model version into the wrong
    // one. Restart the run at every boundary.
    for word in normalize(value).split_whitespace() {
        let mut previous = None;
        for character in word.chars().filter(|character| character.is_alphanumeric()) {
            let character = match character {
                'c' | 'q' => 'k',
                other => other,
            };
            if matches!(character, 'a' | 'e' | 'i' | 'o' | 'u' | 'y') {
                continue;
            }
            if previous != Some(character) {
                output.push(character);
            }
            previous = Some(character);
        }
    }
    output
}

/// OCR provides an independent, exact observation of the canonical term, so
/// this match can tolerate a wider ASR miss than global dictionary retrieval.
/// It remains bounded to nearby transcript word windows and never activates a
/// term that is absent from the screen.
fn contextually_similar_phrase(transcript: &str, form: &str) -> bool {
    let transcript = normalize(transcript);
    let form = normalize(form);
    if transcript.is_empty() || form.is_empty() {
        return false;
    }
    if contains_phrase(&transcript, &form) {
        return true;
    }

    let transcript_words = transcript.split_whitespace().collect::<Vec<_>>();
    let form_words = form.split_whitespace().collect::<Vec<_>>();
    let compact_form = form_words.concat();
    let shortest_window = form_words.len().saturating_sub(1).max(1);
    let longest_window = (form_words.len() + 1).min(transcript_words.len());
    for window_len in shortest_window..=longest_window {
        for window in transcript_words.windows(window_len) {
            if broadly_similar_word(&window.concat(), &compact_form) {
                return true;
            }
        }
    }
    false
}

fn broadly_similar_word(left: &str, right: &str) -> bool {
    let longest = left.chars().count().max(right.chars().count());
    if longest < 5 {
        return false;
    }
    let distance = levenshtein(left, right);
    distance <= 3 && distance * 2 <= longest
}

fn levenshtein(left: &str, right: &str) -> usize {
    let right: Vec<char> = right.chars().collect();
    let mut previous: Vec<usize> = (0..=right.len()).collect();
    for (row, left_character) in left.chars().enumerate() {
        let mut current = vec![row + 1];
        for (column, right_character) in right.iter().enumerate() {
            let cost = usize::from(left_character != *right_character);
            current.push(
                (current[column] + 1)
                    .min(previous[column + 1] + 1)
                    .min(previous[column] + cost),
            );
        }
        previous = current;
    }
    previous[right.len()]
}

fn replace_ascii_phrase(text: &str, from: &str, to: &str) -> (String, u64) {
    let needle = from.trim();
    if needle.is_empty() || needle.eq_ignore_ascii_case(to) {
        return (text.to_string(), 0);
    }
    let lower_text = text.to_ascii_lowercase();
    let lower_needle = needle.to_ascii_lowercase();
    let mut output = String::with_capacity(text.len());
    let mut cursor = 0;
    let mut count = 0;
    while let Some(relative) = lower_text[cursor..].find(&lower_needle) {
        let start = cursor + relative;
        let end = start + lower_needle.len();
        let left_boundary = start == 0 || !text.as_bytes()[start - 1].is_ascii_alphanumeric();
        let right_boundary = end == text.len() || !text.as_bytes()[end].is_ascii_alphanumeric();
        if left_boundary && right_boundary {
            output.push_str(&text[cursor..start]);
            output.push_str(to);
            cursor = end;
            count += 1;
        } else {
            output.push_str(&text[cursor..end]);
            cursor = end;
        }
    }
    output.push_str(&text[cursor..]);
    (output, count)
}

fn wav_duration_ms(path: &Path) -> Result<u64> {
    let bytes = fs::read(path).with_context(|| format!("read wav {}", path.display()))?;
    if bytes.len() < 44 || &bytes[0..4] != b"RIFF" || &bytes[8..12] != b"WAVE" {
        bail!("unsupported WAV header");
    }
    let channels = u16::from_le_bytes([bytes[22], bytes[23]]) as u64;
    let sample_rate = u32::from_le_bytes([bytes[24], bytes[25], bytes[26], bytes[27]]) as u64;
    let bits = u16::from_le_bytes([bytes[34], bytes[35]]) as u64;
    let data_bytes = u32::from_le_bytes([bytes[40], bytes[41], bytes[42], bytes[43]]) as u64;
    let bytes_per_second = sample_rate * channels * bits / 8;
    if bytes_per_second == 0 {
        bail!("invalid WAV rate");
    }
    Ok(data_bytes * 1_000 / bytes_per_second)
}

fn write_json_atomic(path: &Path, value: &impl Serialize) -> Result<()> {
    let parent = path
        .parent()
        .with_context(|| format!("path has no parent: {}", path.display()))?;
    fs::create_dir_all(parent)?;
    let temporary = path.with_extension("json.tmp");
    let bytes = serde_json::to_vec_pretty(value)?;
    fs::write(&temporary, bytes).with_context(|| format!("write {}", temporary.display()))?;
    fs::rename(&temporary, path).with_context(|| format!("replace {}", path.display()))?;
    Ok(())
}

fn now_unix_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_millis())
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::{
        extract_transcript_payload, register_screen_image_at, safe_polish_output,
        update_recording_at, CorpusExportOptions, DictionaryEntry, DictionaryFile,
        RecordingMetadata, ScreenImageCaptureRequest, SettingsFile, SCREENSHOT_FILE_NAME,
    };
    use std::fs;
    use std::path::{Path, PathBuf};

    fn entry(phrase: &str, replacement: Option<&str>) -> DictionaryEntry {
        DictionaryEntry {
            phrase: phrase.to_string(),
            replacement: replacement.map(str::to_string),
            spoken_forms: Vec::new(),
            source: "test".into(),
            starred: false,
            usage_count: 0,
        }
    }

    fn test_wav() -> Vec<u8> {
        let mut wav = vec![0_u8; 44];
        wav[0..4].copy_from_slice(b"RIFF");
        wav[4..8].copy_from_slice(&36_u32.to_le_bytes());
        wav[8..12].copy_from_slice(b"WAVE");
        wav[12..16].copy_from_slice(b"fmt ");
        wav[16..20].copy_from_slice(&16_u32.to_le_bytes());
        wav[20..22].copy_from_slice(&1_u16.to_le_bytes());
        wav[22..24].copy_from_slice(&1_u16.to_le_bytes());
        wav[24..28].copy_from_slice(&16_000_u32.to_le_bytes());
        wav[28..32].copy_from_slice(&32_000_u32.to_le_bytes());
        wav[32..34].copy_from_slice(&2_u16.to_le_bytes());
        wav[34..36].copy_from_slice(&16_u16.to_le_bytes());
        wav[36..40].copy_from_slice(b"data");
        wav[40..44].copy_from_slice(&0_u32.to_le_bytes());
        wav
    }

    fn enable_capture(app_support: &Path) {
        fs::create_dir_all(app_support).unwrap();
        fs::write(
            app_support.join("settings.json"),
            r#"{"schema_version":1,"training_capture_enabled":true,"include_screen_images":true,"screenshot_retention_seconds":60}"#,
        )
        .unwrap();
    }

    fn capture_request() -> ScreenImageCaptureRequest {
        ScreenImageCaptureRequest {
            display_id: "display-1".into(),
            pixel_width: 2_560,
            pixel_height: 1_440,
            scale_factor: 2.0,
            capture_origin: "swift-bar-stop".into(),
            captured_at_unix_ms: None,
            consented_at_unix_ms: Some(900),
            screen_capture_permission_granted: true,
        }
    }

    fn attach_fixture(
        app_support: &std::path::Path,
    ) -> (tempfile::TempDir, PathBuf, PathBuf, Vec<u8>) {
        enable_capture(app_support);
        let recording = app_support.join("Corpus").join("session-1");
        fs::create_dir_all(&recording).unwrap();
        fs::write(recording.join("audio.wav"), test_wav()).unwrap();
        let image_dir = tempfile::tempdir().unwrap();
        let image = vec![0x89_u8, b'P', b'N', b'G', 0x0d, 0x0a, 0x1a, 0x0a, 1];
        let image_path = image_dir.path().join("capture.png");
        fs::write(&image_path, &image).unwrap();
        (image_dir, recording.join("audio.wav"), image_path, image)
    }

    fn unconsented_recording(app_support: &std::path::Path, id: &str, intended: &str) {
        let recording = app_support.join("Corpus").join(id);
        fs::create_dir_all(&recording).unwrap();
        fs::write(recording.join("audio.wav"), test_wav()).unwrap();
        let mut metadata = RecordingMetadata::new(id.to_string(), "test", "audio.wav".into());
        metadata.raw_transcript = "private generated ASR text".into();
        metadata.final_transcript = "private final text".into();
        metadata.intended_transcript = Some(intended.into());
        super::save_recording_at(app_support, &metadata).unwrap();
    }

    #[test]
    fn screen_capture_registers_hashed_provenance_atomically() {
        let support = tempfile::tempdir().unwrap();
        let app_support = support.path().to_path_buf();
        let (_image_dir, audio, image_path, image) = attach_fixture(&app_support);
        let recording =
            register_screen_image_at(&app_support, &audio, &image_path, capture_request(), "bar")
                .unwrap();
        let capture = recording.screen_image.as_ref().unwrap();
        assert_eq!(capture.file, SCREENSHOT_FILE_NAME);
        assert_eq!(capture.display_id, "display-1");
        assert_eq!(capture.pixel_dimensions.width, 2_560);
        assert_eq!(capture.pixel_dimensions.height, 1_440);
        assert_eq!(capture.scale_factor.to_string(), "2");
        assert_eq!(capture.capture_origin, "swift-bar-stop");
        assert!(capture.captured_at_unix_ms > 0);
        assert!(capture.deletion_deadline_unix_ms > capture.captured_at_unix_ms);
        assert_eq!(
            capture.consent_version,
            super::TRAINING_CAPTURE_CONSENT_VERSION
        );
        assert_eq!(capture.consented_at_unix_ms, 900);
        assert_eq!(capture.capability, "local_training_corpus_image");
        let retained = fs::read(
            app_support
                .join("Corpus")
                .join("session-1")
                .join(SCREENSHOT_FILE_NAME),
        )
        .unwrap();
        assert_eq!(retained, image);
        let saved: RecordingMetadata = serde_json::from_slice(
            &fs::read(
                app_support
                    .join("Corpus")
                    .join("session-1")
                    .join("metadata.json"),
            )
            .unwrap(),
        )
        .unwrap();
        assert_eq!(saved.screen_image.unwrap().sha256, capture.sha256);
        let sidecar: serde_json::Value = serde_json::from_slice(
            &fs::read(
                app_support
                    .join("Corpus")
                    .join("session-1")
                    .join(super::SCREENSHOT_MANIFEST_FILE_NAME),
            )
            .unwrap(),
        )
        .unwrap();
        assert_eq!(sidecar["recording_id"], "session-1");
        assert_eq!(sidecar["capture"]["sha256"], capture.sha256);
    }

    #[test]
    fn screen_capture_fails_closed_without_explicit_scope() {
        let support = tempfile::tempdir().unwrap();
        let app_support = support.path().to_path_buf();
        let (_image_dir, audio, image_path, _image) = attach_fixture(&app_support);
        fs::write(app_support.join("settings.json"), r#"{"schema_version":1}"#).unwrap();
        let result =
            register_screen_image_at(&app_support, &audio, &image_path, capture_request(), "bar");
        assert!(result.is_err());
        assert!(!app_support
            .join("Corpus")
            .join("session-1")
            .join(SCREENSHOT_FILE_NAME)
            .exists());
    }

    #[test]
    fn screen_capture_rejects_replacement_without_orphan_sidecar() {
        let support = tempfile::tempdir().unwrap();
        let app_support = support.path().to_path_buf();
        let (_image_dir, audio, image_path, _image) = attach_fixture(&app_support);
        register_screen_image_at(&app_support, &audio, &image_path, capture_request(), "bar")
            .unwrap();
        let second =
            register_screen_image_at(&app_support, &audio, &image_path, capture_request(), "bar");
        assert!(second.is_err());
        let recording = app_support.join("Corpus").join("session-1");
        assert_eq!(
            fs::read(recording.join(SCREENSHOT_FILE_NAME))
                .unwrap()
                .len(),
            9
        );
        assert!(recording
            .join(super::SCREENSHOT_MANIFEST_FILE_NAME)
            .is_file());
    }

    #[test]
    fn older_settings_gain_safe_capture_defaults() {
        let settings: SettingsFile = serde_json::from_str(r#"{"schema_version":1}"#).unwrap();
        assert!(!settings.training_capture_enabled);
        assert!(!settings.include_screen_images);
        assert_eq!(settings.screenshot_retention_seconds, 24 * 60 * 60);
    }

    #[test]
    fn expired_screenshot_is_deleted_without_audio_or_metadata() {
        let support = tempfile::tempdir().unwrap();
        let app_support = support.path().to_path_buf();
        let (_image_dir, audio, image_path, _image) = attach_fixture(&app_support);
        register_screen_image_at(&app_support, &audio, &image_path, capture_request(), "bar")
            .unwrap();
        let deleted = super::delete_expired_screenshots_at(&app_support, u128::MAX).unwrap();
        assert_eq!(deleted, ["session-1"]);
        let recording = app_support.join("Corpus").join("session-1");
        assert!(recording.join("audio.wav").is_file());
        assert!(recording.join("metadata.json").is_file());
        assert!(!recording.join(SCREENSHOT_FILE_NAME).exists());
        assert!(!recording
            .join(super::SCREENSHOT_MANIFEST_FILE_NAME)
            .exists());
    }

    #[test]
    fn export_only_consented_can_omit_text_and_copy_verified_images() {
        let support = tempfile::tempdir().unwrap();
        let app_support = support.path().to_path_buf();
        let (_image_dir, audio, image_path, image) = attach_fixture(&app_support);
        register_screen_image_at(&app_support, &audio, &image_path, capture_request(), "bar")
            .unwrap();
        let mut metadata =
            super::load_recording_at(&audio.with_file_name("metadata.json")).unwrap();
        metadata.raw_transcript = "private raw".into();
        metadata.final_transcript = "private final".into();
        metadata.intended_transcript = Some("Intended label".into());
        super::save_recording_at(&app_support, &metadata).unwrap();
        unconsented_recording(&app_support, "session-2", "Unconsented intended");
        let out = tempfile::tempdir().unwrap();
        let output = super::export_corpus_at(
            &app_support,
            &out.path().join("export"),
            CorpusExportOptions {
                include_screen_images: true,
                only_consented: true,
                include_text: true,
            },
        )
        .unwrap();
        let manifest: serde_json::Value =
            serde_json::from_slice(&fs::read(output.join("manifest.json")).unwrap()).unwrap();
        let items = manifest["items"].as_array().unwrap();
        assert_eq!(items.len(), 1);
        assert_eq!(items[0]["session_id"], "session-1");
        assert_eq!(items[0]["consent_version"], 1);
        assert_eq!(items[0]["intended_transcript"], "Intended label");
        assert!(items[0].get("raw_transcript").is_none());
        assert!(items[0].get("final_transcript").is_none());
        assert_eq!(
            fs::read(output.join("screenshots").join("session-1.png")).unwrap(),
            image
        );
    }

    #[test]
    fn export_rejects_hash_mismatch_and_leaves_no_partial_directory() {
        let support = tempfile::tempdir().unwrap();
        let app_support = support.path().to_path_buf();
        let (_image_dir, audio, image_path, _image) = attach_fixture(&app_support);
        register_screen_image_at(&app_support, &audio, &image_path, capture_request(), "bar")
            .unwrap();
        let image = app_support
            .join("Corpus")
            .join("session-1")
            .join(SCREENSHOT_FILE_NAME);
        fs::write(&image, [0_u8; 9]).unwrap();
        let out = tempfile::tempdir().unwrap();
        let output = out.path().join("export");
        let result = super::export_corpus_at(
            &app_support,
            &output,
            CorpusExportOptions {
                include_screen_images: false,
                only_consented: true,
                include_text: false,
            },
        );
        assert!(result.is_err());
        assert!(!output.exists());
    }

    #[test]
    fn export_screen_images_fail_closed_for_unconsented_recordings() {
        let support = tempfile::tempdir().unwrap();
        let app_support = support.path().to_path_buf();
        let _ = attach_fixture(&app_support);
        unconsented_recording(&app_support, "session-2", "Unconsented intended");
        let result = super::export_corpus_at(
            &app_support,
            &support.path().join("export"),
            CorpusExportOptions {
                include_screen_images: true,
                only_consented: false,
                include_text: false,
            },
        );
        assert!(result.is_err());
    }

    #[test]
    fn update_errors_preserve_the_prior_metadata_and_screenshot() {
        let support = tempfile::tempdir().unwrap();
        let app_support = support.path().to_path_buf();
        let (_image_dir, audio, image_path, _image) = attach_fixture(&app_support);
        let before =
            register_screen_image_at(&app_support, &audio, &image_path, capture_request(), "bar")
                .unwrap();
        let update = update_recording_at(&app_support, "session-1", |_| {
            Err(anyhow::anyhow!("test failure"))
        });
        assert!(update.is_err());
        let after = super::load_recording_at(
            &app_support
                .join("Corpus")
                .join("session-1")
                .join("metadata.json"),
        )
        .unwrap();
        assert_eq!(after.screen_image, before.screen_image);
    }

    #[test]
    fn recording_delete_removes_audio_metadata_and_screenshot() {
        let support = tempfile::tempdir().unwrap();
        let app_support = support.path().to_path_buf();
        let (_image_dir, audio, image_path, _image) = attach_fixture(&app_support);
        register_screen_image_at(&app_support, &audio, &image_path, capture_request(), "bar")
            .unwrap();
        super::delete_recording_at(&app_support, "session-1").unwrap();
        assert!(!app_support.join("Corpus").join("session-1").exists());
    }

    #[test]
    fn exact_replacements_are_case_insensitive_and_word_bounded() {
        let dictionary = DictionaryFile {
            entries: vec![entry("black well", Some("Blackwell"))],
            ..Default::default()
        };
        let result = dictionary.apply_exact_replacements("BLACK WELL and black wellness");
        assert_eq!(result.text, "Blackwell and black wellness");
        assert_eq!(result.applied[0].count, 1);
    }

    #[test]
    fn phonetic_matching_reaches_multi_word_terms() {
        let dictionary = DictionaryFile {
            entries: vec![entry("Claude Code", None)],
            ..Default::default()
        };
        let prepared = dictionary.prepare_polish_input("ask clod kode to look at it");
        assert!(
            prepared.contains("clod kode => Claude Code"),
            "multi-word term was not reachable: {prepared}"
        );
        assert!(prepared.contains("ask Claude Code to look at it"));
    }

    #[test]
    fn phonetic_matching_does_not_swallow_a_neighbouring_word() {
        // Vowels are dropped, so `koo bloss is` keys the same as `koo bloss`.
        // Taking the longer window would delete `is` before the model sees it.
        let dictionary = DictionaryFile {
            entries: vec![entry("cuBLAS", None)],
            ..Default::default()
        };
        let prepared = dictionary.prepare_polish_input("koo bloss is slow");
        assert!(
            prepared.contains("cuBLAS is slow"),
            "a word was absorbed into the replacement: {prepared}"
        );
    }

    #[test]
    fn phonetic_matching_does_not_absorb_a_soundless_word() {
        let dictionary = DictionaryFile {
            entries: vec![entry("Speculative Decoding", None)],
            ..Default::default()
        };
        let prepared = dictionary.prepare_polish_input("try a speculativ decoding pass");
        assert!(
            prepared.contains("try a Speculative Decoding pass"),
            "the article was absorbed into the term: {prepared}"
        );
    }

    #[test]
    fn phonetic_matching_keeps_version_numbers_distinct() {
        // Collapsing the repeat across the boundary made `gpt 5` key the same as
        // `GPT-5.5`, quietly promoting one model version to another.
        let dictionary = DictionaryFile {
            entries: vec![entry("GPT-5.5", None)],
            ..Default::default()
        };
        let prepared = dictionary.prepare_polish_input("gpt 5 is out now");
        assert!(
            prepared.contains("gpt 5 is out now"),
            "one version was rewritten as another: {prepared}"
        );
    }

    #[test]
    fn phonetic_matching_does_not_swallow_a_real_word_beside_a_term() {
        // `my macbook` keys the same as `macbook`, so an unguarded match takes
        // the `my` with it and the word is gone before the model sees it.
        let dictionary = DictionaryFile {
            entries: vec![entry("MacBook", None)],
            ..Default::default()
        };
        let prepared = dictionary.prepare_polish_input("on my macbuk today");
        assert!(
            prepared.contains("on my"),
            "the neighbouring word was absorbed: {prepared}"
        );
    }

    #[test]
    fn phonetic_matching_treats_short_function_words_as_ordinary() {
        // `at` is missing from the word list, so without a length rule the
        // phrase reads as exotic and `even at` gets rewritten to `uv init`.
        let dictionary = DictionaryFile {
            entries: vec![entry("uv init", None)],
            ..Default::default()
        };
        let prepared = dictionary.prepare_polish_input("it is worse even at batch one");
        assert!(
            prepared.contains("worse even at batch one"),
            "ordinary speech was rewritten as a term: {prepared}"
        );
    }

    #[test]
    fn phonetic_matching_leaves_ordinary_english_alone() {
        let dictionary = DictionaryFile {
            entries: vec![entry("Pallas", None)],
            ..Default::default()
        };
        let prepared = dictionary.prepare_polish_input("please stop there");
        assert!(
            prepared.contains("please stop there"),
            "an ordinary word was guessed at: {prepared}"
        );
        assert!(!prepared.contains("=> Pallas"));
    }

    #[test]
    fn relevant_entries_retrieve_exact_and_fuzzy_terms() {
        let dictionary = DictionaryFile {
            entries: vec![
                entry("black well", Some("Blackwell")),
                entry("Pufferlib", None),
                entry("unrelated", None),
            ],
            ..Default::default()
        };
        let entries = dictionary.relevant_entries("Use black well with puffer lib");
        assert!(entries.iter().any(|entry| entry.phrase == "black well"));
        assert!(!entries.iter().any(|entry| entry.phrase == "unrelated"));
    }

    #[test]
    fn global_fuzzy_retrieval_rejects_short_acronyms_and_unrelated_words() {
        let dictionary = DictionaryFile {
            entries: vec![
                entry("FOV", None),
                entry("Vercel", None),
                entry("Claude", None),
            ],
            ..Default::default()
        };
        let entries = dictionary.relevant_entries("Ask cloud for this kernel");
        assert_eq!(
            entries
                .iter()
                .map(|entry| entry.canonical())
                .collect::<Vec<_>>(),
            vec!["Claude"]
        );
    }

    #[test]
    fn merge_deduplicates_case_insensitively() {
        let mut dictionary = DictionaryFile::default();
        assert_eq!(dictionary.merge([entry("vLLM", None)]), 1);
        assert_eq!(dictionary.merge([entry("VLLM", None)]), 0);
        assert_eq!(dictionary.entries.len(), 1);
    }

    #[test]
    fn older_settings_gain_native_app_defaults() {
        let settings: SettingsFile = serde_json::from_str(
            r#"{"schema_version":1,"streaming":true,"local_history":true,"screen_context":true,"microphone_priority":["USB Microphone"]}"#,
        )
        .unwrap();
        assert!(settings.instant_mic);
        assert_eq!(settings.shortcut_mode, "fn");
        assert_eq!(settings.profile_prefix, super::DEFAULT_PROFILE_PREFIX);
        assert!(settings.competitor_quit_muted.is_empty());
    }

    #[test]
    fn settings_round_trip_keeps_competitor_mutes() {
        let settings: SettingsFile = serde_json::from_str(
            r#"{"schema_version":2,"competitor_quit_muted":["com.electron.wispr-flow"]}"#,
        )
        .unwrap();
        assert_eq!(settings.competitor_quit_muted, ["com.electron.wispr-flow"]);
        let json = serde_json::to_string(&settings).unwrap();
        assert!(json.contains(r#""competitor_quit_muted":["com.electron.wispr-flow"]"#));
    }

    #[test]
    fn extracts_transcript_when_model_echoes_dictionary_envelope() {
        let output =
            "<phonon_dictionary>terms</phonon_dictionary>\n<transcript>\nUse vLLM.\n</transcript>";
        assert_eq!(extract_transcript_payload(output), "Use vLLM.");
        assert_eq!(extract_transcript_payload("Use CUDA."), "Use CUDA.");
        assert_eq!(
            extract_transcript_payload("<phonon_dictionary>\ncanonical_terms: CUDA"),
            ""
        );
    }

    #[test]
    fn safe_polish_rejects_dictionary_envelopes_and_implausible_expansion() {
        let dictionary = DictionaryFile::default();
        assert_eq!(
            safe_polish_output(
                &dictionary,
                "Can you hear me?",
                "<phonon_dictionary>\ncanonical_terms: CUDA, cuDNN"
            )
            .text,
            "Can you hear me?"
        );
        assert_eq!(
            safe_polish_output(
                &dictionary,
                "Okay.",
                "This response expanded into far too many unrelated words and should never be inserted into the focused application."
            )
            .text,
            "Okay."
        );
    }

    #[test]
    fn safe_polish_preserves_meaningful_small_word_deletions() {
        let dictionary = DictionaryFile {
            entries: vec![entry("h one hundred", Some("H100"))],
            ..Default::default()
        };
        let result = safe_polish_output(
            &dictionary,
            "Move the job to an h one hundred.",
            "Move the job to H100.",
        );
        assert_eq!(result.text, "Move the job to an H100.");
    }

    #[test]
    fn safe_polish_allows_filler_and_duplicate_removal() {
        let dictionary = DictionaryFile::default();
        assert_eq!(
            safe_polish_output(&dictionary, "Uh run the test.", "Run the test.").text,
            "Run the test."
        );
        assert_eq!(
            safe_polish_output(&dictionary, "Run the the test.", "Run the test.").text,
            "Run the test."
        );
    }

    #[test]
    fn screen_context_only_boosts_terms_relevant_to_the_transcript() {
        let dictionary = DictionaryFile {
            entries: vec![
                entry("Claude", None),
                entry("CUDA", None),
                entry("unrelated", None),
            ],
            ..Default::default()
        };
        let terms = dictionary.screen_confirmed_terms(
            "Ask cloud to repair the CUDA code.",
            "Claude Settings CUDA Toolkit unrelated",
        );
        assert!(terms.contains(&"Claude".to_string()));
        assert!(terms.contains(&"CUDA".to_string()));
        assert!(!terms.contains(&"unrelated".to_string()));
    }

    #[test]
    fn phonetic_retrieval_and_screen_context_recover_a_wider_asr_miss() {
        let dictionary = DictionaryFile {
            entries: vec![entry("cuBLAS", None), entry("unrelated", None)],
            ..Default::default()
        };
        assert_eq!(
            dictionary
                .relevant_entries("Use Kubloss here")
                .into_iter()
                .map(|entry| entry.canonical())
                .collect::<Vec<_>>(),
            vec!["cuBLAS"]
        );

        let terms = dictionary.screen_confirmed_terms(
            "Use Kubloss here",
            "CUDA Toolkit cuBLAS documentation unrelated",
        );
        assert_eq!(terms, vec!["cuBLAS"]);

        let input = dictionary.prepare_polish_input_with_context(
            "Use Kubloss here",
            "CUDA Toolkit cuBLAS documentation",
        );
        assert!(input.contains("canonical_terms: cuBLAS"));
        assert!(input.contains("screen_confirmed_terms: cuBLAS"));
    }

    #[test]
    fn polish_input_considers_the_complete_dictionary_but_sends_only_relevant_terms() {
        let dictionary = DictionaryFile {
            entries: vec![
                entry("cuBLAS", None),
                entry("cuDNN", None),
                entry("unrelated", None),
            ],
            ..Default::default()
        };
        let transcript = "Do you know the difference between Koo Bloss and Koo DNN?";
        let relevant = dictionary
            .relevant_entries(transcript)
            .into_iter()
            .map(|entry| entry.canonical())
            .collect::<Vec<_>>();
        assert!(relevant.contains(&"cuBLAS"));
        assert!(relevant.contains(&"cuDNN"));

        let input = dictionary.prepare_polish_input(transcript);
        assert!(input.contains("canonical_terms: cuBLAS, cuDNN"));
        assert!(!input.contains("unrelated"));
        assert!(input.contains("likely_terms: cuBLAS, cuDNN"));
        assert!(input.contains("phonetic_matches: koo bloss => cuBLAS; koo dnn => cuDNN"));
        assert!(input.contains(
            "<transcript>\nDo you know the difference between cuBLAS and cuDNN?\n</transcript>"
        ));
    }
}
