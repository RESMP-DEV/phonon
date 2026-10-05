use anyhow::{bail, Context, Result};
use phonon_core::data::{
    app_support_dir, corpus_dir, delete_recording, dictionary_path, import_recording,
    list_recordings, load_recording_by_id, polish_config, safe_polish_output,
    set_intended_transcript, settings_path, usage_stats, DictionaryEntry, DictionaryFile,
    SettingsFile,
};
use phonon_llm::{ServeJson, ServeJsonResp};
use serde::{Deserialize, Serialize};
use std::fs::{self, File, OpenOptions};
use std::io;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

#[derive(Debug, Deserialize)]
struct WisprEntry {
    phrase: String,
    replacement: Option<String>,
    #[serde(default)]
    source: String,
    #[serde(default, rename = "isStarred")]
    is_starred: u64,
    #[serde(default, rename = "frequencyUsed")]
    frequency_used: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct CorrectionCase {
    id: String,
    raw: String,
    intended: String,
}

#[derive(Debug, Serialize)]
struct CorrectionEvaluation {
    id: String,
    raw: String,
    intended: String,
    output: String,
    passed: bool,
    latency_ms: f64,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct LegacyHistoryEntry {
    audio_path: Option<PathBuf>,
    timestamp: Option<String>,
    transcript: String,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct AquaSettings {
    #[serde(default)]
    dictionary: Vec<AquaDictionaryValue>,
    #[serde(default)]
    replacements: Vec<AquaReplacementValue>,
    #[serde(default)]
    custom_instructions: String,
    #[serde(default)]
    version: Option<u64>,
    #[serde(default)]
    schema_version: Option<u64>,
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
enum AquaDictionaryValue {
    Phrase(String),
    Invalid(serde_json::Value),
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
enum AquaReplacementValue {
    Mapping {
        #[serde(default)]
        from: String,
        #[serde(default)]
        to: String,
    },
    Invalid(serde_json::Value),
}

#[derive(Debug, Serialize)]
struct AquaImportReport {
    dry_run: bool,
    settings_version: Option<u64>,
    settings_schema_version: Option<u64>,
    inspected: usize,
    imported: usize,
    merged: usize,
    unchanged: usize,
    invalid: usize,
    instruction_characters: usize,
    dictionary: PathBuf,
    backup: Option<PathBuf>,
}

pub fn default_aqua_settings() -> Result<PathBuf> {
    let home = std::env::var_os("HOME").context("HOME is not set")?;
    Ok(PathBuf::from(home).join("Library/Application Support/Aqua Voice/settings.json"))
}

pub fn default_wispr_database() -> Result<PathBuf> {
    let home = std::env::var_os("HOME").context("HOME is not set")?;
    Ok(PathBuf::from(home).join("Library/Application Support/Wispr Flow/flow.sqlite"))
}

pub fn default_dictionary_terms_path() -> Result<PathBuf> {
    Ok(app_support_dir()?.join("dictionary_new_terms.txt"))
}

pub fn import_txt(path: &Path) -> Result<()> {
    if !path.is_file() {
        bail!("dictionary terms file not found: {}", path.display());
    }
    let contents = std::fs::read_to_string(path)
        .with_context(|| format!("read dictionary terms from {}", path.display()))?;
    let entries = contents
        .lines()
        .map(str::trim)
        .filter(|line| !line.is_empty() && !line.starts_with('#'))
        .map(|phrase| DictionaryEntry {
            phrase: phrase.to_string(),
            replacement: None,
            spoken_forms: Vec::new(),
            source: "phonon-term-list".into(),
            starred: false,
            usage_count: 0,
        });
    let mut dictionary = DictionaryFile::load()?;
    let imported = dictionary.merge(entries);
    dictionary.save()?;
    println!(
        "imported {imported} new entries; {} total\n{}",
        dictionary.entries.len(),
        dictionary_path()?.display()
    );
    Ok(())
}

pub fn import_wispr(path: &Path) -> Result<()> {
    if !path.is_file() {
        bail!("Wispr Flow database not found: {}", path.display());
    }
    let query = "SELECT phrase,replacement,source,isStarred,frequencyUsed FROM Dictionary WHERE isDeleted=0 AND isSnippet=0 ORDER BY phrase COLLATE NOCASE";
    let output = Command::new("sqlite3")
        .arg("-json")
        .arg(path)
        .arg(query)
        .output()
        .context("run sqlite3 to read Wispr Flow dictionary")?;
    if !output.status.success() {
        bail!(
            "sqlite3 failed: {}",
            String::from_utf8_lossy(&output.stderr).trim()
        );
    }
    let wispr: Vec<WisprEntry> =
        serde_json::from_slice(&output.stdout).context("parse Wispr Flow dictionary export")?;
    let mut dictionary = DictionaryFile::load()?;
    let imported = dictionary.merge(wispr.iter().map(|entry| DictionaryEntry {
        phrase: entry.phrase.clone(),
        replacement: entry.replacement.clone(),
        spoken_forms: Vec::new(),
        source: if entry.source.is_empty() {
            "wispr-flow".into()
        } else {
            format!("wispr-flow:{}", entry.source)
        },
        starred: entry.is_starred != 0,
        usage_count: entry.frequency_used,
    }));
    dictionary.save()?;
    println!(
        "imported {imported} new entries; {} total\n{}",
        dictionary.entries.len(),
        dictionary_path()?.display()
    );
    Ok(())
}

fn aqua_entries(settings: &AquaSettings) -> (Vec<DictionaryEntry>, usize) {
    let mut entries = Vec::new();
    let mut invalid = 0;
    for value in &settings.dictionary {
        match value {
            AquaDictionaryValue::Phrase(phrase) if !phrase.trim().is_empty() => {
                entries.push(DictionaryEntry {
                    phrase: phrase.trim().to_string(),
                    replacement: None,
                    spoken_forms: Vec::new(),
                    source: "aqua-voice".into(),
                    starred: false,
                    usage_count: 0,
                });
            }
            AquaDictionaryValue::Invalid(value) => {
                let _ = value;
                invalid += 1;
            }
            _ => invalid += 1,
        }
    }
    for value in &settings.replacements {
        match value {
            AquaReplacementValue::Mapping { from, to }
                if !from.trim().is_empty() && !to.trim().is_empty() =>
            {
                let from = from.trim();
                let to = to.trim();
                entries.push(DictionaryEntry {
                    phrase: from.to_string(),
                    replacement: if from.eq_ignore_ascii_case(to) {
                        None
                    } else {
                        Some(to.to_string())
                    },
                    spoken_forms: Vec::new(),
                    source: "aqua-voice".into(),
                    starred: false,
                    usage_count: 0,
                });
            }
            AquaReplacementValue::Invalid(value) => {
                let _ = value;
                invalid += 1;
            }
            _ => invalid += 1,
        }
    }
    (entries, invalid)
}

fn entry_key(entry: &DictionaryEntry) -> (String, Option<String>) {
    (
        entry.phrase.trim().to_lowercase(),
        entry
            .replacement
            .as_ref()
            .map(|replacement| replacement.trim().to_lowercase()),
    )
}

fn aqua_report(
    mut dictionary: DictionaryFile,
    settings: &AquaSettings,
    dry_run: bool,
    dictionary_path: PathBuf,
) -> AquaImportReport {
    let (incoming, invalid) = aqua_entries(settings);
    let before = dictionary
        .entries
        .iter()
        .cloned()
        .map(|entry| (entry_key(&entry), entry))
        .collect::<std::collections::BTreeMap<_, _>>();
    let imported = dictionary.merge(incoming.iter().cloned());
    let after = dictionary
        .entries
        .iter()
        .cloned()
        .map(|entry| (entry_key(&entry), entry))
        .collect::<std::collections::BTreeMap<_, _>>();
    let mut merged = 0;
    let mut unchanged = 0;
    for entry in incoming {
        let key = entry_key(&entry);
        if let (Some(previous), Some(current)) = (before.get(&key), after.get(&key)) {
            if previous == current {
                unchanged += 1;
            } else {
                merged += 1;
            }
        }
    }
    AquaImportReport {
        dry_run,
        settings_version: settings.version,
        settings_schema_version: settings.schema_version,
        inspected: settings.dictionary.len() + settings.replacements.len(),
        imported,
        merged,
        unchanged,
        invalid,
        instruction_characters: settings.custom_instructions.chars().count(),
        dictionary: dictionary_path,
        backup: None,
    }
}

fn print_aqua_report(report: &AquaImportReport, json: bool) -> Result<()> {
    if json {
        println!("{}", serde_json::to_string_pretty(report)?);
        return Ok(());
    }
    if report.dry_run {
        println!("dry run; no changes written");
    }
    println!(
        "inspected {}; imported {}; merged {}; unchanged {}; invalid {}; instructions {} characters",
        report.inspected,
        report.imported,
        report.merged,
        report.unchanged,
        report.invalid,
        report.instruction_characters
    );
    println!("dictionary: {}", report.dictionary.display());
    if let Some(backup) = &report.backup {
        println!("backup: {}", backup.display());
    }
    Ok(())
}

pub fn import_aqua(settings_path: &Path, dry_run: bool, json: bool) -> Result<()> {
    if !settings_path.is_file() {
        bail!("Aqua Voice settings not found: {}", settings_path.display());
    }
    let bytes =
        fs::read(settings_path).with_context(|| format!("read {}", settings_path.display()))?;
    let settings: AquaSettings = serde_json::from_slice(&bytes)
        .with_context(|| format!("parse {}", settings_path.display()))?;
    let target = dictionary_path()?;
    let dictionary = DictionaryFile::load()?;
    let mut report = aqua_report(dictionary, &settings, dry_run, target.clone());
    if dry_run {
        return print_aqua_report(&report, json);
    }
    report.backup = create_dated_backup(&target, SystemTime::now())?;
    let mut dictionary = DictionaryFile::load()?;
    dictionary.merge(aqua_entries(&settings).0);
    dictionary.save()?;
    print_aqua_report(&report, json)
}

fn utc_timestamp(now: SystemTime) -> String {
    let duration = now.duration_since(UNIX_EPOCH).unwrap_or_default();
    let days = duration.as_secs().div_euclid(86_400) as i64;
    let seconds = duration.as_secs().rem_euclid(86_400);
    // Howard Hinnant's civil-from-days algorithm; std has no UTC formatter.
    let z = days + 719_468;
    let era = z.div_euclid(146_097);
    let day_of_era = z.rem_euclid(146_097);
    let year_of_era =
        (day_of_era - day_of_era / 1_460 + day_of_era / 36_524 - day_of_era / 146_096) / 365;
    let year = year_of_era + era * 400;
    let day_of_year = day_of_era - (365 * year_of_era + year_of_era / 4 - year_of_era / 100);
    let shifted_month = (5 * day_of_year + 2) / 153;
    let day = day_of_year - (153 * shifted_month + 2) / 5 + 1;
    let month = if shifted_month < 10 {
        shifted_month + 3
    } else {
        shifted_month - 9
    };
    let year = if month <= 2 { year + 1 } else { year };
    format!(
        "{year:04}{month:02}{day:02}T{:02}{:02}{:02}Z",
        seconds / 3_600,
        seconds.rem_euclid(3_600) / 60,
        seconds.rem_euclid(60)
    )
}

fn create_dated_backup(path: &Path, now: SystemTime) -> Result<Option<PathBuf>> {
    if !path.is_file() {
        return Ok(None);
    }
    let file_name = path
        .file_name()
        .and_then(std::ffi::OsStr::to_str)
        .context("dictionary path has no UTF-8 file name")?;
    let timestamp = utc_timestamp(now);
    let mut suffix = 0;
    loop {
        let backup_name = if suffix == 0 {
            format!("{file_name}.{timestamp}.bak")
        } else {
            format!("{file_name}.{timestamp}-{suffix}.bak")
        };
        let backup = path.with_file_name(backup_name);
        let source = File::open(path)?;
        match OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&backup)
        {
            Ok(mut destination) => {
                let mut source = source;
                io::copy(&mut source, &mut destination)?;
                return Ok(Some(backup));
            }
            Err(error) if error.kind() == io::ErrorKind::AlreadyExists => {
                suffix += 1;
                continue;
            }
            Err(error) => return Err(error.into()),
        }
    }
}

pub fn list_dictionary(json: bool) -> Result<()> {
    let dictionary = DictionaryFile::load()?;
    if json {
        println!("{}", serde_json::to_string_pretty(&dictionary)?);
        return Ok(());
    }
    println!("{} dictionary entries", dictionary.entries.len());
    for entry in dictionary.entries {
        match entry.replacement {
            Some(replacement) => println!("{} -> {replacement}", entry.phrase),
            None => println!("{}", entry.phrase),
        }
    }
    Ok(())
}

pub fn add_dictionary(
    phrase: String,
    replacement: Option<String>,
    spoken_forms: Vec<String>,
) -> Result<()> {
    let mut dictionary = DictionaryFile::load()?;
    let added = dictionary.merge([DictionaryEntry {
        phrase,
        replacement,
        spoken_forms,
        source: "phonon-manual".into(),
        starred: true,
        usage_count: 0,
    }]);
    dictionary.save()?;
    println!(
        "{}; {} total",
        if added == 1 { "added" } else { "updated" },
        dictionary.entries.len()
    );
    Ok(())
}

pub fn learn_from_recording(id: &str, from: &str, to: &str) -> Result<()> {
    let recording = load_recording_by_id(id)?;
    if !recording
        .raw_transcript
        .to_lowercase()
        .contains(&from.to_lowercase())
    {
        bail!("{from:?} does not appear in recording {id}'s raw transcript");
    }
    let mut dictionary = DictionaryFile::load()?;
    dictionary.merge([DictionaryEntry {
        phrase: from.to_string(),
        replacement: Some(to.to_string()),
        spoken_forms: Vec::new(),
        source: format!("phonon-history:{id}"),
        starred: true,
        usage_count: 0,
    }]);
    dictionary.save()?;
    println!("learned {from} -> {to} from {id}");
    Ok(())
}

pub fn test_dictionary(text: &str, screen_context: &str, json: bool) -> Result<()> {
    let dictionary = DictionaryFile::load()?;
    let candidates = dictionary
        .relevant_entries(text)
        .into_iter()
        .cloned()
        .collect::<Vec<_>>();
    let corrected = dictionary.apply_exact_replacements(text);
    let screen_confirmed_terms = dictionary.screen_confirmed_terms(text, screen_context);
    if json {
        println!(
            "{}",
            serde_json::to_string_pretty(&serde_json::json!({
                "input": text,
                "candidates": candidates,
                "screen_confirmed_terms": screen_confirmed_terms,
                "polish_input": dictionary.prepare_polish_input_with_context(text, screen_context),
                "output": corrected.text,
                "applied": corrected.applied,
            }))?
        );
    } else {
        println!("input:  {text}");
        println!("output: {}", corrected.text);
        println!("candidates: {}", candidates.len());
        for entry in candidates {
            println!("  {} -> {}", entry.phrase, entry.canonical());
        }
        if !screen_confirmed_terms.is_empty() {
            println!("screen confirmed: {}", screen_confirmed_terms.join(", "));
        }
    }
    Ok(())
}

pub fn evaluate_dictionary(root: &Path, fixtures: &Path, json: bool) -> Result<()> {
    let cases: Vec<CorrectionCase> = serde_json::from_slice(
        &std::fs::read(fixtures).with_context(|| format!("read {}", fixtures.display()))?,
    )
    .with_context(|| format!("parse {}", fixtures.display()))?;
    let dictionary = DictionaryFile::load()?;
    let mut helper = ServeJson::spawn_with(root, &polish_config()?)?;
    let (_, warmup) = helper.warmup()?;
    if !warmup.ok {
        bail!("LLM warmup failed: {:?}", warmup.error);
    }
    let mut evaluations = Vec::new();
    for case in cases {
        let input = dictionary.prepare_polish_input(&case.raw);
        let (wall, response) = helper.run_text(&input)?;
        if !response.ok {
            bail!("correction case {} failed: {:?}", case.id, response.error);
        }
        let model_output = response.output_text.unwrap_or_default();
        let output = safe_polish_output(&dictionary, &case.raw, &model_output).text;
        evaluations.push(CorrectionEvaluation {
            passed: comparable(&output) == comparable(&case.intended),
            id: case.id,
            raw: case.raw,
            intended: case.intended,
            output,
            latency_ms: wall.as_secs_f64() * 1_000.0,
        });
    }
    helper.shutdown();
    if json {
        println!("{}", serde_json::to_string_pretty(&evaluations)?);
    } else {
        let passed = evaluations
            .iter()
            .filter(|evaluation| evaluation.passed)
            .count();
        println!(
            "dictionary correction: {passed}/{} passed",
            evaluations.len()
        );
        for evaluation in evaluations {
            let status = if evaluation.passed { "PASS" } else { "FAIL" };
            println!(
                "{status} {} ({:.0}ms)",
                evaluation.id, evaluation.latency_ms
            );
            if !evaluation.passed {
                println!("  raw:      {}", evaluation.raw);
                println!("  intended: {}", evaluation.intended);
                println!("  output:   {}", evaluation.output);
            }
        }
    }
    Ok(())
}

#[derive(Debug, Clone, Serialize)]
struct PolishEvalPass {
    wall_ms: f64,
    latency_ms: f64,
    ttft_ms: f64,
    generated_tokens: u64,
    prompt_tokens: u64,
    cached_prompt_tokens: u64,
}

#[derive(Debug, Clone, Serialize)]
struct PolishEvalCase {
    id: String,
    source: String,
    raw: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    intended: Option<String>,
    /// The model's text before `safe_polish_output`.
    model_output: String,
    /// What the user would have received.
    output: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    exact: Option<bool>,
    passes: Vec<PolishEvalPass>,
}

#[derive(Debug, Clone, Serialize)]
struct PolishEvalReport {
    config: PolishEvalConfig,
    ready_message: String,
    profile_tokens: u64,
    cases: Vec<PolishEvalCase>,
}

#[derive(Debug, Clone, Serialize)]
struct PolishEvalConfig {
    profile_dir: Option<PathBuf>,
    profile_token_budget: Option<u32>,
    prefix_cache: bool,
    passes: usize,
}

pub struct PolishEvalArgs {
    pub fixtures: Vec<PathBuf>,
    pub unlabeled: bool,
    pub limit: Option<usize>,
    pub passes: usize,
    pub json: bool,
}

fn pass_from_resp(wall: Duration, response: &ServeJsonResp) -> PolishEvalPass {
    let d = response.diagnostics.as_ref();
    PolishEvalPass {
        wall_ms: wall.as_secs_f64() * 1_000.0,
        latency_ms: d.and_then(|x| x.latency_ms).unwrap_or(0.0),
        ttft_ms: d.and_then(|x| x.ttft_ms).unwrap_or(0.0),
        generated_tokens: d.and_then(|x| x.generated_tokens).unwrap_or(0),
        prompt_tokens: d.and_then(|x| x.prompt_tokens).unwrap_or(0),
        cached_prompt_tokens: d.and_then(|x| x.cached_prompt_tokens).unwrap_or(0),
    }
}

/// Run the shipped correction path (dictionary retrieval, sidecar, output
/// guard) over corpus recordings and fixture cases, reporting text and
/// timing per case. `scripts/eval_profile_prefix.py` drives this once per
/// prompt configuration and scores the results.
pub fn polish_eval(root: &Path, args: PolishEvalArgs) -> Result<()> {
    if args.passes == 0 {
        bail!("--passes must be at least 1");
    }
    let mut cases: Vec<(String, String, String, Option<String>)> = Vec::new();
    let mut corpus = list_recordings()?
        .into_iter()
        .filter(|recording| {
            recording.speech_detected != Some(false) && !recording.raw_transcript.trim().is_empty()
        })
        .filter(|recording| {
            args.unlabeled
                || recording
                    .intended_transcript
                    .as_deref()
                    .is_some_and(|text| !text.trim().is_empty())
        })
        .collect::<Vec<_>>();
    if let Some(limit) = args.limit {
        corpus.truncate(limit);
    }
    for recording in corpus {
        let intended = recording
            .intended_transcript
            .filter(|text| !text.trim().is_empty());
        cases.push((
            recording.id,
            "corpus".into(),
            recording.raw_transcript,
            intended,
        ));
    }
    for fixtures in &args.fixtures {
        let fixture_cases: Vec<CorrectionCase> = serde_json::from_slice(
            &std::fs::read(fixtures).with_context(|| format!("read {}", fixtures.display()))?,
        )
        .with_context(|| format!("parse {}", fixtures.display()))?;
        let source = format!(
            "fixture:{}",
            fixtures
                .file_stem()
                .and_then(|stem| stem.to_str())
                .unwrap_or("fixture")
        );
        for case in fixture_cases {
            cases.push((case.id, source.clone(), case.raw, Some(case.intended)));
        }
    }
    if cases.is_empty() {
        bail!("no cases: no corpus recording has an intended transcript and no fixtures were given (try --unlabeled)");
    }

    let config = polish_config()?;
    let dictionary = DictionaryFile::load()?;
    let mut helper = ServeJson::spawn_with(root, &config)?;
    let (_, warmup) = helper.warmup()?;
    if !warmup.ok {
        bail!("LLM warmup failed: {:?}", warmup.error);
    }
    let ready_message = warmup
        .status
        .as_ref()
        .and_then(|status| status.message.clone())
        .unwrap_or_default();

    let mut profile_tokens = 0;
    let mut results = Vec::new();
    for (id, source, raw, intended) in cases {
        let input = dictionary.prepare_polish_input(&raw);
        let mut passes = Vec::new();
        let mut model_output = String::new();
        for _ in 0..args.passes {
            let (wall, response) = helper.run_text(&input)?;
            if !response.ok {
                bail!("case {id} failed: {:?}", response.error);
            }
            profile_tokens = response
                .diagnostics
                .as_ref()
                .and_then(|d| d.profile_tokens)
                .unwrap_or(profile_tokens);
            passes.push(pass_from_resp(wall, &response));
            model_output = response.output_text.unwrap_or_default();
        }
        let output = safe_polish_output(&dictionary, &raw, &model_output).text;
        let exact = intended
            .as_deref()
            .map(|intended| comparable(&output) == comparable(intended));
        results.push(PolishEvalCase {
            id,
            source,
            raw,
            intended,
            model_output,
            output,
            exact,
            passes,
        });
    }
    helper.shutdown();

    let report = PolishEvalReport {
        config: PolishEvalConfig {
            profile_dir: config.profile_dir.clone(),
            profile_token_budget: config.profile_token_budget,
            prefix_cache: config.prefix_cache,
            passes: args.passes,
        },
        ready_message,
        profile_tokens,
        cases: results,
    };
    if args.json {
        println!("{}", serde_json::to_string_pretty(&report)?);
        return Ok(());
    }
    let labeled = report
        .cases
        .iter()
        .filter(|case| case.exact.is_some())
        .count();
    let exact = report
        .cases
        .iter()
        .filter(|case| case.exact == Some(true))
        .count();
    let mut ttft = report
        .cases
        .iter()
        .filter_map(|case| case.passes.last().map(|pass| pass.ttft_ms))
        .collect::<Vec<_>>();
    ttft.sort_by(f64::total_cmp);
    println!("{}", report.ready_message);
    println!(
        "cases: {} ({labeled} labeled, {exact} exact); profile tokens: {}; median ttft (last pass): {:.0} ms",
        report.cases.len(),
        report.profile_tokens,
        ttft.get(ttft.len() / 2).copied().unwrap_or(0.0)
    );
    for case in &report.cases {
        let mark = match case.exact {
            Some(true) => "PASS",
            Some(false) => "FAIL",
            None => "    ",
        };
        let pass = case.passes.last();
        println!(
            "{mark} {} ttft={:.0}ms total={:.0}ms cached={}",
            case.id,
            pass.map_or(0.0, |p| p.ttft_ms),
            pass.map_or(0.0, |p| p.latency_ms),
            pass.map_or(0, |p| p.cached_prompt_tokens)
        );
        if case.exact == Some(false) {
            println!("  raw:      {}", case.raw);
            println!(
                "  intended: {}",
                case.intended.as_deref().unwrap_or_default()
            );
            println!("  output:   {}", case.output);
        }
    }
    Ok(())
}

pub fn show_paths() -> Result<()> {
    std::fs::create_dir_all(corpus_dir()?)?;
    SettingsFile::load_or_create()?;
    println!("data:       {}", app_support_dir()?.display());
    println!("settings:   {}", settings_path()?.display());
    println!("dictionary: {}", dictionary_path()?.display());
    println!("corpus:     {}", corpus_dir()?.display());
    Ok(())
}

pub fn list_corpus(search: Option<&str>, json: bool) -> Result<()> {
    let search = search.map(str::to_lowercase);
    let recordings = list_recordings()?
        .into_iter()
        .filter(|recording| {
            search.as_ref().is_none_or(|query| {
                recording.raw_transcript.to_lowercase().contains(query)
                    || recording.final_transcript.to_lowercase().contains(query)
                    || recording
                        .intended_transcript
                        .as_deref()
                        .unwrap_or_default()
                        .to_lowercase()
                        .contains(query)
            })
        })
        .collect::<Vec<_>>();
    if json {
        println!("{}", serde_json::to_string_pretty(&recordings)?);
    } else {
        for recording in recordings {
            let transcript = if recording.final_transcript.is_empty() {
                &recording.raw_transcript
            } else {
                &recording.final_transcript
            };
            println!(
                "{}  {}  {}",
                recording.id,
                recording.source,
                transcript.chars().take(100).collect::<String>()
            );
        }
    }
    Ok(())
}

pub fn show_recording(id: &str) -> Result<()> {
    println!(
        "{}",
        serde_json::to_string_pretty(&load_recording_by_id(id)?)?
    );
    Ok(())
}

pub fn migrate_legacy_history() -> Result<()> {
    let path = app_support_dir()?.join("History.json");
    if !path.is_file() {
        bail!("legacy history not found: {}", path.display());
    }
    let entries: Vec<LegacyHistoryEntry> = serde_json::from_slice(&std::fs::read(&path)?)
        .with_context(|| format!("parse {}", path.display()))?;
    let mut imported = 0;
    let mut missing_audio = 0;
    for entry in entries {
        let Some(audio_path) = entry.audio_path else {
            missing_audio += 1;
            continue;
        };
        if !audio_path.is_file() {
            missing_audio += 1;
            continue;
        }
        let stem = audio_path
            .file_stem()
            .and_then(|value| value.to_str())
            .context("legacy WAV has no UTF-8 filename")?;
        let id = format!("legacy_{stem}");
        let (_, created) = import_recording(
            &audio_path,
            &id,
            "legacy-macos-app",
            &entry.transcript,
            entry.timestamp,
        )?;
        imported += usize::from(created);
    }
    let recordings_dir = app_support_dir()?.join("Recordings");
    let mut orphan_wavs = 0;
    if recordings_dir.is_dir() {
        for directory_entry in std::fs::read_dir(&recordings_dir)? {
            let audio_path = directory_entry?.path();
            if audio_path.extension().and_then(|value| value.to_str()) != Some("wav") {
                continue;
            }
            let stem = audio_path
                .file_stem()
                .and_then(|value| value.to_str())
                .context("legacy WAV has no UTF-8 filename")?;
            let id = format!("legacy_{stem}");
            let (_, created) = import_recording(&audio_path, &id, "legacy-macos-app", "", None)?;
            if created {
                imported += 1;
                orphan_wavs += 1;
            }
        }
    }
    println!(
        "migrated {imported} WAV-backed entries ({orphan_wavs} without history text); {missing_audio} history rows had no available WAV"
    );
    Ok(())
}

pub fn set_intended(id: &str, text: &str) -> Result<()> {
    let recording = set_intended_transcript(id, text)?;
    println!(
        "{} intended transcript saved to {}",
        recording.id,
        corpus_dir()?
            .join(&recording.id)
            .join("metadata.json")
            .display()
    );
    Ok(())
}

pub fn delete_corpus_recording(id: &str) -> Result<()> {
    delete_recording(id)?;
    println!("deleted recording {id}");
    Ok(())
}

pub fn print_stats(json: bool) -> Result<()> {
    let stats = usage_stats()?;
    if json {
        println!("{}", serde_json::to_string_pretty(&stats)?);
    } else {
        println!("recordings:       {}", stats.recordings);
        println!("words:            {}", stats.words);
        println!(
            "speaking seconds: {:.1}",
            stats.speaking_ms as f64 / 1_000.0
        );
        println!("dictionary fixes: {}", stats.dictionary_fixes);
    }
    Ok(())
}

fn comparable(value: &str) -> String {
    value
        .chars()
        .filter(|character| character.is_alphanumeric() || character.is_whitespace())
        .flat_map(char::to_lowercase)
        .collect::<String>()
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
}

#[cfg(test)]
mod tests {
    use super::{aqua_report, comparable, create_dated_backup, utc_timestamp, AquaSettings};
    use std::path::PathBuf;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::time::{Duration, UNIX_EPOCH};

    struct TempDir(PathBuf);

    impl TempDir {
        fn new(label: &str) -> Self {
            static NEXT: AtomicUsize = AtomicUsize::new(0);
            let path = std::env::temp_dir().join(format!(
                "phonon-aqua-{}-{}-{}",
                label,
                std::process::id(),
                NEXT.fetch_add(1, Ordering::Relaxed)
            ));
            std::fs::create_dir_all(&path).unwrap();
            Self(path)
        }
    }

    impl Drop for TempDir {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    fn aqua_settings() -> AquaSettings {
        serde_json::from_str(
            r#"{
                "unrelated": "ignored",
                "version": 89,
                "dictionary": ["Kernel", "  Term "],
                "replacements": [{"from": " black well ", "to": "Blackwell"}]
            }"#,
        )
        .unwrap()
    }

    #[test]
    fn comparison_ignores_punctuation_but_not_words() {
        assert_eq!(comparable("Use CUDA, now."), "use cuda now");
        assert_ne!(comparable("Use CUDA"), comparable("Use CUDA now"));
    }

    #[test]
    fn aqua_mappings_normalize_and_count_invalid_values() {
        let settings: AquaSettings = serde_json::from_str(
            r#"{
                "version": 89,
                "dictionary": ["Alpha", "   ", true],
                "replacements": [{"from": " Beta ", "to": "Gamma"}, {"from": "", "to": "x"}, {}],
                "customInstructions": "abc"
            }"#,
        )
        .unwrap();
        let report = aqua_report(
            Default::default(),
            &settings,
            true,
            PathBuf::from("dictionary.json"),
        );
        assert_eq!(report.inspected, 6);
        assert_eq!(report.imported, 2);
        assert_eq!(report.merged, 0);
        assert_eq!(report.unchanged, 0);
        assert_eq!(report.invalid, 4);
        assert_eq!(report.instruction_characters, 3);
        assert_eq!(report.settings_version, Some(89));
    }

    #[test]
    fn aqua_mappings_are_case_insensitive_and_idempotent() {
        let settings = aqua_settings();
        let first = aqua_report(
            Default::default(),
            &settings,
            true,
            PathBuf::from("dictionary.json"),
        );
        assert_eq!(first.inspected, 3);
        assert_eq!(first.imported, 3);
        assert_eq!(first.merged, 0);
        assert_eq!(first.unchanged, 0);

        let dictionary = phonon_core::data::DictionaryFile {
            entries: vec![
                phonon_core::data::DictionaryEntry {
                    phrase: "KERNEL".into(),
                    replacement: None,
                    spoken_forms: Vec::new(),
                    source: "manual".into(),
                    starred: true,
                    usage_count: 2,
                },
                phonon_core::data::DictionaryEntry {
                    phrase: "Term".into(),
                    replacement: None,
                    spoken_forms: Vec::new(),
                    source: "manual".into(),
                    starred: false,
                    usage_count: 0,
                },
                phonon_core::data::DictionaryEntry {
                    phrase: "BLACK WELL".into(),
                    replacement: Some("Blackwell".into()),
                    spoken_forms: Vec::new(),
                    source: "manual".into(),
                    starred: true,
                    usage_count: 3,
                },
            ],
            ..Default::default()
        };
        let second = aqua_report(
            dictionary,
            &settings,
            true,
            PathBuf::from("dictionary.json"),
        );
        assert_eq!(second.imported, 0);
        assert_eq!(second.merged, 0);
        assert_eq!(second.unchanged, 3);
    }

    #[test]
    fn aqua_backup_copies_dictionary_to_a_dated_sibling() {
        let directory = TempDir::new("backup");
        let dictionary = directory.0.join("dictionary.json");
        std::fs::write(&dictionary, br#"{"entries":[]}"#).unwrap();
        let now = UNIX_EPOCH + Duration::from_secs(1_767_225_600);
        assert_eq!(utc_timestamp(now), "20260101T000000Z");

        let backup = create_dated_backup(&dictionary, now).unwrap().unwrap();
        assert_eq!(
            backup,
            directory.0.join("dictionary.json.20260101T000000Z.bak")
        );
        assert!(backup.is_file());
        assert_eq!(std::fs::read(&backup).unwrap(), b"{\"entries\":[]}");
        assert_eq!(std::fs::read(&dictionary).unwrap(), b"{\"entries\":[]}");
    }
}
