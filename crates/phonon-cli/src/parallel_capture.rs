use anyhow::{bail, Context, Result};
use phonon_core::data::{app_support_dir, SettingsFile};
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

const CONSENT_VERSION: &str = "parallel-local-audio-v1";

#[derive(Debug, Clone, Serialize, Deserialize)]
struct CaptureMetadata {
    schema_version: u32,
    status: String,
    session_id: String,
    started_at_unix_ms: u128,
    finished_at_unix_ms: Option<u128>,
    pid: Option<u32>,
    exit_code: Option<i32>,
    device_index: u32,
    output: PathBuf,
    stdout: PathBuf,
    stderr: PathBuf,
    ffmpeg: String,
    command: Vec<String>,
    sample_rate: u32,
    channels: u32,
    consent_version: String,
}

fn now_ms() -> Result<u128> {
    Ok(SystemTime::now().duration_since(UNIX_EPOCH)?.as_millis())
}

fn sessions_root() -> Result<PathBuf> {
    Ok(app_support_dir()?.join("ParallelCaptures"))
}

fn session_dir(session_id: Option<&str>) -> Result<PathBuf> {
    let Some(session_id) = session_id else {
        return sessions_root();
    };
    if session_id.is_empty()
        || !session_id.chars().all(|character| {
            character.is_ascii_alphanumeric() || character == '-' || character == '_'
        })
    {
        bail!("invalid session id: {session_id:?}");
    }
    Ok(sessions_root()?.join(session_id))
}

fn latest_session_dir() -> Result<PathBuf> {
    let root = sessions_root()?;
    let latest = fs::read_dir(&root)
        .with_context(|| format!("read {}", root.display()))?
        .filter_map(Result::ok)
        .map(|entry| entry.path())
        .filter(|path| path.is_dir() && path.join("metadata.json").is_file())
        .max();
    let Some(latest) = latest else {
        bail!("no parallel capture sessions");
    };
    Ok(latest)
}

fn resolve_session(session_id: Option<&str>) -> Result<PathBuf> {
    match session_id {
        Some(session_id) => session_dir(Some(session_id)),
        None => latest_session_dir(),
    }
}

fn atomic_json(path: &Path, value: &impl Serialize) -> Result<()> {
    let temporary = path.with_extension("json.tmp");
    {
        let mut sink = fs::File::create(&temporary)
            .with_context(|| format!("create {}", temporary.display()))?;
        sink.write_all(serde_json::to_vec_pretty(value)?.as_slice())?;
        sink.write_all(b"\n")?;
    }
    fs::rename(&temporary, path).with_context(|| format!("publish {}", path.display()))
}

fn load_metadata(directory: &Path) -> Result<CaptureMetadata> {
    let path = directory.join("metadata.json");
    Ok(serde_json::from_slice(
        &fs::read(&path).with_context(|| format!("read {}", path.display()))?,
    )?)
}

pub fn start(
    device_index: u32,
    duration_seconds: Option<u64>,
    ffmpeg: &str,
    session_id: Option<&str>,
) -> Result<()> {
    let settings_path = app_support_dir()?.join("settings.json");
    let settings = SettingsFile::load_at(settings_path.clone())?;
    if !settings.training_capture_enabled {
        bail!(
            "training capture is disabled; enable it in {} before parallel recording",
            settings_path.display()
        );
    }

    let started = now_ms()?;
    let session_id = session_id
        .map(str::to_string)
        .unwrap_or_else(|| format!("session-{started}"));
    let directory = session_dir(Some(&session_id))?;
    if directory.exists() {
        bail!("session already exists: {}", directory.display());
    }
    fs::create_dir_all(&directory).with_context(|| format!("create {}", directory.display()))?;
    let output = directory.join("microphone-48k.wav");
    let stdout_path = directory.join("ffmpeg.stdout.log");
    let stderr_path = directory.join("ffmpeg.stderr.log");

    let mut command = vec![
        ffmpeg.to_string(),
        "-hide_banner".into(),
        "-nostdin".into(),
        "-f".into(),
        "avfoundation".into(),
        "-framerate".into(),
        "1".into(),
        "-i".into(),
        format!(":{device_index}"),
        "-vn".into(),
        "-ac".into(),
        "1".into(),
        "-ar".into(),
        "48000".into(),
        "-sample_fmt".into(),
        "s16".into(),
    ];
    if let Some(seconds) = duration_seconds {
        command.extend(["-t".into(), seconds.to_string()]);
    }
    command.push(output.to_string_lossy().into_owned());

    let metadata = CaptureMetadata {
        schema_version: 1,
        status: "recording".into(),
        session_id,
        started_at_unix_ms: started,
        finished_at_unix_ms: None,
        pid: None,
        exit_code: None,
        device_index,
        output: output.clone(),
        stdout: stdout_path.clone(),
        stderr: stderr_path.clone(),
        ffmpeg: ffmpeg.to_string(),
        command: command.clone(),
        sample_rate: 48_000,
        channels: 1,
        consent_version: CONSENT_VERSION.into(),
    };
    atomic_json(&directory.join("metadata.json"), &metadata)?;

    let stdout = Stdio::from(fs::File::create(&stdout_path)?);
    let stderr = Stdio::from(fs::File::create(&stderr_path)?);
    let mut child = Command::new(ffmpeg)
        .args(&command[1..])
        .stdout(stdout)
        .stderr(stderr)
        .spawn()
        .with_context(|| format!("spawn {ffmpeg}"))?;
    let mut running = metadata.clone();
    running.pid = Some(child.id());
    atomic_json(&directory.join("metadata.json"), &running)?;

    let status = child
        .wait()
        .with_context(|| format!("wait for ffmpeg pid {}", child.id()))?;
    let mut finished = running.clone();
    finished.finished_at_unix_ms = Some(now_ms()?);
    finished.exit_code = status.code();
    // A terminated signal is the documented stop path (Ctrl+C or capture-stop),
    // not a failure: ffmpeg finalizes the WAV when it receives SIGINT/SIGTERM.
    finished.status = if status.success() {
        "complete".into()
    } else if status.code().is_none() {
        "stopped".into()
    } else {
        "failed".into()
    };
    atomic_json(&directory.join("metadata.json"), &finished)?;
    if finished.status == "failed" {
        bail!("ffmpeg exited with status {}", status.code().unwrap_or(-1));
    }
    Ok(())
}

pub fn stop(session_id: Option<&str>) -> Result<()> {
    let directory = resolve_session(session_id)?;
    let mut metadata = load_metadata(&directory)?;
    if metadata.status != "recording" {
        bail!("session is not recording: {}", directory.display());
    }
    let Some(pid) = metadata.pid else {
        bail!("recording session has no pid");
    };
    let signal_result = unsafe { libc_kill(pid as i32, 2) };
    if signal_result != 0 {
        bail!("failed to interrupt ffmpeg pid {pid}");
    }
    // ffmpeg normally finalizes quickly; the parent, if still running, publishes
    // the final metadata. This refresh handles a parent interrupted after stop.
    std::thread::sleep(Duration::from_millis(250));
    if let Ok(reloaded) = load_metadata(&directory) {
        metadata = reloaded;
    }
    println!(
        "{}",
        serde_json::to_string_pretty(&json!({
            "session": directory.display().to_string(),
            "status": metadata.status,
            "output": metadata.output,
        }))?
    );
    Ok(())
}

pub fn status(session_id: Option<&str>, json: bool) -> Result<()> {
    let directory = resolve_session(session_id)?;
    let mut metadata = load_metadata(&directory)?;
    if metadata.status == "recording" {
        if let Some(pid) = metadata.pid {
            if unsafe { libc_kill(pid as i32, 0) } != 0 {
                metadata.status = "stale".into();
            }
        }
    }
    if json {
        println!("{}", serde_json::to_string_pretty(&metadata)?);
    } else {
        println!(
            "{}\t{}\t{}",
            metadata.session_id,
            metadata.status,
            metadata.output.display()
        );
    }
    Ok(())
}

// Avoid a libc dependency for two fixed signals. SIGINT is 2 and probe is 0 on
// all Unix targets supported by Phonon.
extern "C" {
    #[link_name = "kill"]
    fn libc_kill(pid: i32, signal: i32) -> i32;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn session_ids_reject_path_traversal() {
        assert!(session_dir(Some("session-1")).is_ok());
        assert!(session_dir(Some("../escape")).is_err());
        assert!(session_dir(Some("with space")).is_err());
        assert!(session_dir(Some("")).is_err());
    }

    #[test]
    fn consent_version_is_stable() {
        assert_eq!(CONSENT_VERSION, "parallel-local-audio-v1");
    }
}
