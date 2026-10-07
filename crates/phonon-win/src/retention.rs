//! Consent-gated Windows audio retention.

use std::fs;
use std::path::{Path, PathBuf};
use std::time::{Duration, SystemTime};

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};

pub const SETTINGS_FILE_NAME: &str = "settings.json";
pub const RECORDINGS_DIRECTORY: &str = "recordings";
const DEFAULT_RETENTION_DAYS: u64 = 7;

#[derive(Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct AudioRetention {
    #[serde(default)]
    pub retain_audio: bool,
    #[serde(default = "default_retention_days")]
    pub audio_retention_days: u64,
}

impl Default for AudioRetention {
    fn default() -> Self {
        Self {
            retain_audio: false,
            audio_retention_days: DEFAULT_RETENTION_DAYS,
        }
    }
}

fn default_retention_days() -> u64 {
    DEFAULT_RETENTION_DAYS
}

fn settings_path(root: &Path) -> PathBuf {
    root.join(SETTINGS_FILE_NAME)
}

fn recordings_path(root: &Path) -> PathBuf {
    root.join(RECORDINGS_DIRECTORY)
}

pub fn enforce(root: &Path) -> Result<Vec<PathBuf>> {
    let contents = match fs::read_to_string(settings_path(root)) {
        Ok(contents) => contents,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
            return enforce_with_policy(root, &AudioRetention::default());
        }
        Err(error) => return Err(error).with_context(|| format!("read {}", root.display())),
    };
    let policy = match serde_json::from_str(&contents) {
        Ok(policy) => policy,
        Err(error) => {
            let fail_closed = AudioRetention::default();
            let _ = enforce_with_policy(root, &fail_closed);
            return Err(error).with_context(|| format!("parse {}", settings_path(root).display()));
        }
    };
    enforce_with_policy(root, &policy)
}

fn enforce_with_policy(root: &Path, policy: &AudioRetention) -> Result<Vec<PathBuf>> {
    let directory = recordings_path(root);
    if !directory.is_dir() {
        return Ok(Vec::new());
    }
    let mut removed = Vec::new();
    for entry in
        fs::read_dir(&directory).with_context(|| format!("read {}", directory.display()))?
    {
        let path = entry?.path();
        if path.extension().and_then(|value| value.to_str()) != Some("wav") {
            continue;
        }
        let age = file_age(&path)?;
        let expired = !policy.retain_audio
            || policy.audio_retention_days == 0
            || age >= retention_window(policy.audio_retention_days);
        if expired {
            fs::remove_file(&path).with_context(|| format!("delete {}", path.display()))?;
            removed.push(path);
        }
    }
    Ok(removed)
}

fn file_age(path: &Path) -> Result<Duration> {
    let modified = fs::metadata(path)
        .and_then(|metadata| metadata.modified())
        .with_context(|| format!("read metadata for {}", path.display()))?;
    let now = SystemTime::now();
    if modified > now {
        return Ok(Duration::ZERO);
    }
    Ok(now.duration_since(modified).unwrap_or(Duration::ZERO))
}

fn retention_window(days: u64) -> Duration {
    Duration::from_secs(days.saturating_mul(24 * 60 * 60))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::time::UNIX_EPOCH;

    static SCRATCH_SEQUENCE: AtomicU64 = AtomicU64::new(0);

    struct Scratch(PathBuf);

    impl Drop for Scratch {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.0);
        }
    }

    fn scratch() -> Scratch {
        let path = std::env::temp_dir().join(format!(
            "phonon-win-retention-{}-{}-{}",
            std::process::id(),
            SCRATCH_SEQUENCE.fetch_add(1, Ordering::Relaxed),
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        fs::create_dir_all(&path).unwrap();
        Scratch(path)
    }

    #[test]
    fn missing_settings_fail_closed_to_no_retention() {
        let root = scratch();
        let recordings = recordings_path(&root.0);
        fs::create_dir_all(&recordings).unwrap();
        let current = recordings.join("current.wav");
        fs::write(&current, b"synthetic").unwrap();

        let removed = enforce(&root.0).unwrap();

        assert_eq!(removed, [current]);
        assert!(!recordings.join("current.wav").exists());
    }

    #[test]
    fn explicit_consent_keeps_current_and_removes_expired_audio() {
        let root = scratch();
        fs::write(
            settings_path(&root.0),
            r#"{"retain_audio":true,"audio_retention_days":2}"#,
        )
        .unwrap();
        let recordings = recordings_path(&root.0);
        fs::create_dir_all(&recordings).unwrap();
        let current = recordings.join("current.wav");
        let old = recordings.join("old.wav");
        fs::write(&current, b"synthetic").unwrap();
        fs::write(&old, b"synthetic").unwrap();
        let old_time = SystemTime::now() - Duration::from_secs(3 * 24 * 60 * 60);
        fs::File::options()
            .write(true)
            .open(&old)
            .unwrap()
            .set_modified(old_time)
            .unwrap();

        let removed = enforce(&root.0).unwrap();

        assert_eq!(removed.len(), 1);
        assert_eq!(removed[0], old);
        assert!(current.is_file());
        assert!(!old.exists());
    }

    #[test]
    fn malformed_settings_remove_audio_then_return_the_recovery_error() {
        let root = scratch();
        fs::write(settings_path(&root.0), r#"{"retain_audio":"yes"}"#).unwrap();
        let recordings = recordings_path(&root.0);
        fs::create_dir_all(&recordings).unwrap();
        fs::write(recordings.join("private.wav"), b"synthetic").unwrap();

        let error = enforce(&root.0).unwrap_err();

        assert!(error.to_string().contains("parse"), "{error:#}");
        assert!(!recordings.join("private.wav").exists());
    }
}
