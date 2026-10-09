use anyhow::{bail, Context, Result};
use serde::Deserialize;
use serde_json::json;
use std::io::{BufRead, BufReader, Write};
use std::path::Path;
use std::process::{Child, ChildStdin, Command, Stdio};
use std::sync::mpsc::{self, Receiver};
use std::thread;
use std::time::Instant;

/// Pinned ASR model and runtime. Exact revisions so an upstream release cannot
/// change what shipped users run.
pub const ASR_MODEL_ID: &str = "mlx-community/parakeet-tdt-0.6b-v2";
pub const ASR_MODEL_REVISION: &str = "8ae155301e23d820d82aa60d24817c900e69e487";
pub const ASR_RUNTIME_REQUIREMENT: &str = "parakeet-mlx==0.5.2";
pub const SALM_RUNTIME_REQUIREMENTS: [&str; 5] = [
    "liquid-audio==1.3.0",
    "peft",
    "safetensors",
    "soundfile",
    "transformers>=5.4,<6",
];

pub const SALM_VISION_RUNTIME_REQUIREMENTS: [&str; 9] = [
    "liquid-audio==1.3.0",
    "peft",
    "safetensors",
    "soundfile",
    "transformers>=5.4,<6",
    "pillow",
    "datasets",
    "einops",
    "torchvision",
];

pub const REVERSE_SALM_RUNTIME_REQUIREMENTS: [&str; 4] = [
    "liquid-audio==1.3.0",
    "peft",
    "soundfile",
    "transformers>=5.4,<6",
];

/// Pinned interpreter for both sidecars. Without this, uv falls back to whatever
/// `python3` it finds, which on a Mac with no developer tooling is the system
/// CPython 3.9 that MLX publishes no wheels for. An open-ended range would also
/// drift onto each new release before MLX supports it.
pub const PYTHON_REQUIREMENT: &str = "3.12";

/// Last lines a dying sidecar wrote, so its exit can be reported instead of
/// hanging the loader at whatever percentage it reached.
pub struct StderrTail(std::sync::Arc<std::sync::Mutex<Vec<String>>>);

impl StderrTail {
    const KEEP: usize = 12;

    /// Drains `stderr` on a background thread, keeping only the tail.
    pub fn capture(stderr: std::process::ChildStderr) -> Self {
        let lines = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
        let sink = std::sync::Arc::clone(&lines);
        thread::spawn(move || {
            for line in BufReader::new(stderr).lines().map_while(Result::ok) {
                let Ok(mut kept) = sink.lock() else { return };
                if kept.len() == Self::KEEP {
                    kept.remove(0);
                }
                kept.push(line);
            }
        });
        Self(lines)
    }

    pub fn text(&self) -> String {
        self.0
            .lock()
            .map(|lines| lines.join("\n"))
            .unwrap_or_default()
    }

    /// Message for a sidecar whose stdout closed before it was ready.
    pub fn exit_message(&self, what: &str) -> String {
        let tail = self.text();
        let tail = tail.trim();
        if tail.is_empty() {
            format!("{what} exited before becoming ready")
        } else {
            format!("{what} exited before becoming ready: {tail}")
        }
    }
}

#[derive(Debug, Deserialize)]
struct SidecarMsg {
    #[serde(rename = "type")]
    kind: String,
    #[serde(rename = "kind")]
    result_kind: Option<String>,
    pct: Option<f64>,
    msg: Option<String>,
    model: Option<String>,
    text: Option<String>,
    seconds: Option<f64>,
    id: Option<String>,
    path: Option<String>,
    image_feature_count: Option<usize>,
    #[serde(default)]
    partial: bool,
}

#[derive(Debug, Clone)]
pub enum AsrEvent {
    Status {
        pct: f64,
        msg: String,
    },
    Ready {
        model: String,
        load_ms: f64,
    },
    Result {
        id: Option<String>,
        text: String,
        seconds: f64,
        partial: bool,
    },
    ImageResult {
        id: Option<String>,
        path: String,
        text: String,
        seconds: f64,
        image_feature_count: usize,
    },
    Error {
        msg: String,
    },
}

pub struct AsrSidecar {
    child: Child,
    stdin: ChildStdin,
    rx: Receiver<AsrEvent>,
}

impl AsrSidecar {
    pub fn spawn(root: &Path) -> Result<Self> {
        let engine = AsrEngineSelection::from_environment();
        Self::spawn_engine(root, engine)
    }

    pub fn spawn_engine(root: &Path, engine: AsrEngineSelection) -> Result<Self> {
        let script = engine.script(root);
        if !script.is_file() {
            bail!("missing {}", script.display());
        }
        let uv = resolve_uv().context("uv not found; install it with Homebrew")?;
        let started = Instant::now();
        let mut command = Command::new(&uv);
        command.args(engine.uv_arguments(root));
        command
            .current_dir(root)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());
        if !engine.uses_custom_runtime() {
            apply_offline_policy(&mut command, &uv, ASR_RUNTIME_REQUIREMENT);
        }
        let mut child = command.spawn().context("spawn ASR")?;
        let stdout = child.stdout.take().context("asr stdout")?;
        let stdin = child.stdin.take().context("asr stdin")?;
        let stderr = StderrTail::capture(child.stderr.take().context("asr stderr")?);
        let (tx, rx) = mpsc::channel();
        thread::spawn(move || {
            let mut ready = false;
            for line in BufReader::new(stdout).lines().map_while(Result::ok) {
                let Ok(msg) = serde_json::from_str::<SidecarMsg>(&line) else {
                    continue;
                };
                let event = match msg.kind.as_str() {
                    "status" => AsrEvent::Status {
                        pct: msg.pct.unwrap_or(0.0),
                        msg: msg.msg.unwrap_or_default(),
                    },
                    "ready" => {
                        ready = true;
                        AsrEvent::Ready {
                            model: msg.model.unwrap_or_else(|| "parakeet".into()),
                            load_ms: started.elapsed().as_secs_f64() * 1000.0,
                        }
                    }
                    "result" => {
                        if msg
                            .result_kind
                            .as_deref()
                            .is_some_and(|kind| kind == "image")
                        {
                            AsrEvent::ImageResult {
                                id: msg.id,
                                path: msg.path.unwrap_or_default(),
                                text: msg.text.unwrap_or_default(),
                                seconds: msg.seconds.unwrap_or(0.0),
                                image_feature_count: msg.image_feature_count.unwrap_or(0),
                            }
                        } else {
                            AsrEvent::Result {
                                id: msg.id,
                                text: msg.text.unwrap_or_default(),
                                seconds: msg.seconds.unwrap_or(0.0),
                                partial: msg.partial,
                            }
                        }
                    }
                    "error" => AsrEvent::Error {
                        msg: msg.msg.unwrap_or_else(|| "asr error".into()),
                    },
                    _ => continue,
                };
                if tx.send(event).is_err() {
                    return;
                }
            }
            if !ready {
                let _ = tx.send(AsrEvent::Error {
                    msg: stderr.exit_message("ASR sidecar"),
                });
            }
        });
        Ok(Self { child, stdin, rx })
    }

    pub fn poll(&self) -> Vec<AsrEvent> {
        let mut events = Vec::new();
        while let Ok(event) = self.rx.try_recv() {
            events.push(event);
        }
        events
    }

    pub fn send(&mut self, value: serde_json::Value) -> Result<()> {
        writeln!(self.stdin, "{value}")?;
        self.stdin.flush()?;
        Ok(())
    }

    pub fn transcribe(&mut self, path: &Path, id: Option<&str>) -> Result<()> {
        self.send(json!({"cmd":"transcribe", "path":path, "id":id}))
    }

    pub fn caption(&mut self, path: &Path, id: Option<&str>) -> Result<()> {
        self.send(json!({
            "cmd":"caption",
            "path":path,
            "id":id,
            "capability":"screen_image_model",
            "consent":true
        }))
    }

    pub fn stream_start(&mut self, id: Option<&str>) -> Result<()> {
        self.send(json!({"cmd":"stream_start", "id":id}))
    }

    pub fn stream_chunk(&mut self, pcm16: &str, id: Option<&str>) -> Result<()> {
        self.send(json!({"cmd":"stream_chunk", "pcm16":pcm16, "id":id}))
    }

    pub fn stream_stop(&mut self, id: Option<&str>) -> Result<()> {
        self.send(json!({"cmd":"stream_stop", "id":id}))
    }

    pub fn warmup_stream(&mut self, path: &Path, id: Option<&str>) -> Result<()> {
        self.send(json!({"cmd":"warmup_stream", "path":path, "id":id}))
    }

    pub fn shutdown(&mut self) {
        let _ = self.send(json!({"cmd":"shutdown"}));
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

pub fn resolve_uv() -> Option<std::path::PathBuf> {
    let bundled = std::env::current_exe().ok().and_then(|executable| {
        executable
            .parent()
            .map(|directory| directory.join("uv"))
            .filter(|path| path.is_file())
    });
    bundled
        .or_else(|| which::which("uv").ok())
        .or_else(|| {
            ["/opt/homebrew/bin/uv", "/usr/local/bin/uv"]
                .into_iter()
                .map(std::path::PathBuf::from)
                .find(|path| path.is_file())
        })
        .or_else(|| {
            std::env::var_os("HOME").and_then(|home| {
                let path = std::path::PathBuf::from(home).join(".local/bin/uv");
                path.is_file().then_some(path)
            })
        })
}

/// True when uv can assemble the sidecar environment for `requirement` from
/// its cache alone. Without this, uv revalidates its PyPI index entry once the
/// entry goes stale, and with no network that fails after three retries
/// (about 33 s) even though every wheel is already on disk. Costs 50-150 ms.
pub fn uv_offline_ready(uv: &Path, requirement: &str) -> bool {
    uv_offline_ready_for(uv, &[requirement.to_owned()])
}

/// True when uv can assemble a multi-dependency sidecar environment offline.
pub fn uv_offline_ready_for(uv: &Path, requirements: &[String]) -> bool {
    let mut command = Command::new(uv);
    command.args(["run", "--offline", "--python", PYTHON_REQUIREMENT]);
    for requirement in requirements {
        command.args(["--with", requirement]);
    }
    command.args(["python", "-c", ""]);
    command
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .map(|status| status.success())
        .unwrap_or(false)
}

/// Pin a sidecar launch to uv's cache when the cache is enough, so a Mac with
/// no network or a captive portal starts dictation instead of waiting on
/// PyPI. A first launch with nothing cached still resolves online.
pub fn apply_offline_policy(command: &mut Command, uv: &Path, requirement: &str) {
    if uv_offline_ready(uv, requirement) {
        command.env("UV_OFFLINE", "1");
    }
}

impl Drop for AsrSidecar {
    fn drop(&mut self) {
        let _ = self.child.kill();
    }
}

/// A normalized ASR launch plan.
///
/// Environment overrides remain the user-facing compatibility layer, but the
/// app and benchmark consume this object so the default fused vision SALM and
/// compatibility engines cannot drift into two different protocols.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AsrEngineSelection {
    script: String,
    runtime_requirements: Vec<String>,
}

impl AsrEngineSelection {
    pub fn parakeet() -> Self {
        Self {
            script: "sidecar/asr_server.py".into(),
            runtime_requirements: vec![ASR_RUNTIME_REQUIREMENT.to_owned()],
        }
    }

    /// Best measured native single-model prototype. The adapter must be installed at
    /// `~/.local/share/phonon/salm/lora_adapter.safetensors` with rank 16.
    pub fn salm() -> Self {
        Self {
            script: "sidecar/salm_server.py".into(),
            runtime_requirements: SALM_RUNTIME_REQUIREMENTS
                .iter()
                .map(|value| (*value).to_owned())
                .collect(),
        }
    }

    /// Default fused audio-native model with the SigLIP2 vision lane installed.
    /// The fused language weights and vision rows must live under
    /// `~/.local/share/phonon/salm-vision/`.
    pub fn salm_vision() -> Self {
        Self {
            script: "sidecar/salm_vision_server.py".into(),
            runtime_requirements: SALM_VISION_RUNTIME_REQUIREMENTS
                .iter()
                .map(|value| (*value).to_owned())
                .collect(),
        }
    }

    pub fn reverse_salm() -> Self {
        Self {
            script: "sidecar/reverse_salm_server.py".into(),
            runtime_requirements: REVERSE_SALM_RUNTIME_REQUIREMENTS
                .iter()
                .map(|value| (*value).to_owned())
                .collect(),
        }
    }

    pub fn custom(script: impl Into<String>, runtime_requirements: Vec<String>) -> Self {
        Self {
            script: script.into(),
            runtime_requirements: if runtime_requirements.is_empty() {
                vec![ASR_RUNTIME_REQUIREMENT.to_owned()]
            } else {
                runtime_requirements
            },
        }
    }

    pub fn from_environment() -> Self {
        match std::env::var("PHONON_ASR_ENGINE").ok().as_deref() {
            Some("parakeet") => return Self::parakeet(),
            Some("salm") => return Self::salm(),
            Some("salm_vision") => return Self::salm_vision(),
            Some("reverse_salm") => return Self::reverse_salm(),
            _ => {}
        }
        let script = std::env::var("PHONON_ASR_SCRIPT")
            .ok()
            .filter(|value| !value.trim().is_empty())
            .unwrap_or_else(|| "sidecar/salm_vision_server.py".into());
        let runtime_requirements: Vec<String> = std::env::var("PHONON_ASR_WITH")
            .map(|value| {
                value
                    .split_whitespace()
                    .map(str::to_owned)
                    .collect::<Vec<_>>()
            })
            .unwrap_or_default();
        if script == "sidecar/asr_server.py" {
            Self::parakeet()
        } else if script == "sidecar/salm_vision_server.py" {
            Self::salm_vision()
        } else {
            Self {
                script,
                runtime_requirements,
            }
        }
    }

    pub fn script(&self, root: &Path) -> std::path::PathBuf {
        root.join(&self.script)
    }

    pub fn runtime_requirements(&self) -> &[String] {
        &self.runtime_requirements
    }

    /// Short operator-facing name for logs and `phonon doctor`.
    pub fn engine_name(&self) -> &'static str {
        match self.script.as_str() {
            "sidecar/asr_server.py" => "parakeet",
            "sidecar/salm_vision_server.py" => "salm vision",
            "sidecar/salm_server.py" => "salm",
            "sidecar/reverse_salm_server.py" => "reverse salm",
            _ => "custom ASR",
        }
    }

    pub fn uses_custom_script(&self) -> bool {
        self.script != "sidecar/asr_server.py"
    }

    pub fn uses_custom_runtime(&self) -> bool {
        self.runtime_requirements.as_slice() != [ASR_RUNTIME_REQUIREMENT]
    }

    pub fn uv_arguments(&self, root: &Path) -> Vec<String> {
        let mut args = vec!["run".into(), "--python".into(), PYTHON_REQUIREMENT.into()];
        for requirement in &self.runtime_requirements {
            args.push("--with".into());
            args.push(requirement.clone());
        }
        args.push("python".into());
        args.push(
            self.script(root)
                .to_str()
                .map(str::to_owned)
                .unwrap_or_else(|| self.script.clone()),
        );
        if !self.uses_custom_script() {
            args.push("--model".into());
            args.push(ASR_MODEL_ID.into());
            args.push("--revision".into());
            args.push(ASR_MODEL_REVISION.into());
        }
        args
    }
}
