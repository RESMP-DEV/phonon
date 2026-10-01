use phonon_asr::{
    AsrEngineSelection, ASR_MODEL_ID, ASR_MODEL_REVISION, ASR_RUNTIME_REQUIREMENT,
    PYTHON_REQUIREMENT,
};
use std::path::Path;

#[test]
fn default_selection_keeps_the_pinned_parakeet_command() {
    let root = Path::new("/repo");
    let engine = AsrEngineSelection::parakeet();

    assert!(!engine.uses_custom_script());
    assert!(!engine.uses_custom_runtime());
    assert_eq!(
        engine.uv_arguments(root),
        vec![
            "run",
            "--python",
            PYTHON_REQUIREMENT,
            "--with",
            ASR_RUNTIME_REQUIREMENT,
            "python",
            "/repo/sidecar/asr_server.py",
            "--model",
            ASR_MODEL_ID,
            "--revision",
            ASR_MODEL_REVISION,
        ]
    );
}

#[test]
fn custom_salm_selection_replaces_runtime_and_skips_parakeet_pins() {
    let root = Path::new("/repo");
    let engine = AsrEngineSelection::custom(
        "sidecar/salm_server.py",
        vec![
            "liquid-audio==1.3.0".into(),
            "peft".into(),
            "soundfile".into(),
        ],
    );

    assert!(engine.uses_custom_script());
    assert!(engine.uses_custom_runtime());
    assert_eq!(
        engine.uv_arguments(root),
        vec![
            "run",
            "--python",
            PYTHON_REQUIREMENT,
            "--with",
            "liquid-audio==1.3.0",
            "--with",
            "peft",
            "--with",
            "soundfile",
            "python",
            "/repo/sidecar/salm_server.py",
        ]
    );
}

#[test]
fn custom_script_with_empty_runtime_falls_back_to_cached_parakeet_runtime() {
    let engine = AsrEngineSelection::custom("sidecar/salm_server.py", Vec::new());

    assert!(engine.uses_custom_script());
    assert!(!engine.uses_custom_runtime());
    assert_eq!(engine.runtime_requirements(), [ASR_RUNTIME_REQUIREMENT]);
}
