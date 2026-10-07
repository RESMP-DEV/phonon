use phonon_asr::{
    AsrEngineSelection, ASR_MODEL_ID, ASR_MODEL_REVISION, ASR_RUNTIME_REQUIREMENT,
    PYTHON_REQUIREMENT, REVERSE_SALM_RUNTIME_REQUIREMENTS, SALM_RUNTIME_REQUIREMENTS,
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
fn native_salm_is_an_explicit_non_default_engine() {
    let root = Path::new("/repo");
    let engine = AsrEngineSelection::salm();

    assert!(engine.uses_custom_script());
    assert!(engine.uses_custom_runtime());
    assert_eq!(
        engine.script(root),
        Path::new("/repo/sidecar/salm_server.py")
    );
    let requirements: Vec<_> = SALM_RUNTIME_REQUIREMENTS
        .iter()
        .map(|value| (*value).to_owned())
        .collect();
    assert_eq!(engine.runtime_requirements(), requirements.as_slice());
}

#[test]
fn native_salm_is_selectable_by_environment() {
    unsafe { std::env::set_var("PHONON_ASR_ENGINE", "salm") };
    let engine = AsrEngineSelection::from_environment();
    assert_eq!(engine, AsrEngineSelection::salm());
    unsafe { std::env::remove_var("PHONON_ASR_ENGINE") };
}

#[test]
fn reverse_salm_is_an_explicit_non_default_engine() {
    let root = Path::new("/repo");
    let engine = AsrEngineSelection::reverse_salm();

    assert!(engine.uses_custom_script());
    assert!(engine.uses_custom_runtime());
    let requirements: Vec<_> = REVERSE_SALM_RUNTIME_REQUIREMENTS
        .iter()
        .map(|value| (*value).to_owned())
        .collect();
    assert_eq!(engine.runtime_requirements(), requirements.as_slice());
    assert_eq!(
        engine.uv_arguments(root),
        vec![
            "run".to_owned(),
            "--python".to_owned(),
            PYTHON_REQUIREMENT.to_owned(),
            "--with".to_owned(),
            "liquid-audio==1.3.0".to_owned(),
            "--with".to_owned(),
            "peft".to_owned(),
            "--with".to_owned(),
            "soundfile".to_owned(),
            "--with".to_owned(),
            "transformers>=5.4,<6".to_owned(),
            "python".to_owned(),
            "/repo/sidecar/reverse_salm_server.py".to_owned(),
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
