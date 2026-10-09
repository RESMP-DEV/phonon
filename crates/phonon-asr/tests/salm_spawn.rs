use phonon_asr::{AsrEngineSelection, AsrEvent, AsrSidecar};
use std::env;
use std::path::{Path, PathBuf};
use std::thread::sleep;
use std::time::{Duration, Instant};

#[test]
fn native_salm_real_spawn_transcribes_opt_in_wav() {
    let Ok(root) = env::var("PHONON_SALM_TEST_ROOT") else {
        return;
    };
    let wav =
        PathBuf::from(env::var("PHONON_SALM_TEST_WAV").expect("test WAV required with test root"));
    assert!(Path::new(&root).join("sidecar/salm_server.py").is_file());
    assert!(wav.is_file());

    let mut sidecar = AsrSidecar::spawn_engine(Path::new(&root), AsrEngineSelection::salm())
        .expect("spawn native SALM");
    let deadline = Instant::now() + Duration::from_secs(300);
    let mut ready = false;
    while Instant::now() < deadline && !ready {
        for event in sidecar.poll() {
            match event {
                AsrEvent::Ready { model, .. } => {
                    assert!(model.contains("LFM2.5-Audio"));
                    ready = true;
                }
                AsrEvent::Error { msg } => panic!("native SALM startup failed: {msg}"),
                AsrEvent::Status { .. } => {}
                AsrEvent::Result { .. } => {}
                AsrEvent::ImageResult { .. } => {}
            }
        }
        sleep(Duration::from_millis(100));
    }
    assert!(ready, "native SALM ready timeout");

    sidecar.transcribe(&wav, Some("spawn-probe")).unwrap();
    let deadline = Instant::now() + Duration::from_secs(300);
    loop {
        let now = Instant::now();
        assert!(now < deadline, "native SALM transcription timeout");
        for event in sidecar.poll() {
            if let AsrEvent::Result {
                id,
                text,
                seconds,
                partial,
            } = event
            {
                assert_eq!(id.as_deref(), Some("spawn-probe"));
                assert!(!text.is_empty());
                assert!(seconds.is_finite() && seconds > 0.0);
                assert!(!partial);
                sidecar.shutdown();
                return;
            }
        }
        sleep(Duration::from_millis(100));
    }
}
