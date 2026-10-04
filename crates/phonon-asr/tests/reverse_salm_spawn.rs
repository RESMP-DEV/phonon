use phonon_asr::{AsrEngineSelection, AsrEvent, AsrSidecar};
use std::env;
use std::path::{Path, PathBuf};
use std::thread::sleep;
use std::time::{Duration, Instant};

#[test]
fn reverse_salm_real_spawn_transcribes_opt_in_wav() {
    let Ok(root) = env::var("PHONON_REVERSE_SALM_TEST_ROOT") else {
        return;
    };
    let wav = PathBuf::from(
        env::var("PHONON_REVERSE_SALM_TEST_WAV").expect("test WAV required with test root"),
    );
    assert!(Path::new(&root)
        .join("sidecar/reverse_salm_server.py")
        .is_file());
    assert!(wav.is_file());

    let mut sidecar =
        AsrSidecar::spawn_engine(Path::new(&root), AsrEngineSelection::reverse_salm())
            .expect("spawn reverse SALM");
    let deadline = Instant::now() + Duration::from_secs(240);
    let mut ready = false;
    while Instant::now() < deadline && !ready {
        for event in sidecar.poll() {
            match event {
                AsrEvent::Ready { model, .. } => {
                    assert_eq!(model, "reverse-audio-vl");
                    ready = true;
                }
                AsrEvent::Error { msg } => panic!("reverse SALM startup failed: {msg}"),
                AsrEvent::Status { .. } => {}
                AsrEvent::Result { .. } => {}
            }
        }
        sleep(Duration::from_millis(100));
    }
    assert!(ready, "reverse SALM ready timeout");

    sidecar.transcribe(&wav, Some("spawn-probe")).unwrap();
    let deadline = Instant::now() + Duration::from_secs(240);
    loop {
        let now = Instant::now();
        assert!(now < deadline, "reverse SALM transcription timeout");
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
