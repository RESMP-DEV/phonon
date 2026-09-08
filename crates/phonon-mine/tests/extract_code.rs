//! Integration test: tiny HOME tree under tests/fixtures, exact extract output.

use std::fs;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

use phonon_mine::{extract_code, find_repos, ExtractLimits};

static N: AtomicU64 = AtomicU64::new(0);

struct Scratch(PathBuf);
impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

fn materialize_fixture_home() -> Scratch {
    let src = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/home");
    let id = N.fetch_add(1, Ordering::Relaxed);
    let dst = std::env::temp_dir().join(format!(
        "phonon-mine-int-{}-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0),
        id
    ));
    copy_tree(&src, &dst);
    plant_git(&dst);
    Scratch(dst)
}

fn copy_tree(src: &Path, dst: &Path) {
    fs::create_dir_all(dst).unwrap();
    for ent in fs::read_dir(src).unwrap() {
        let ent = ent.unwrap();
        let name = ent.file_name();
        let to = dst.join(&name);
        let ty = ent.file_type().unwrap();
        if ty.is_dir() {
            copy_tree(&ent.path(), &to);
        } else {
            fs::copy(ent.path(), to).unwrap();
        }
    }
}

fn plant_git(dir: &Path) {
    let marker = dir.join(".repo");
    if marker.is_file() {
        fs::write(dir.join(".git"), "gitdir: fake\n").unwrap();
    }
    if let Ok(rd) = fs::read_dir(dir) {
        for ent in rd.flatten() {
            if ent.file_type().map(|t| t.is_dir()).unwrap_or(false) {
                plant_git(&ent.path());
            }
        }
    }
}

#[test]
fn fixture_tree_pins_exact_extract_output() {
    let home = materialize_fixture_home();
    let mut lines = Vec::new();
    let stats = extract_code(&home.0, ExtractLimits::default(), |l| {
        lines.push(l.to_string())
    });
    let actual = lines.join("\n") + if lines.is_empty() { "" } else { "\n" };
    let expected_path =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/expected/code.txt");
    let expected = fs::read_to_string(&expected_path).unwrap_or_default();
    assert_eq!(
        actual, expected,
        "extract output drifted.\n--- actual ---\n{actual}--- expected ---\n{expected}"
    );

    let blob = actual.as_str();
    assert!(blob.contains("ident_alpha"));
    assert!(blob.contains("hello_world"));
    assert!(blob.contains("MMM aaa zzz"));
    assert!(blob.contains("Widget"));
    assert!(blob.contains("nested_mod_token"));
    assert!(blob.contains("readme_txt_token"));
    assert!(!blob.contains("MINJS_ONLY_TOKEN"));
    assert!(!blob.contains("DOTFILE_ONLY_TOKEN"));
    assert!(!blob.contains("TARGET_ONLY_TOKEN"));
    assert!(!blob.contains("NODEMOD_ONLY_TOKEN"));
    assert!(!blob.contains("BUILD_ONLY_TOKEN"));
    assert!(!blob.contains("excluded_phonon_token"));
    assert!(!blob.contains("HELD_SUPPORT_TOKEN"));
    assert!(!blob.contains("TOO_DEEP_TOKEN"));
    assert!(!blob.contains("VENV_ONLY_TOKEN"));
    assert!(!blob.contains("DOTDIR_TOKEN"));

    let repos = find_repos(&home.0, 3);
    let names: Vec<String> = repos
        .iter()
        .map(|p| {
            p.strip_prefix(&home.0)
                .unwrap()
                .to_string_lossy()
                .replace('\\', "/")
        })
        .collect();
    assert!(names.contains(&"alpha".into()), "{names:?}");
    assert!(names.contains(&"beta".into()), "{names:?}");
    assert!(names.contains(&"phonon".into()), "{names:?}");
    assert!(names.contains(&"deep/d2/d3".into()), "{names:?}");
    assert!(!names.iter().any(|n| n.contains("d4")), "{names:?}");

    assert_eq!(stats.get_int("code_repos"), Some(4));
    assert_eq!(stats.get_int("code_files"), Some(lines.len() as u64));
    assert_eq!(stats.get_int("lines"), None);
}
