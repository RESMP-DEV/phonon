use std::collections::BTreeSet;
use std::fs::{self, File};
use std::io::{self, Write};
use std::path::{Component, Path, PathBuf};
use std::time::Instant;

use serde_json::{Map, Value};

use crate::{
    is_tree_skip, join_tokens, path_str_prefix, phonon_support_dir, repo_docs_excluded,
    select_code_file, tokenize, CODE_MAX_BYTES, CODE_MAX_FILES, CODE_MAX_FILES_PER_REPO,
    FIND_REPOS_MAX_DEPTH,
};

#[derive(Clone, Copy, Debug)]
pub struct ExtractLimits {
    pub max_bytes: u64,
    pub max_files_per_repo: usize,
    pub max_files: usize,
    pub max_repo_depth: usize,
}

impl Default for ExtractLimits {
    fn default() -> Self {
        Self {
            max_bytes: CODE_MAX_BYTES,
            max_files_per_repo: CODE_MAX_FILES_PER_REPO,
            max_files: CODE_MAX_FILES,
            max_repo_depth: FIND_REPOS_MAX_DEPTH,
        }
    }
}

#[derive(Clone, Debug)]
pub enum StatVal {
    Int(u64),
    Float(f64),
}

#[derive(Clone, Debug, Default)]
pub struct ExtractStats {
    entries: Vec<(String, StatVal)>,
}

impl ExtractStats {
    fn add_int(&mut self, key: &str, n: u64) {
        if let Some((_, StatVal::Int(v))) = self.entries.iter_mut().find(|(k, _)| k == key) {
            *v = v.saturating_add(n);
            return;
        }
        self.entries.push((key.to_string(), StatVal::Int(n)));
    }

    fn set_int(&mut self, key: &str, n: u64) {
        if let Some((_, slot)) = self.entries.iter_mut().find(|(k, _)| k == key) {
            *slot = StatVal::Int(n);
            return;
        }
        self.entries.push((key.to_string(), StatVal::Int(n)));
    }

    fn set_float(&mut self, key: &str, n: f64) {
        if let Some((_, slot)) = self.entries.iter_mut().find(|(k, _)| k == key) {
            *slot = StatVal::Float(n);
            return;
        }
        self.entries.push((key.to_string(), StatVal::Float(n)));
    }

    pub fn get_int(&self, key: &str) -> Option<u64> {
        self.entries
            .iter()
            .find_map(|(k, v)| match (k.as_str(), v) {
                (kk, StatVal::Int(n)) if kk == key => Some(*n),
                _ => None,
            })
    }

    pub fn entries(&self) -> &[(String, StatVal)] {
        &self.entries
    }

    pub fn to_json_map(&self) -> Map<String, Value> {
        let mut m = Map::new();
        for (k, v) in &self.entries {
            m.insert(k.clone(), stat_to_value(v));
        }
        m
    }
}

fn stat_to_value(v: &StatVal) -> Value {
    match v {
        StatVal::Int(n) => Value::Number((*n).into()),
        StatVal::Float(f) => NumberFloat(*f).into_value(),
    }
}

struct NumberFloat(f64);

impl NumberFloat {
    fn into_value(self) -> Value {
        Value::Number(serde_json::Number::from_f64(self.0).unwrap_or_else(|| 0.into()))
    }
}

/// Git repos under `home` to `max_depth`, excluding `$HOME/Library` and dot dirs.
/// Same rules as `profile_miner.extract.find_repos`.
pub fn find_repos(home: &Path, max_depth: usize) -> Vec<PathBuf> {
    let mut repos = Vec::new();
    let mut walk = OsWalk::new(home.to_path_buf());
    while let Some((dirpath, mut dirnames, _files)) = walk.next_dir() {
        let depth = dir_depth(home, &dirpath);
        if depth == 0 {
            dirnames.retain(|d| d != "Library" && !d.starts_with('.'));
            walk.push_dirs(&dirpath, &dirnames);
            continue;
        }
        dirnames.retain(|d| !d.starts_with('.') && !is_tree_skip(d));
        if is_git_repo(&dirpath) {
            repos.push(dirpath);
            continue;
        }
        if depth >= max_depth {
            continue;
        }
        walk.push_dirs(&dirpath, &dirnames);
    }
    repos
}

/// Walk repos and emit one line of unique sorted identifiers per selected file.
pub fn extract_code(
    home: &Path,
    limits: ExtractLimits,
    mut emit: impl FnMut(&str),
) -> ExtractStats {
    let support = phonon_support_dir(home);
    let repos = find_repos(home, limits.max_repo_depth);
    let mut stats = ExtractStats::default();
    let mut total = 0usize;

    for repo in &repos {
        if path_str_prefix(repo, &support) || repo_docs_excluded(name_of(repo)) {
            continue;
        }
        let mut n_repo = 0usize;
        let mut walk = OsWalk::new(repo.clone());
        while let Some((dirpath, mut dirnames, filenames)) = walk.next_dir() {
            dirnames.retain(|d| !d.starts_with('.') && !is_tree_skip(d));
            for fname in filenames {
                if n_repo >= limits.max_files_per_repo || total >= limits.max_files {
                    dirnames.clear();
                    break;
                }
                if !select_code_file(&fname) {
                    continue;
                }
                let path = dirpath.join(&fname);
                match fs::metadata(&path) {
                    Ok(meta) if meta.len() > limits.max_bytes => {
                        stats.add_int("code_files_too_big", 1);
                        continue;
                    }
                    Ok(_) => {}
                    Err(_) => continue,
                }
                let bytes = match fs::read(&path) {
                    Ok(b) => b,
                    Err(_) => continue,
                };
                let toks = tokenize(&bytes);
                if toks.is_empty() {
                    continue;
                }
                n_repo += 1;
                total += 1;
                stats.add_int("code_tokens", toks.len() as u64);
                let line = join_tokens(&toks);
                emit(&line);
            }
            walk.push_dirs(&dirpath, &dirnames);
        }
        stats.add_int("code_files", n_repo as u64);
    }
    stats.set_int("code_repos", repos.len() as u64);
    stats
}

/// Write `extract/code.txt` and merge `extract/counts.json` like the Python stage.
pub fn write_extract_code(
    home: &Path,
    out: &Path,
    until: Option<&str>,
) -> io::Result<ExtractStats> {
    let t0 = Instant::now();
    let extract_dir = out.join("extract");
    fs::create_dir_all(&extract_dir)?;
    let code_path = extract_dir.join("code.txt");
    let mut file = io::BufWriter::with_capacity(1 << 20, File::create(&code_path)?);
    let mut n = 0u64;
    let mut uniq = BTreeSet::new();
    let mut write_err: Option<io::Error> = None;
    let mut stats = extract_code(home, ExtractLimits::default(), |line| {
        if write_err.is_some() {
            return;
        }
        if let Err(err) = writeln!(file, "{line}") {
            write_err = Some(err);
            return;
        }
        n += 1;
        uniq.insert(line.to_string());
    });
    if let Some(err) = write_err {
        return Err(err);
    }
    file.flush()?;
    stats.set_int("lines", n);
    stats.set_int("unique_lines", uniq.len() as u64);
    stats.set_float("seconds", round1(t0.elapsed().as_secs_f64()));

    let counts_path = extract_dir.join("counts.json");
    let mut prev = if counts_path.exists() {
        let raw = fs::read_to_string(&counts_path)?;
        serde_json::from_str::<Value>(&raw).unwrap_or(Value::Object(Map::new()))
    } else {
        Value::Object(Map::new())
    };
    if let Value::Object(map) = &mut prev {
        map.insert("code".to_string(), Value::Object(stats.to_json_map()));
        map.insert(
            "until".to_string(),
            match until {
                Some(s) => Value::String(s.to_string()),
                None => Value::Null,
            },
        );
    }
    write_json_py(&counts_path, &prev)?;
    Ok(stats)
}

fn round1(x: f64) -> f64 {
    (x * 10.0).round() / 10.0
}

fn name_of(path: &Path) -> &str {
    path.file_name().and_then(|s| s.to_str()).unwrap_or("")
}

fn is_git_repo(dir: &Path) -> bool {
    let git = dir.join(".git");
    match fs::metadata(&git) {
        Ok(m) => m.is_dir() || m.is_file(),
        Err(_) => false,
    }
}

fn is_symlink(path: &Path) -> bool {
    fs::symlink_metadata(path)
        .map(|m| m.file_type().is_symlink())
        .unwrap_or(false)
}

fn dir_depth(home: &Path, dir: &Path) -> usize {
    if dir == home {
        return 0;
    }
    match dir.strip_prefix(home) {
        Ok(rel) => rel
            .components()
            .filter(|c| !matches!(c, Component::CurDir))
            .count(),
        Err(_) => 0,
    }
}

/// Top-down walk matching CPython 3.12 `os.walk(topdown=True, followlinks=False)`.
struct OsWalk {
    stack: Vec<PathBuf>,
}

impl OsWalk {
    fn new(top: PathBuf) -> Self {
        Self { stack: vec![top] }
    }

    fn next_dir(&mut self) -> Option<(PathBuf, Vec<String>, Vec<String>)> {
        while let Some(top) = self.stack.pop() {
            match scandir_split(&top) {
                Ok((dirs, files)) => return Some((top, dirs, files)),
                Err(_) => continue,
            }
        }
        None
    }

    fn push_dirs(&mut self, parent: &Path, dirs: &[String]) {
        for dirname in dirs.iter().rev() {
            let new_path = parent.join(dirname);
            if !is_symlink(&new_path) {
                self.stack.push(new_path);
            }
        }
    }
}

fn scandir_split(dir: &Path) -> io::Result<(Vec<String>, Vec<String>)> {
    let mut dirs = Vec::new();
    let mut files = Vec::new();
    let rd = fs::read_dir(dir)?;
    for entry in rd {
        let entry = match entry {
            Ok(e) => e,
            Err(_) => continue,
        };
        let name = match entry.file_name().into_string() {
            Ok(s) => s,
            Err(_) => continue,
        };
        if entry.path().is_dir() {
            dirs.push(name);
        } else {
            files.push(name);
        }
    }
    Ok((dirs, files))
}

/// `json.dump(..., indent=1, ensure_ascii=False)` without a trailing newline.
fn write_json_py(path: &Path, value: &Value) -> io::Result<()> {
    let mut tmp = path.as_os_str().to_os_string();
    tmp.push(".tmp");
    let tmp = PathBuf::from(tmp);
    let mut buf = String::new();
    dump_py(value, 0, &mut buf);
    fs::write(&tmp, buf)?;
    fs::rename(&tmp, path)?;
    Ok(())
}

fn dump_py(value: &Value, indent: usize, out: &mut String) {
    match value {
        Value::Null => out.push_str("null"),
        Value::Bool(true) => out.push_str("true"),
        Value::Bool(false) => out.push_str("false"),
        Value::Number(n) => {
            if n.is_f64() {
                let f = n.as_f64().unwrap_or(0.0);
                python_float(f, out);
            } else {
                out.push_str(&n.to_string());
            }
        }
        Value::String(s) => {
            out.push_str(&serde_json::to_string(s).unwrap_or_else(|_| "\"\"".into()))
        }
        Value::Array(arr) => {
            if arr.is_empty() {
                out.push_str("[]");
                return;
            }
            out.push_str("[\n");
            for (i, v) in arr.iter().enumerate() {
                for _ in 0..(indent + 1) {
                    out.push(' ');
                }
                dump_py(v, indent + 1, out);
                if i + 1 != arr.len() {
                    out.push(',');
                }
                out.push('\n');
            }
            for _ in 0..indent {
                out.push(' ');
            }
            out.push(']');
        }
        Value::Object(map) => {
            if map.is_empty() {
                out.push_str("{}");
                return;
            }
            out.push_str("{\n");
            for (i, (k, v)) in map.iter().enumerate() {
                for _ in 0..(indent + 1) {
                    out.push(' ');
                }
                out.push_str(&serde_json::to_string(k).unwrap_or_else(|_| "\"\"".into()));
                out.push_str(": ");
                dump_py(v, indent + 1, out);
                if i + 1 != map.len() {
                    out.push(',');
                }
                out.push('\n');
            }
            for _ in 0..indent {
                out.push(' ');
            }
            out.push('}');
        }
    }
}

fn python_float(f: f64, out: &mut String) {
    // Python json.dumps uses a decimal point for floats (`86.0`, `1.2`).
    let r = (f * 10.0).round() / 10.0;
    if (r - r.round()).abs() < 1e-12 {
        out.push_str(&format!("{r:.1}"));
    } else {
        out.push_str(&format!("{r}"));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU64, Ordering};

    static N: AtomicU64 = AtomicU64::new(0);

    struct Scratch(PathBuf);
    impl Drop for Scratch {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.0);
        }
    }

    fn scratch() -> Scratch {
        let id = N.fetch_add(1, Ordering::Relaxed);
        let p = std::env::temp_dir().join(format!(
            "phonon-mine-{}-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos())
                .unwrap_or(0),
            id
        ));
        fs::create_dir_all(&p).unwrap();
        Scratch(p)
    }

    fn plant_repo(path: &Path) {
        fs::create_dir_all(path).unwrap();
        fs::write(path.join(".git"), "gitdir: fake\n").unwrap();
    }

    fn write(path: &Path, body: &str) {
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent).unwrap();
        }
        fs::write(path, body).unwrap();
    }

    fn collect(home: &Path, limits: ExtractLimits) -> (Vec<String>, ExtractStats) {
        let mut lines = Vec::new();
        let stats = extract_code(home, limits, |l| lines.push(l.to_string()));
        (lines, stats)
    }

    #[test]
    fn find_repos_depth_library_dot_and_tree_skip() {
        let tmp = scratch();
        let home = &tmp.0;
        plant_repo(&home.join("alpha"));
        plant_repo(&home.join("Library").join("held"));
        plant_repo(&home.join(".dotdir").join("hidden"));
        plant_repo(&home.join("skip_venv").join("venv").join("inside"));
        plant_repo(&home.join("deep").join("d2").join("d3"));
        plant_repo(&home.join("deep").join("d2").join("d3not").join("d4"));
        plant_repo(&home.join("phonon"));

        let repos = find_repos(home, 3);
        let names: BTreeSet<String> = repos
            .iter()
            .map(|p| {
                p.strip_prefix(home)
                    .unwrap()
                    .to_string_lossy()
                    .replace('\\', "/")
            })
            .collect();
        assert!(names.contains("alpha"), "{names:?}");
        assert!(names.contains("phonon"), "{names:?}");
        assert!(names.contains("deep/d2/d3"), "{names:?}");
        assert!(!names.iter().any(|n| n.contains("held")), "{names:?}");
        assert!(!names.iter().any(|n| n.contains("hidden")), "{names:?}");
        assert!(!names.iter().any(|n| n.contains("inside")), "{names:?}");
        assert!(!names.iter().any(|n| n.contains("d4")), "{names:?}");
    }

    #[test]
    fn extract_skips_phonon_support_tree_and_dotfiles() {
        let tmp = scratch();
        let home = &tmp.0;
        plant_repo(&home.join("alpha"));
        write(
            &home.join("alpha/src/lib.rs"),
            "fn ident_alpha() { let x = 1; }\n",
        );
        write(
            &home.join("alpha/src/vendor.min.js"),
            "MINJS_ONLY_TOKEN zz\n",
        );
        write(&home.join("alpha/.hidden.rs"), "DOTFILE_ONLY_TOKEN zz\n");
        write(&home.join("alpha/target/skip.rs"), "TARGET_ONLY_TOKEN zz\n");
        write(
            &home.join("alpha/node_modules/pkg.js"),
            "NODEMOD_ONLY_TOKEN zz\n",
        );
        write(&home.join("alpha/build/out.py"), "BUILD_ONLY_TOKEN zz\n");
        plant_repo(&home.join("phonon"));
        write(
            &home.join("phonon/main.rs"),
            "fn excluded_phonon_token() {}\n",
        );
        plant_repo(
            &home
                .join("Library")
                .join("Application Support")
                .join("Phonon")
                .join("held"),
        );
        write(
            &home
                .join("Library")
                .join("Application Support")
                .join("Phonon")
                .join("held")
                .join("secret.py"),
            "HELD_SUPPORT_TOKEN zz\n",
        );

        let (lines, stats) = collect(home, ExtractLimits::default());
        let blob = lines.join("\n");
        assert!(blob.contains("ident_alpha"), "{blob}");
        assert!(!blob.contains("MINJS_ONLY_TOKEN"), "{blob}");
        assert!(!blob.contains("DOTFILE_ONLY_TOKEN"), "{blob}");
        assert!(!blob.contains("TARGET_ONLY_TOKEN"), "{blob}");
        assert!(!blob.contains("NODEMOD_ONLY_TOKEN"), "{blob}");
        assert!(!blob.contains("BUILD_ONLY_TOKEN"), "{blob}");
        assert!(!blob.contains("excluded_phonon_token"), "{blob}");
        assert!(!blob.contains("HELD_SUPPORT_TOKEN"), "{blob}");
        // find_repos still counts phonon; Library was not entered.
        assert_eq!(stats.get_int("code_repos"), Some(2));
        assert_eq!(stats.get_int("code_files"), Some(1));
    }

    #[test]
    fn per_repo_and_total_file_caps() {
        let tmp = scratch();
        let home = &tmp.0;
        plant_repo(&home.join("r1"));
        plant_repo(&home.join("r2"));
        for name in ["a.py", "b.py", "c.py"] {
            write(
                &home.join("r1").join(name),
                &format!("token_{} zz\n", name.chars().next().unwrap()),
            );
            write(
                &home.join("r2").join(name),
                &format!("other_{} zz\n", name.chars().next().unwrap()),
            );
        }
        let limits = ExtractLimits {
            max_bytes: CODE_MAX_BYTES,
            max_files_per_repo: 2,
            max_files: 120_000,
            max_repo_depth: 3,
        };
        let (lines, stats) = collect(home, limits);
        assert_eq!(lines.len(), 4, "{lines:?}");
        assert_eq!(stats.get_int("code_files"), Some(4));

        let limits = ExtractLimits {
            max_bytes: CODE_MAX_BYTES,
            max_files_per_repo: 4000,
            max_files: 3,
            max_repo_depth: 3,
        };
        let (lines, stats) = collect(home, limits);
        assert_eq!(lines.len(), 3, "{lines:?}");
        assert_eq!(stats.get_int("code_files"), Some(3));
    }

    #[test]
    fn too_big_files_count_and_are_skipped() {
        let tmp = scratch();
        let home = &tmp.0;
        plant_repo(&home.join("r"));
        write(&home.join("r/small.py"), "SMALL_OK_TOKEN zz\n");
        let big = home.join("r/big.py");
        let mut body = Vec::from(b"BIG_ONLY_TOKEN zz\n".as_slice());
        body.resize(400_001, b'x');
        fs::write(&big, body).unwrap();
        let exactly = home.join("r/exact.py");
        let mut body = Vec::from(b"EXACT_OK_TOKEN zz\n".as_slice());
        body.resize(400_000, b'x');
        fs::write(&exactly, body).unwrap();

        let (lines, stats) = collect(home, ExtractLimits::default());
        let blob = lines.join("\n");
        assert!(blob.contains("SMALL_OK_TOKEN"), "{blob}");
        assert!(blob.contains("EXACT_OK_TOKEN"), "{blob}");
        assert!(!blob.contains("BIG_ONLY_TOKEN"), "{blob}");
        assert_eq!(stats.get_int("code_files_too_big"), Some(1));
        assert_eq!(stats.get_int("code_files"), Some(2));
    }

    #[test]
    fn counts_json_python_indent_and_until_null() {
        let tmp = scratch();
        let out = tmp.0.join("out");
        plant_repo(&tmp.0.join("r"));
        write(&tmp.0.join("r/a.py"), "ab zz\n");
        write_extract_code(&tmp.0, &out, None).unwrap();
        let raw = fs::read_to_string(out.join("extract/counts.json")).unwrap();
        assert!(raw.starts_with("{\n \"code\": {\n"), "{raw}");
        assert!(raw.contains("\"until\": null"), "{raw}");
        assert!(!raw.ends_with('\n'), "{raw:?}");
        write_extract_code(&tmp.0, &out, Some("2026-08-30T00:00:00")).unwrap();
        let raw = fs::read_to_string(out.join("extract/counts.json")).unwrap();
        assert!(raw.contains("\"until\": \"2026-08-30T00:00:00\""), "{raw}");
    }
}
