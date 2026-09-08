use std::io::{self, Write};
use std::path::PathBuf;
use std::process::ExitCode;

use anyhow::{bail, Context, Result};
use clap::{Parser, Subcommand};
use phonon_mine::{default_home, write_extract_code};

#[derive(Parser, Debug)]
#[command(
    name = "phonon-mine",
    version,
    about = "Phonon profile-miner extract stages"
)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand, Debug)]
enum Command {
    /// Identifiers from source files in git repos under $HOME.
    ///
    /// Reproduces `python -m profile_miner extract --source code` into
    /// `$OUT/extract/code.txt` and `$OUT/extract/counts.json`.
    ExtractCode {
        /// Miner output root (Python `$PHONON_MINER_OUT`).
        #[arg(long)]
        out: PathBuf,
        /// Stored in counts.json; code extract does not filter on mtime.
        #[arg(long)]
        until: Option<String>,
    },
}

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(err) => {
            eprintln!("phonon-mine: {err:#}");
            ExitCode::FAILURE
        }
    }
}

fn run() -> Result<()> {
    let cli = Cli::parse();
    match cli.command {
        Command::ExtractCode { out, until } => {
            let home = default_home().context("HOME is unset")?;
            if !home.is_dir() {
                bail!("HOME is not a directory: {}", home.display());
            }
            let out = expand_out(out)?;
            let stats = write_extract_code(&home, &out, until.as_deref())
                .with_context(|| format!("writing extract under {}", out.display()))?;
            let mut stderr = io::stderr().lock();
            write!(stderr, "[extract] code:")?;
            if let Some(n) = stats.get_int("lines") {
                write!(stderr, " {n} lines")?;
            }
            if let Some(n) = stats.get_int("unique_lines") {
                write!(stderr, " ({n} unique)")?;
            }
            if let Some((_, phonon_mine::StatVal::Float(s))) =
                stats.entries().iter().find(|(k, _)| k == "seconds")
            {
                write!(stderr, " in {s}s")?;
            }
            writeln!(stderr, " {:?}", stats.to_json_map())?;
            Ok(())
        }
    }
}

fn expand_out(p: PathBuf) -> Result<PathBuf> {
    let expanded = if let Some(s) = p.to_str() {
        if s == "~" {
            default_home().context("HOME is unset")?
        } else if let Some(rest) = s.strip_prefix("~/") {
            default_home().context("HOME is unset")?.join(rest)
        } else {
            p
        }
    } else {
        p
    };
    if expanded.is_absolute() {
        Ok(expanded)
    } else {
        Ok(std::env::current_dir()?.join(expanded))
    }
}
