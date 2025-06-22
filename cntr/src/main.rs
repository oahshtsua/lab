use anyhow::Result;
use clap::Parser;
use cntr::count_in_path;

/// Count the number of lines (or words) in the given files
#[derive(Parser)]
struct Args {
    /// Count words instead of lines
    #[arg(short, long)]
    words: bool,
    /// Files to process
    #[arg(required = true)]
    files: Vec<String>,
}

fn main() -> Result<()> {
    let args = Args::parse();

    for path in args.files {
        let count = count_in_path(&path)?;
        if args.words {
            println!("{}: {} words", path, count.words);
        } else {
            println!("{}: {} lines", path, count.lines);
        }
    }
    Ok(())
}
