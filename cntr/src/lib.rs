use anyhow::{Context, Result};

use std::fs::File;
use std::io::{BufRead, BufReader};

#[derive(Default)]
pub struct Count {
    pub words: usize,
    pub lines: usize,
}

fn count(mut input: impl BufRead) -> Result<Count> {
    let mut count = Count::default();
    let mut line = String::new();

    while input.read_line(&mut line)? > 0 {
        count.lines += 1;
        count.words += line.split_whitespace().count();
        line.clear();
    }
    Ok(count)
}

pub fn count_in_path(path: &String) -> Result<Count> {
    let file = File::open(path).with_context(|| path.clone())?;
    let file = BufReader::new(file);
    let count = count(file).with_context(|| path.clone())?;
    Ok(count)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::{BufReader, Cursor, Error, ErrorKind, Read, Result};

    struct ErrorReader;

    impl Read for ErrorReader {
        fn read(&mut self, _buf: &mut [u8]) -> Result<usize> {
            Err(Error::new(ErrorKind::Other, "oh no"))
        }
    }

    #[test]
    fn counts_words_and_lines_in_memory_input() {
        let input = Cursor::new("word1 word2\nword3");
        let count = count(input).unwrap();
        assert_eq!(count.lines, 2, "wrong line count");
        assert_eq!(count.words, 3, "wrong word count");
    }

    #[test]
    fn counts_words_and_lines_from_file() {
        let file = String::from("tests/data/dummy.txt");
        let count = count_in_path(&file).unwrap();

        assert_eq!(count.lines, 5, "wrong line count");
        assert_eq!(count.words, 10, "wrong word count");
    }

    #[test]
    fn returns_error_when_input_cannot_be_read() {
        let input = BufReader::new(ErrorReader);
        let result = count(input);
        assert!(result.is_err(), "no error returned")
    }
}
