use assert_cmd::Command;
use predicates::prelude::*;

#[test]
fn binary_errors_out_with_no_args() {
    Command::cargo_bin("cntr")
        .unwrap()
        .assert()
        .failure()
        .stderr(predicate::str::contains("Usage"));
}

#[test]
fn binary_counts_lines_for_multiple_files() {
    Command::cargo_bin("cntr")
        .unwrap()
        .args(["tests/data/dummy.txt", "tests/data/dummy2.txt"])
        .assert()
        .success()
        .stdout("tests/data/dummy.txt: 5 lines\ntests/data/dummy2.txt: 0 lines\n");
}

#[test]
fn binary_counts_words_for_files_with_word_flag() {
    Command::cargo_bin("cntr")
        .unwrap()
        .arg("-w")
        .args(["tests/data/dummy.txt", "tests/data/dummy2.txt"])
        .assert()
        .success()
        .stdout("tests/data/dummy.txt: 10 words\ntests/data/dummy2.txt: 0 words\n");
}
