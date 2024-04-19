use std::cmp::{Ordering, Reverse};
use std::collections::BinaryHeap;
use ndarray::{Array1, ArrayView1};

#[derive(PartialEq, Clone, Copy, Debug)]
pub struct SearchResult {
    pub id: usize,
    pub score: f32,
}

impl Eq for SearchResult {}

impl PartialOrd for SearchResult {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for SearchResult {
    fn cmp(&self, other: &SearchResult) -> Ordering {
        self.score.partial_cmp(&other.score).unwrap()
    }
}

/// creates a progress bar with the default template
pub fn new_progress_bar(name: &str, delta_refresh: usize, elems: usize) -> indicatif::ProgressBar {
    let pb = indicatif::ProgressBar::new(elems as u64);
    pb.set_draw_delta(delta_refresh as u64);
    let rest =
        "[{elapsed_precise}] [{bar:40.cyan/blue}] ({pos}/{len}, ETA {eta}, SPEED: {per_sec})";
    pb.set_style(indicatif::ProgressStyle::default_bar().template(&format!("{}: {}", name, rest)));
    pb
}

pub fn get_largest(scores: ArrayView1<f32>, k: usize) -> Array1<usize> {
    let mut heap: BinaryHeap<Reverse<SearchResult>> = BinaryHeap::new();
    let mut threshold = f32::MIN;
    scores.iter().enumerate().for_each(|(id, &score)| {
        if score > threshold {
            heap.push(Reverse(SearchResult { id, score }));
            if heap.len() > k {
                threshold = heap.pop().unwrap().0.score;
            }
        }
    });
    Array1::from(
        heap.into_sorted_vec()
            .iter()
            .map(|e| e.0.id)
            .collect::<Vec<_>>(),
    )
}