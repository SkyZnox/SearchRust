#![doc = include_str!("../README.md")]

mod collections;
use std::time::Instant;
use rand::Rng;
use crate::collections::Collection;

/// Embedding size
/// Used for the size of the embeddings
const EMBEDDING_SIZE: usize = 768;

/// Top N results
/// Used for the display of the n higher rates
const TOP_N_RESULTS: usize = 10;

/// main
/// Main function used for the init of the collection and the dataset then for the search with the display of the result
fn main() {
    let time = Instant::now();
    let mut collection = Collection::new();
    for i in 0..1000000 {
        collection.add_embedding(uuid::Uuid::new_v4());
    }
    println!("Init time: {}", time.elapsed().as_millis());

    let time = Instant::now();
    let top_results = collection.search("example query");
    println!("Search time: {}", time.elapsed().as_millis());


    for (doc_id, similarity) in top_results.results {
        println!("Document ID: {:?}, Similarity: {}", doc_id, similarity);
    }
    println!("Collection length : {}", collection.get_length())
}